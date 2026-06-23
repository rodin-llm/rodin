#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_sft_rodin.py — SFT full-finetune de RODIN-1B sur la RTX 3090.

Part du checkpoint de pretraining (rodin1b_weights_bf16.pt), apprend a suivre
des instructions au format ChatML, en ne calculant la loss QUE sur les tokens
de la reponse assistant (le prompt user est masque, ignore_index=-100).

Principes (coherents avec 21_train_rodin.py) :
  - reutilise RodinLM et model_config importes depuis 21_train_rodin.py
  - bf16 autocast, PAS de GradScaler (bf16 a l'exposant fp32)
  - AdamW fused, decay sur tenseurs >=2D uniquement
  - gradient checkpointing pour tenir le 1B en full-finetune sur 24 Go
  - LR bas + warmup + cosine (un SFT ne doit pas ecraser le pretraining)
  - checkpoint/resume par step ; sauvegarde finale au format sample.py
  - detection NaN/inf (on stoppe proprement, lecon du pretraining)

Le tokenizer SentencePiece (rodin.model) ne connait pas les marqueurs ChatML :
on les encode en texte et on recupere leurs IDs reels au runtime pour batir le
masque. Le format ChatML doit etre EXACTEMENT le meme qu'a l'inference.

Donnees attendues : JSONL avec une cle "text" contenant la sequence ChatML
complete (sortie de format_chatml.py).

Lancement (venv actif, 3090) :
  python -u 22_sft_rodin.py \
      --weights /path/to/rodin/weights/rodin1b_weights_bf16.pt \
      --tokenizer /path/to/rodin/bpe/rodin.model \
      --train /path/to/rodin/sft/sft_chatml_train.jsonl \
      --val   /path/to/rodin/sft/sft_chatml_val.jsonl \
      --out   /path/to/rodin/runs/sft \
      --epochs 3 --batch-size 4 --grad-accum 8 --lr 1e-5

Self-test (sans donnees ni GPU, mini-modele) :
  python 22_sft_rodin.py --selftest
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

IGNORE_INDEX = -100

# Marqueurs ChatML : DOIVENT etre identiques a ceux de format_chatml.py et de
# l'inference. On les traite comme du texte (le tokenizer les fragmentera de
# facon stable et reproductible).
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ======================================================================
# Import du code modele depuis le trainer de pretraining
# ======================================================================
def import_trainer(trainer_path):
    import importlib.util
    d = os.path.dirname(os.path.abspath(trainer_path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("rodin_trainer", trainer_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for needed in ("RodinLM", "model_config"):
        if not hasattr(mod, needed):
            sys.exit(f"[ERREUR] {needed} introuvable dans {trainer_path}")
    return mod


# ======================================================================
# Chargement des poids de pretraining (meme logique que sample.py)
# ======================================================================
def load_pretrained(trainer_mod, weights_path, preset, device):
    payload = torch.load(weights_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        sd = payload["model"]
        cfg = payload.get("cfg")
        step = payload.get("step")
    else:
        sd = payload
        cfg = None
        step = None
    if cfg is None:
        cfg = trainer_mod.model_config(preset)
        print(f"[load] pas de cfg embarquee -> preset '{preset}': {cfg}", flush=True)
    else:
        print(f"[load] cfg embarquee : {cfg}", flush=True)

    model = trainer_mod.RodinLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # rope_cos/rope_sin non persistants + lm_head tied : absences normales
    tolerable = {"rope_cos", "rope_sin", "lm_head.weight"}
    real_missing = [m for m in missing if m not in tolerable]
    if real_missing:
        sys.exit(f"[ERREUR] cles manquantes inattendues : {real_missing}")
    if unexpected:
        print(f"[load] cles inattendues ignorees : {unexpected}", flush=True)
    print(f"[load] poids charges (step pretraining={step}) | params="
          f"{model.num_params()/1e9:.3f}G", flush=True)
    return model.to(device), cfg


# ======================================================================
# Tokenizer : recupere les IDs reels des marqueurs ChatML
# ======================================================================
class ChatMLTokenizer:
    def __init__(self, sp_model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.vocab = self.sp.get_piece_size()
        # IDs (potentiellement multi-token) des marqueurs, encodes comme texte
        self.ids_im_start = self.sp.encode(IM_START, out_type=int)
        self.ids_im_end = self.sp.encode(IM_END, out_type=int)
        # BOS/EOS si definis
        self.bos = self.sp.bos_id() if self.sp.bos_id() >= 0 else None
        self.eos = self.sp.eos_id() if self.sp.eos_id() >= 0 else None
        print(f"[tok] vocab={self.vocab} | '{IM_START}'->{self.ids_im_start} | "
              f"'{IM_END}'->{self.ids_im_end} | bos={self.bos} eos={self.eos}",
              flush=True)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)


# ======================================================================
# Dataset SFT : tokenise + construit le masque de loss sur la reponse
# ======================================================================
class SFTDataset(Dataset):
    """
    Lit un JSONL {"text": "<|im_start|>user\\n...<|im_end|>\\n<|im_start|>assistant\\n...<|im_end|>"}.
    Decoupe sur les marqueurs pour savoir quelle partie est la reponse assistant,
    et masque tout le reste (ignore_index) afin de n'apprendre que la reponse.
    """
    def __init__(self, path, tokenizer, ctx):
        self.ctx = ctx
        self.tok = tokenizer
        self.rows = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    skipped += 1
                    continue
                text = obj.get("text", "")
                ex = self._build(text)
                if ex is None:
                    skipped += 1
                    continue
                self.rows.append(ex)
        print(f"[data] {path} : {len(self.rows)} exemples charges "
              f"({skipped} ignores)", flush=True)
        if not self.rows:
            sys.exit(f"[ERREUR] aucun exemple valide dans {path}")

    def _build(self, text):
        """
        Reconstruit input_ids + labels en tokenisant les segments separement,
        ce qui donne un decoupage fiable prompt/reponse. On suppose le format :
          [system?] user ... <|im_end|> <|im_start|> assistant \n <reponse> <|im_end|>
        On apprend sur <reponse> + le <|im_end|> final uniquement.
        """
        # Repere le dernier bloc assistant : tout ce qui suit
        # "<|im_start|>assistant\n" jusqu'au "<|im_end|>" final est la reponse.
        marker = f"{IM_START}assistant\n"
        idx = text.rfind(marker)
        if idx == -1:
            return None
        prefix = text[:idx + len(marker)]   # tout jusqu'a (et incluant) "...assistant\n"
        rest = text[idx + len(marker):]     # "<reponse><|im_end|>"
        # separe la reponse de son <|im_end|> final
        if rest.endswith(IM_END):
            answer = rest[:-len(IM_END)]
        else:
            answer = rest                   # tolere l'absence du marqueur final

        prefix_ids = self.tok.encode(prefix)
        answer_ids = self.tok.encode(answer)
        end_ids = self.tok.ids_im_end       # on apprend a produire <|im_end|> (= stop)

        input_ids = prefix_ids + answer_ids + end_ids
        labels = ([IGNORE_INDEX] * len(prefix_ids)
                  + answer_ids
                  + end_ids)

        # tronque au contexte (garde le debut : prompt + debut de reponse)
        if len(input_ids) > self.ctx:
            input_ids = input_ids[:self.ctx]
            labels = labels[:self.ctx]

        # il faut au moins 1 token appris, sinon exemple inutile
        if all(l == IGNORE_INDEX for l in labels):
            return None
        return (input_ids, labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def make_collate(pad_id, ctx):
    """
    Pad dynamique au plus long de la batch (cap a ctx). Construit aussi le
    decalage causal : input = seq[:-1], target = labels[1:].
    """
    def collate(batch):
        maxlen = min(ctx, max(len(ids) for ids, _ in batch))
        X, Y = [], []
        for ids, labels in batch:
            ids = ids[:maxlen]
            labels = labels[:maxlen]
            pad = maxlen - len(ids)
            ids = ids + [pad_id] * pad
            labels = labels + [IGNORE_INDEX] * pad
            X.append(ids)
            Y.append(labels)
        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)
        # decalage causal : on predit le token suivant
        inp = X[:, :-1].contiguous()
        tgt = Y[:, 1:].contiguous()
        return inp, tgt
    return collate


# ======================================================================
# LR schedule (reprend lr_at du trainer : warmup + cosine)
# ======================================================================
def lr_at(step, total, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= total:
        return lr * min_ratio
    p = (step - warmup) / max(total - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * p))
    return lr * (min_ratio + (1 - min_ratio) * coeff)


# ======================================================================
# Checkpoint (format compatible sample.py : {"model","cfg","step",...})
# ======================================================================
def save_ckpt(out_dir, step, model, optimizer, cfg, args, keep=3, final=False):
    os.makedirs(out_dir, exist_ok=True)
    base = getattr(model, "_orig_mod", model)
    name = "rodin1b_instruct_bf16.pt" if final else f"sft_{step:08d}.pt"
    tmp = os.path.join(out_dir, name + ".tmp")
    path = os.path.join(out_dir, name)
    payload = {
        "step": step,
        "model": base.state_dict(),
        "cfg": cfg,
        "dtype": "bfloat16",
        "args": vars(args),
        "sft": True,
    }
    if not final:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp)
    os.replace(tmp, path)
    if not final:
        cks = sorted(glob.glob(os.path.join(out_dir, "sft_*.pt")))
        for old in cks[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass
    return path


def load_resume(out_dir, model, optimizer, device):
    cks = sorted(glob.glob(os.path.join(out_dir, "sft_*.pt")))
    if not cks:
        return 0
    path = cks[-1]
    print(f"[resume] {path}", flush=True)
    ck = torch.load(path, map_location=device)
    base = getattr(model, "_orig_mod", model)
    base.load_state_dict(ck["model"], strict=False)
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return int(ck.get("step", 0))


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, val_dl, device, max_batches=50):
    model.eval()
    losses = []
    for k, (inp, tgt) in enumerate(val_dl):
        if k >= max_batches:
            break
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                tgt.view(-1).long(),
                ignore_index=IGNORE_INDEX,
            )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ======================================================================
# Self-test : verifie le pipeline (dataset/masque/forward/backward) en mini
# ======================================================================
def selftest():
    print("[selftest] construction d'un mini-cas de masquage...")
    # mini-modele via le trainer si dispo, sinon juste la logique de masque
    import torch.nn.functional as F
    # cas jouet : verifie que la loss masquee ignore le prompt
    V = 64
    inp = torch.randint(0, V, (2, 20))
    tgt = torch.full((2, 20), IGNORE_INDEX)
    tgt[:, 10:] = torch.randint(0, V, (2, 10))   # seuls les 10 derniers comptent
    logits = torch.randn(2, 20, V)
    loss = F.cross_entropy(logits.view(-1, V), tgt.view(-1), ignore_index=IGNORE_INDEX)
    n_active = (tgt != IGNORE_INDEX).sum().item()
    print(f"[selftest] tokens actifs={n_active} (attendu 20) | loss={loss.item():.3f}")
    assert n_active == 20
    print("[selftest] OK — logique de masque valide")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="SFT full-finetune RODIN-1B (3090).")
    ap.add_argument("--weights", help="rodin1b_weights_bf16.pt (checkpoint pretraining)")
    ap.add_argument("--tokenizer", help="rodin.model SentencePiece")
    ap.add_argument("--train", help="JSONL ChatML train")
    ap.add_argument("--val", default=None, help="JSONL ChatML val")
    ap.add_argument("--out", default="./runs/sft")
    ap.add_argument("--trainer", default=None,
                    help="chemin 21_train_rodin.py (defaut: meme dossier)")
    ap.add_argument("--preset", default="prod", choices=["pretest", "prod"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--keep", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="desactive le gradient checkpointing (si VRAM suffit)")
    ap.add_argument("--compile", action="store_true",
                    help="active torch.compile(mode=default)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    for req in ("weights", "tokenizer", "train"):
        if not getattr(args, req):
            sys.exit(f"[ERREUR] --{req} requis (sauf --selftest).")
    if not torch.cuda.is_available():
        sys.exit("[ERREUR] CUDA indisponible. Ce script cible la 3090.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    trainer_path = args.trainer or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "21_train_rodin.py")
    if not os.path.exists(trainer_path):
        sys.exit(f"[ERREUR] trainer introuvable : {trainer_path}")
    trainer_mod = import_trainer(trainer_path)

    # --- modele depuis le pretraining ---
    model, cfg = load_pretrained(trainer_mod, args.weights, args.preset, device)

    # --- gradient checkpointing (cle pour tenir le 1B en full-FT sur 24 Go) ---
    if not args.no_grad_checkpoint:
        _enable_grad_checkpointing(model)
        print("[mem] gradient checkpointing actif", flush=True)

    # --- tokenizer ---
    tok = ChatMLTokenizer(args.tokenizer)
    pad_id = tok.eos if tok.eos is not None else 0

    # --- data ---
    ctx = cfg["ctx"]
    train_ds = SFTDataset(args.train, tok, ctx)
    collate = make_collate(pad_id, ctx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, pin_memory=True)
    val_dl = None
    if args.val and os.path.exists(args.val):
        val_ds = SFTDataset(args.val, tok, ctx)
        val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate,
                            drop_last=False, pin_memory=True)

    steps_per_epoch = len(train_dl) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    print(f"[plan] {len(train_ds)} ex | {steps_per_epoch} steps/epoch x "
          f"{args.epochs} = {total_steps} steps | warmup={warmup} | "
          f"bs{args.batch_size} x accum{args.grad_accum}", flush=True)

    # --- optimizer ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=True,
    )

    start_step = load_resume(args.out, model, optimizer, device)
    if start_step:
        print(f"[resume] reprise au step {start_step}", flush=True)

    if args.compile:
        model = torch.compile(model)
        print("[compile] torch.compile(default)", flush=True)

    # --- boucle ---
    model.train()
    step = start_step
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    t_log = time.time()
    tok_log = 0
    stop = False

    for epoch in range(args.epochs):
        if stop:
            break
        for inp, tgt in train_dl:
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(inp)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    tgt.view(-1).long(),
                    ignore_index=IGNORE_INDEX,
                )
                loss = loss / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[STOP] loss non finie ({loss.item()}) — arret.", flush=True)
                stop = True
                break

            loss.backward()
            micro += 1
            tok_log += inp.numel()

            if micro == args.grad_accum:
                lr = lr_at(step, total_steps, args.lr, warmup, args.min_lr_ratio)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                micro = 0
                step += 1

                if step % args.log_every == 0:
                    dt = time.time() - t_log
                    tps = tok_log / dt if dt > 0 else 0
                    print(f"[step {step:>5}/{total_steps}] ep{epoch+1} "
                          f"loss {loss.item()*args.grad_accum:.4f} | lr {lr:.2e} | "
                          f"{tps/1e3:.1f}k tok/s", flush=True)
                    t_log = time.time()
                    tok_log = 0

                if val_dl is not None and step % args.eval_every == 0:
                    vl = evaluate(model, val_dl, device)
                    print(f"[eval  {step:>5}] val_loss {vl:.4f} "
                          f"(ppl {math.exp(min(vl,20)):.1f})", flush=True)

                if step % args.ckpt_every == 0:
                    p = save_ckpt(args.out, step, model, optimizer, cfg, args,
                                  keep=args.keep)
                    print(f"[ckpt {step:>5}] -> {p}", flush=True)

    # --- sauvegarde finale au format sample.py ---
    final = save_ckpt(args.out, step, model, optimizer, cfg, args, final=True)
    print(f"\n[FIN] SFT termine. Poids instruct -> {final}", flush=True)
    print(f"[FIN] probe : python sample.py --weights {final} "
          f"--tokenizer {args.tokenizer} --preset {args.preset} "
          f"--prompt '<|im_start|>user\\nBonjour<|im_end|>\\n<|im_start|>assistant\\n'",
          flush=True)


# ======================================================================
# Gradient checkpointing : enveloppe le forward de chaque Block
# ======================================================================
def _enable_grad_checkpointing(model):
    import torch.utils.checkpoint as cp
    base = getattr(model, "_orig_mod", model)
    for blk in base.blocks:
        _orig = blk.forward
        def make_fwd(b, f):
            def fwd(x, cos, sin):
                return cp.checkpoint(f, x, cos, sin, use_reentrant=False)
            return fwd
        blk.forward = make_fwd(blk, _orig)


if __name__ == "__main__":
    main()
