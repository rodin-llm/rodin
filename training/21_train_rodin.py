# 21_train_rodin.py
# Trainer minimal RODIN pour la RTX 3090 (PRETEST) et, identique, pour le run cloud.
# Objectif PRETEST : voir le LOSS DESCENDRE sur train.bin/val.bin du blend 400M,
# et valider dataloader + checkpoint/resume + courbe de loss. Ce n'est PAS le
# modele final (le 1B reel se fait au run cloud) : ici ~410M params, plus petit.
#
# Decisions figees (handoff) :
#   - modele LLaMA-style : RoPE, RMSNorm, SwiGLU, ctx 2048,
#   - bf16 (autocast, PAS de GradScaler : bf16 a l'exposant fp32, inutile),
#   - checkpoint/resume toutes les 500 steps, garder les 5 derniers,
#   - num_workers=4, prefetch_factor=2 (via rodin_data.make_dataloader),
#   - resume = un simple offset (index de fenetre) -> stocke dans le checkpoint,
#   - zero dependance cloud, zero dependance au reste du repo (sauf rodin_data.py).
#
# torch.compile : par defaut mode="default" sur Ampere (sur). reduce-overhead
# (CUDA graphs) est LE levier MFU a tester -> flag --compile-mode.
#
# Lancement pretest (sur la 3090, venv actif) :
#   python -u 21_train_rodin.py \
#       --train /data/rodin/blend/train.bin \
#       --val   /data/rodin/blend/val.bin \
#       --out   /data/rodin/runs/pretest \
#       --max-steps 6000 --batch-size 8 --grad-accum 4 \
#       --compile-mode default
#
# Reprise (apres crash / watchdog) : RELANCER LA MEME COMMANDE. Le trainer
# detecte le dernier checkpoint dans --out et reprend a l'offset exact.
#
# Self-test rapide sans GPU/donnees (forward/backward sur tenseurs aleatoires) :
#   python 21_train_rodin.py --selftest

import argparse
import json
import math
import os
import sys
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# CONFIG MODELE (PRETEST ~410M). Le run cloud 1B se fait via un autre preset
# (n'edite que ces champs ; ctx/vocab restent figes).
# ======================================================================
def model_config(preset):
    if preset == "pretest":      # ~410M params
        return dict(dim=1024, n_layers=28, n_heads=16, n_kv_heads=16,
                    ffn_hidden=2730, vocab=64000, ctx=2048,
                    rope_theta=10000.0, rmsnorm_eps=1e-5)
    if preset == "prod":         # ~1B params (RODIN-1B, pour le run cloud)
        return dict(dim=2048, n_layers=22, n_heads=16, n_kv_heads=16,
                    ffn_hidden=5461, vocab=64000, ctx=2048,
                    rope_theta=10000.0, rmsnorm_eps=1e-5)
    raise ValueError(f"preset inconnu : {preset}")


# ======================================================================
# Briques LLaMA-style
# ======================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # calcul en fp32 pour la stabilite, retour au dtype d'entree
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dt)) * self.weight


def precompute_rope(head_dim, max_seq, theta, device):
    """Retourne cos, sin de forme (max_seq, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)                 # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)          # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k : (B, n_heads, T, head_dim) ; cos,sin : (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)              # (1,1,T,hd)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg["dim"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.head_dim = self.dim // self.n_heads
        assert self.dim % self.n_heads == 0, "dim doit etre divisible par n_heads"
        assert self.n_heads % self.n_kv_heads == 0, \
            "n_heads doit etre un multiple de n_kv_heads (GQA)"
        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.n_rep = self.n_heads // self.n_kv_heads

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        if self.n_rep > 1:                            # GQA : repete K,V
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        # SDPA : utilise FlashAttention/efficient backend si dispo, causal
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden = cfg["dim"], cfg["ffn_hidden"]
        self.w1 = nn.Linear(dim, hidden, bias=False)     # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)     # up
        self.w2 = nn.Linear(hidden, dim, bias=False)     # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["dim"], cfg["rmsnorm_eps"])
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg["dim"], cfg["rmsnorm_eps"])
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class RodinLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab"], cfg["dim"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["dim"], cfg["rmsnorm_eps"])
        self.lm_head = nn.Linear(cfg["dim"], cfg["vocab"], bias=False)
        self.lm_head.weight = self.tok_emb.weight     # weight tying
        head_dim = cfg["dim"] // cfg["n_heads"]
        cos, sin = precompute_rope(head_dim, cfg["ctx"], cfg["rope_theta"],
                                   device="cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                targets.view(-1).long(),
            )
        return logits, loss

    def num_params(self, non_embedding=False):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()   # tied -> compte une fois en moins
        return n


# ======================================================================
# Checkpoint / resume
# ======================================================================
def list_checkpoints(out_dir):
    paths = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    def step_of(p):
        try:
            return int(os.path.basename(p).split("_")[1].split(".")[0])
        except Exception:
            return -1
    return sorted(paths, key=step_of)


def save_checkpoint(out_dir, step, model, optimizer, window_offset, cfg, args,
                    keep=5):
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, f"ckpt_{step:08d}.pt.tmp")
    final = os.path.join(out_dir, f"ckpt_{step:08d}.pt")
    payload = {
        "step": step,
        "window_offset": window_offset,    # index de fenetre pour resume exact
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "args": vars(args),
    }
    torch.save(payload, tmp)
    os.replace(tmp, final)
    # rotation : ne garder que les `keep` derniers
    cks = list_checkpoints(out_dir)
    for old in cks[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass
    return final


def load_latest_checkpoint(out_dir, model, optimizer, device):
    cks = list_checkpoints(out_dir)
    if not cks:
        return 0, 0
    path = cks[-1]
    print(f"[resume] chargement {path}")
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer") is not None:
        optimizer.load_state_dict(ck["optimizer"])
    return int(ck["step"]), int(ck.get("window_offset", 0))


# ======================================================================
# LR schedule : warmup lineaire + cosine decay
# ======================================================================
def lr_at(step, max_steps, lr, warmup, min_ratio=0.1):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return lr * min_ratio
    p = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * p))
    return lr * (min_ratio + (1 - min_ratio) * coeff)


# ======================================================================
# Evaluation
# ======================================================================
@torch.no_grad()
def evaluate(model, val_dl, device, max_batches=50):
    model.eval()
    losses = []
    for k, (x, y) in enumerate(val_dl):
        if k >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ======================================================================
# Self-test (sans GPU obligatoire, sans donnees) : verifie forward/backward
# et le compte de parametres du preset pretest.
# ======================================================================
def selftest():
    cfg = model_config("pretest")
    cfg = dict(cfg, n_layers=2, ctx=128)          # mini pour le test
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RodinLM(cfg).to(dev)
    x = torch.randint(0, cfg["vocab"], (2, cfg["ctx"]), device=dev)
    y = torch.randint(0, cfg["vocab"], (2, cfg["ctx"]), device=dev)
    _, loss = model(x, y)
    loss.backward()
    print(f"[selftest] device={dev}  loss={loss.item():.3f}  "
          f"(attendu ~{math.log(cfg['vocab']):.2f} a l'init)")
    full = RodinLM(model_config('pretest'))
    print(f"[selftest] params preset pretest : "
          f"{full.num_params()/1e6:.1f}M total, "
          f"{full.num_params(non_embedding=True)/1e6:.1f}M non-embedding")
    print("[selftest] OK")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", help="chemin train.bin")
    ap.add_argument("--val", default=None, help="chemin val.bin (optionnel)")
    ap.add_argument("--out", default="./runs/pretest", help="dossier checkpoints")
    ap.add_argument("--preset", default="pretest", choices=["pretest", "prod"])
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--keep", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--compile-mode", default="default",
                    choices=["none", "default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if not args.train:
        sys.exit("[ERREUR] --train requis (sauf --selftest).")
    if not torch.cuda.is_available():
        sys.exit("[ERREUR] CUDA indisponible. Ce trainer cible la 3090.")

    import rodin_data    # import tardif : selftest ne depend pas du dataloader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"

    cfg = model_config(args.preset)
    print(f"[cfg] preset={args.preset}  {cfg}")

    # --- modele ---
    model = RodinLM(cfg).to(device)
    n_tot = model.num_params() / 1e6
    n_ne = model.num_params(non_embedding=True) / 1e6
    print(f"[model] {n_tot:.1f}M params ({n_ne:.1f}M non-embedding)")

    # --- optimizer : AdamW, decay seulement sur tenseurs >=2D ---
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

    # --- resume ---
    start_step, window_offset = load_latest_checkpoint(
        args.out, model, optimizer, device)
    if start_step > 0:
        print(f"[resume] reprise au step {start_step}, "
              f"window_offset {window_offset:,}")

    # --- compile ---
    if args.compile_mode != "none":
        mode = None if args.compile_mode == "default" else args.compile_mode
        print(f"[compile] torch.compile(mode={args.compile_mode})")
        model = torch.compile(model, mode=mode)

    # --- data ---
    _, train_dl = rodin_data.make_dataloader(
        args.train, ctx=cfg["ctx"], batch_size=args.batch_size,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        pin_memory=True, shuffle=False, start_index=window_offset,
        drop_last=True, seed=args.seed)
    val_dl = None
    if args.val:
        _, val_dl = rodin_data.make_dataloader(
            args.val, ctx=cfg["ctx"], batch_size=args.batch_size,
            num_workers=2, prefetch_factor=2, pin_memory=True,
            shuffle=False, drop_last=True, seed=args.seed)

    tokens_per_step = (args.batch_size * args.grad_accum * cfg["ctx"])
    print(f"[train] {tokens_per_step:,} tokens/step "
          f"(bs {args.batch_size} x accum {args.grad_accum} x ctx {cfg['ctx']})")

    # --- boucle ---
    model.train()
    step = start_step
    micro = 0
    t_log = time.time()
    tok_log = 0
    optimizer.zero_grad(set_to_none=True)
    data_iter = iter(train_dl)

    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            # epoch terminee : on relance depuis le debut (rare au pretest)
            print("[data] fin du flux, relance depuis le debut")
            _, train_dl = rodin_data.make_dataloader(
                args.train, ctx=cfg["ctx"], batch_size=args.batch_size,
                num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                pin_memory=True, shuffle=False, start_index=0,
                drop_last=True, seed=args.seed)
            data_iter = iter(train_dl)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss = loss / args.grad_accum
        loss.backward()
        micro += 1
        window_offset += args.batch_size      # fenetres consommees
        tok_log += args.batch_size * cfg["ctx"]

        if micro == args.grad_accum:
            lr = lr_at(step, args.max_steps, args.lr, args.warmup,
                       args.min_lr_ratio)
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
                print(f"[step {step:>6}/{args.max_steps}] "
                      f"loss {loss.item()*args.grad_accum:.4f} | "
                      f"lr {lr:.2e} | {tps/1e3:.1f}k tok/s | "
                      f"off {window_offset:,}", flush=True)
                t_log = time.time()
                tok_log = 0

            if val_dl is not None and step % args.eval_every == 0:
                vloss = evaluate(model, val_dl, device)
                print(f"[eval  {step:>6}] val_loss {vloss:.4f} "
                      f"(ppl {math.exp(min(vloss,20)):.1f})", flush=True)

            if step % args.ckpt_every == 0 or step == args.max_steps:
                # sauver le modele NON compile (state_dict propre)
                base = getattr(model, "_orig_mod", model)
                p = save_checkpoint(args.out, step, base, optimizer,
                                    window_offset, cfg, args, keep=args.keep)
                print(f"[ckpt {step:>6}] -> {p}", flush=True)

    print("[FIN] entrainement termine.")


if __name__ == "__main__":
    main()
