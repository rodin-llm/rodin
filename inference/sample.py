# sample.py
# ======================================================================
# Sonde de COMPLETION BRUTE pour RODIN-1B (modele BASE, pas instruct).
#
# But : charger rodin1b_weights_bf16.pt (poids seuls, exporte par
# 21_train_rodin.py / export_weights_only) et generer du texte a partir
# d'amorces, pour DRESSER LA CARTE des competences ancrees dans le
# pretraining AVANT de choisir les taches du SFT.
#
# IMPORTANT : c'est un modele BASE. Il COMPLETE, il ne REPOND pas. Ne lui
# pose pas de questions facon chat -> donne-lui des AMORCES a continuer.
#   BON   : "La capitale de la France est"
#   MAUVAIS : "Quelle est la capitale de la France ?"  (il enchainera
#             d'autres questions, c'est normal, le format chat viendra au SFT)
#
# Le script importe la VRAIE classe modele depuis 21_train_rodin.py (meme
# dossier) : aucune reimplementation d'archi -> zero risque de mismatch de
# state_dict. Il instancie au preset "prod" (RODIN-1B, dim=2048, 22 couches)
# qui est celui du run cloud.
#
# Le state_dict sauve est PROPRE (le trainer sauve getattr(model,'_orig_mod')),
# donc pas de prefixe torch.compile a stripper. On gere quand meme le cas par
# securite (strip de '_orig_mod.' si jamais present).
#
# ----------------------------------------------------------------------
# UTILISATION (sur la 3090, venv actif) :
#
#   # Mode interactif (tape une amorce, Entree, regarde la completion ;
#   # ligne vide ou Ctrl-D pour quitter) :
#   python sample.py \
#       --weights /data/rodin/rodin1b_weights_bf16.pt \
#       --tokenizer /data/rodin/tokenizer/rodin_bpe_64k.model
#
#   # Une amorce unique en argument :
#   python sample.py --weights ... --tokenizer ... \
#       --prompt "La capitale de la France est"
#
#   # Un BATCH d'amorces depuis un fichier (une amorce par ligne) :
#   python sample.py --weights ... --tokenizer ... \
#       --prompts-file probes.txt
#
#   # Voir la DISTRIBUTION (top-k probas) a chaque token genere -> pour
#   # VOIR de tes yeux le "savoir + le de" dont on a parle :
#   python sample.py --weights ... --tokenizer ... \
#       --prompt "Le ciel est" --show-probs --topk-display 5
#
# Parametres d'echantillonnage :
#   --temperature 0.8   (0 = greedy/deterministe ; >1 = plus aleatoire)
#   --top-k 40          (0 = desactive)
#   --top-p 0.95        (1.0 = desactive)
#   --max-new 120       (tokens a generer)
#   --seed 1234         (reproductibilite)
#
# Greedy pur (la sortie la plus probable, deterministe) :
#   python sample.py ... --prompt "..." --temperature 0
# ======================================================================

import argparse
import os
import sys
import importlib.util

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Import de la classe modele depuis 21_train_rodin.py (meme dossier que ce
# script, ou chemin passe via --trainer). Le nom de fichier commence par un
# chiffre -> import classique impossible, on charge par spec.
# ----------------------------------------------------------------------
def load_trainer_module(trainer_path):
    if not os.path.isfile(trainer_path):
        sys.exit(
            f"[fatal] trainer introuvable : {trainer_path}\n"
            f"        Place sample.py a cote de 21_train_rodin.py, ou passe "
            f"--trainer /chemin/vers/21_train_rodin.py"
        )
    spec = importlib.util.spec_from_file_location("rodin_trainer", trainer_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        # 21_train_rodin.py ne fait rien au niveau module (tout est sous
        # if __name__=='__main__'), donc l'import ne doit pas declencher
        # l'argparse. Si ca arrive, on le signale proprement.
        raise
    for needed in ("RodinLM", "model_config"):
        if not hasattr(mod, needed):
            sys.exit(f"[fatal] {trainer_path} n'expose pas '{needed}'. "
                     f"Mauvais fichier ?")
    return mod


# ----------------------------------------------------------------------
# Chargement des poids
# ----------------------------------------------------------------------
def load_model(trainer_mod, weights_path, preset, device):
    if not os.path.isfile(weights_path):
        sys.exit(f"[fatal] poids introuvables : {weights_path}")

    print(f"[load] lecture {weights_path} ...", flush=True)
    payload = torch.load(weights_path, map_location="cpu")

    # Le payload exporte est {"model": sd, "cfg": ..., "step": ..., ...}.
    # On tolere aussi un state_dict brut (torch.save(model.state_dict())).
    if isinstance(payload, dict) and "model" in payload:
        sd = payload["model"]
        cfg = payload.get("cfg")
        step = payload.get("step")
        dtype = payload.get("dtype", "?")
    else:
        sd = payload
        cfg = None
        step = None
        dtype = "?"

    # Config : on prend celle embarquee dans le fichier si presente (source
    # de verite : c'est la config exacte du run qui a produit les poids),
    # sinon on retombe sur le preset demande.
    if cfg is None:
        cfg = trainer_mod.model_config(preset)
        print(f"[load] pas de cfg dans le fichier -> preset '{preset}' : {cfg}",
              flush=True)
    else:
        print(f"[load] cfg embarquee dans le fichier : {cfg}", flush=True)

    # Strip defensif du prefixe torch.compile (normalement absent : le
    # trainer sauve deja le modele non compile).
    n_stripped = 0
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("_orig_mod."):
            new_sd[k[len("_orig_mod."):]] = v
            n_stripped += 1
        else:
            new_sd[k] = v
    if n_stripped:
        print(f"[load] prefixe '_orig_mod.' retire sur {n_stripped} cles.",
              flush=True)
    sd = new_sd

    model = trainer_mod.RodinLM(cfg)

    # weight tying : lm_head.weight EST tok_emb.weight (meme tenseur). Le
    # state_dict peut ne contenir qu'une des deux cles. On charge en
    # non-strict puis on verifie qu'il ne manque que des cles liees/buffers.
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # rope_cos / rope_sin sont des buffers non persistants (persistent=False)
    # -> absents du state_dict, recalcules a l'init du modele. Normal qu'ils
    # soient "missing". lm_head.weight aussi peut etre missing (weight tying).
    tolerable_missing = {"rope_cos", "rope_sin", "lm_head.weight"}
    real_missing = [m for m in missing
                    if not any(m.endswith(t) for t in tolerable_missing)]
    if real_missing:
        sys.exit(f"[fatal] cles manquantes inattendues au chargement : "
                 f"{real_missing}\n        -> mismatch d'archi / mauvais preset ?")
    if unexpected:
        print(f"[load] WARN cles inattendues ignorees : {unexpected}", flush=True)

    # Re-tie par securite (si lm_head.weight etait manquant, il pointe deja
    # sur tok_emb.weight par construction ; on le reaffirme).
    model.lm_head.weight = model.tok_emb.weight

    model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_unique = model.num_params() if hasattr(model, "num_params") else n_params
    print(f"[load] OK. params (comptage brut) : {n_params/1e9:.3f} G  "
          f"| step={step} | dtype_fichier={dtype} | device={device}", flush=True)
    return model, cfg


# ----------------------------------------------------------------------
# Tokenizer SentencePiece
# ----------------------------------------------------------------------
def load_tokenizer(tok_path):
    if not os.path.isfile(tok_path):
        sys.exit(f"[fatal] tokenizer introuvable : {tok_path}")
    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("[fatal] sentencepiece non installe dans ce venv "
                 "(pip install sentencepiece).")
    sp = spm.SentencePieceProcessor()
    sp.load(tok_path)
    print(f"[tok] charge : {tok_path} | vocab={sp.get_piece_size()}", flush=True)
    return sp


# ----------------------------------------------------------------------
# Filtrage top-k / top-p sur les logits du DERNIER token.
# logits : (vocab,) sur device. Retourne logits filtres (meme forme).
# ----------------------------------------------------------------------
def filter_logits(logits, top_k, top_p):
    logits = logits.clone()

    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # on garde le plus petit ensemble dont la masse cumulee >= top_p
        remove = cum > top_p
        # decalage : on garde toujours au moins le 1er token
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        idx_remove = sorted_idx[remove]
        logits[idx_remove] = float("-inf")

    return logits


# ----------------------------------------------------------------------
# Affichage de la distribution (top-k) du prochain token : pour VOIR
# la proba apprise avant le tirage. Ne modifie rien, lecture seule.
# ----------------------------------------------------------------------
def print_distribution(logits, sp, topk_display):
    probs = F.softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(probs, min(topk_display, probs.size(-1)))
    parts = []
    for p, i in zip(vals.tolist(), idx.tolist()):
        piece = sp.id_to_piece(int(i))
        # SentencePiece marque le debut de mot par U+2581 ('_'). On le rend
        # lisible : on remplace par un point median visible.
        piece = piece.replace("\u2581", "·")
        parts.append(f"{piece!r}={p*100:4.1f}%")
    print("        top: " + "  ".join(parts), flush=True)


# ----------------------------------------------------------------------
# Generation autoregressive d'une amorce.
# ----------------------------------------------------------------------
@torch.no_grad()
def generate(model, sp, cfg, prompt, args, device):
    ctx = cfg["ctx"]

    ids = sp.encode(prompt, out_type=int)
    if len(ids) == 0:
        # certains tokenizers renvoient vide sur chaine vide -> on amorce
        # avec BOS si dispo, sinon on refuse.
        bos = sp.bos_id()
        if bos is not None and bos >= 0:
            ids = [bos]
        else:
            print("[gen] amorce vide, rien a generer.", flush=True)
            return prompt

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    eos = sp.eos_id()

    generated_ids = []
    for step in range(args.max_new):
        # fenetre glissante : on ne donne jamais plus que ctx tokens
        idx_cond = idx[:, -ctx:]
        logits, _ = model(idx_cond)              # (B, T, vocab)
        last = logits[0, -1, :].float()          # (vocab,)

        if args.show_probs:
            # distribution AVANT tout filtrage/temperature -> la vraie
            # croyance du modele
            print_distribution(last, sp, args.topk_display)

        if args.temperature is not None and args.temperature <= 0:
            # greedy : argmax, deterministe
            next_id = int(torch.argmax(last))
        else:
            temp = args.temperature if args.temperature else 1.0
            scaled = last / temp
            scaled = filter_logits(scaled, args.top_k, args.top_p)
            probs = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1))

        generated_ids.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)

        if eos is not None and eos >= 0 and next_id == eos:
            break

    completion = sp.decode(generated_ids)
    return prompt, completion


def run_one(model, sp, cfg, prompt, args, device):
    prompt = prompt.rstrip("\n")
    if not prompt.strip():
        return
    res = generate(model, sp, cfg, prompt, args, device)
    if isinstance(res, tuple):
        p, completion = res
        print("\n" + "=" * 70, flush=True)
        print(f"AMORCE : {p}", flush=True)
        print("-" * 70, flush=True)
        # on colle amorce + completion pour lire la continuation naturelle
        print(f"{p}{completion}", flush=True)
        print("=" * 70 + "\n", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Sonde de completion RODIN-1B (modele base)."
    )
    ap.add_argument("--weights", required=True,
                    help="rodin1b_weights_bf16.pt (poids seuls bf16)")
    ap.add_argument("--tokenizer", required=True,
                    help="modele SentencePiece .model")
    ap.add_argument("--trainer", default=None,
                    help="chemin vers 21_train_rodin.py (defaut : meme dossier "
                         "que sample.py)")
    ap.add_argument("--preset", default="prod", choices=["prod", "pretest"],
                    help="fallback si pas de cfg dans le fichier (defaut prod)")

    ap.add_argument("--prompt", default=None, help="une amorce unique")
    ap.add_argument("--prompts-file", default=None,
                    help="fichier d'amorces (une par ligne)")

    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="0 = greedy deterministe")
    ap.add_argument("--top-k", type=int, default=40, help="0 = desactive")
    ap.add_argument("--top-p", type=float, default=0.95, help="1.0 = desactive")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--show-probs", action="store_true",
                    help="affiche la distribution top-k a chaque token genere")
    ap.add_argument("--topk-display", type=int, default=5,
                    help="nb de tokens affiches avec --show-probs")

    ap.add_argument("--device", default=None,
                    help="cuda / cpu (defaut : cuda si dispo)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda demande mais indisponible -> cpu (lent).", flush=True)
        device = "cpu"

    trainer_path = args.trainer or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "21_train_rodin.py")
    trainer_mod = load_trainer_module(trainer_path)

    model, cfg = load_model(trainer_mod, args.weights, args.preset, device)
    sp = load_tokenizer(args.tokenizer)

    # rope buffers recalcules sur le bon device au 1er forward (.to(x.device)),
    # mais on les place des maintenant pour eviter un transfert par token.
    if hasattr(model, "rope_cos"):
        model.rope_cos = model.rope_cos.to(device)
        model.rope_sin = model.rope_sin.to(device)

    print(f"\n[cfg gen] temp={args.temperature} top_k={args.top_k} "
          f"top_p={args.top_p} max_new={args.max_new} seed={args.seed} "
          f"device={device}\n", flush=True)

    # --- mode batch fichier ---
    if args.prompts_file:
        if not os.path.isfile(args.prompts_file):
            sys.exit(f"[fatal] fichier d'amorces introuvable : {args.prompts_file}")
        with open(args.prompts_file, encoding="utf-8") as f:
            for line in f:
                run_one(model, sp, cfg, line, args, device)
        return

    # --- mode amorce unique ---
    if args.prompt is not None:
        run_one(model, sp, cfg, args.prompt, args, device)
        return

    # --- mode interactif ---
    print("Mode interactif. Tape une AMORCE a completer (pas une question).",
          flush=True)
    print("Ligne vide ou Ctrl-D pour quitter.\n", flush=True)
    while True:
        try:
            line = input("amorce> ")
        except EOFError:
            print()
            break
        if not line.strip():
            break
        run_one(model, sp, cfg, line, args, device)


if __name__ == "__main__":
    main()
