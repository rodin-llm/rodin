# 20_build_blend.py
# Materialiseur de blend RODIN : selectionne des docs par budget-tokens/source
# (avec repetition pour les petites sources), les concatene dans un ORDRE MELANGE
# (interleave des sources) au sein d'un train.bin UNIQUE uint16, et reserve un
# val.bin DISJOINT. Sortie : un .bin a fenetres sequentielles triviales au train.
#
# Architecture (cf. handoff section 7) : on PONDERE A LA MATERIALISATION, pas a
# la volee. Lectures sequentielles au train => MFU maximal, resume = un offset,
# multi-worker gratuit. On transfere ~64 Go (le blend) au lieu des 632 Go de shards.
#
# Garde anti-corruption (3 verrous, cf. discussion) :
#   1. type identique uint16 de bout en bout, aucun cast.
#   2. assert len(tokens extraits) == doc_lengths[i] pour CHAQUE doc copie.
#   3. sanity decode final : 3 docs relus depuis train.bin == lignes JSONL.
#
# Usage :
#   python -u .\scripts\20_build_blend.py --dry-run                       (planifie, n'ecrit rien)
#   python -u .\scripts\20_build_blend.py --dry-run --target-tokens 400_000_000   (pretest, plan)
#   python -u .\scripts\20_build_blend.py --target-tokens 400_000_000     (PRETEST 3090, ~0.8 Go)
#   python -u .\scripts\20_build_blend.py                                 (RUN, 32B, ~64 Go)
#   python -u .\scripts\20_build_blend.py --clean                         (ecrase un blend existant)
#
# Switch pretest/prod = le seul flag --target-tokens. Memes poids => le pretest
# valide le VRAI blend.

import argparse
import json
import os
import sys
import time

import numpy as np
import sentencepiece as spm

# ======================================================================
# CONFIG — editer ici selon la machine (Windows = materialisation)
# ======================================================================
IDX_DIR = r"D:\rodin_index"
SHARD_DIR = r"G:\data\rodin\tokenized"
SP_MODEL = r"G:\data\rodin\bpe\rodin.model"
IN_PATH = r"G:\data\rodin\deduped\merged_final.jsonl"   # sanity decode only
OUT_DIR = r"G:\data\rodin\blend"                        # sortie train/val.bin

MANIFEST = os.path.join(SHARD_DIR, "manifest.json")
SOURCES_MAP = os.path.join(IDX_DIR, "sources_map.json")
EOS_OFFSETS = os.path.join(IDX_DIR, "doc_eos_offsets.u64")
DOC_SOURCES = os.path.join(IDX_DIR, "doc_sources.u8")
DOC_LENGTHS = os.path.join(IDX_DIR, "doc_lengths.u32")
LINE_OFFSETS = os.path.join(IDX_DIR, "line_offsets.u64")   # pour sanity decode

OUT_TRAIN = os.path.join(OUT_DIR, "train.bin")
OUT_VAL = os.path.join(OUT_DIR, "val.bin")
OUT_MANIFEST = os.path.join(OUT_DIR, "blend_manifest.json")

# ----------------------------------------------------------------------
# Chiffres de reference (figes, cf. handoff section 12) — verifies au runtime
N_DOCS = 307_379_231
EXPECTED_TOKENS = 339_519_697_529
EOS_ID = 3
VOCAB = 64_000
N_LINES = 307_379_232
GHOST_LINE = 219_077_862        # doc j <- ligne j (j<GHOST) sinon j+1

# ----------------------------------------------------------------------
# BLEND (DECIDE, handoff section 6) — ancre-moderne, run 32B.
#   poids   = part nominale du budget-tokens
#   max_epochs = plafond de repetition (None = pas de plafond)
# Pour passer en variante PATRIMONIALE (Pleias ~54%), remplacer ce dict par
# le bloc commente plus bas. C'est le SEUL endroit a editer pour changer le
# registre cible de RODIN.
SOURCE_PLAN = {
    "hplt":        {"weight": 0.30, "max_epochs": None},
    "pleia_news":  {"weight": 0.28, "max_epochs": None},
    "wikipedia":   {"weight": 0.18, "max_epochs": None},   # repete ~3.35x
    "pleia_books": {"weight": 0.11, "max_epochs": None},
    "cc100":       {"weight": 0.10, "max_epochs": None},
    "legifrance":  {"weight": 0.02, "max_epochs": 4},      # plafonne COUVERTURE
    "wikisource":  {"weight": 0.01, "max_epochs": 4},      # plafonne COUVERTURE
}
# --- Variante PATRIMONIALE (commentee, reversible jusqu'au run) :
# SOURCE_PLAN = {
#     "pleia_news":  {"weight": 0.42, "max_epochs": None},
#     "pleia_books": {"weight": 0.12, "max_epochs": None},
#     "hplt":        {"weight": 0.20, "max_epochs": None},
#     "cc100":       {"weight": 0.08, "max_epochs": None},
#     "wikipedia":   {"weight": 0.14, "max_epochs": None},
#     "legifrance":  {"weight": 0.03, "max_epochs": 4},
#     "wikisource":  {"weight": 0.01, "max_epochs": 4},
# }

TARGET_TOKENS_DEFAULT = 32_000_000_000   # 32B (run). Pretest via --target-tokens.
VAL_TOKENS = 10_000_000                  # ~10M tokens, DISJOINT du train.
SEED = 1234
WRITE_BUF_TOKENS = 128 * 1024 * 1024     # 256 Mo (uint16) par flush
INTERLEAVE_CHUNK = 2000                  # docs par segment d'interleave


# ======================================================================
# Acces shards — mecanique du script 17 (mmap par shard, cumul manifest)
# ======================================================================
class ShardReader:
    def __init__(self):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            man = json.load(f)
        if man.get("total_tokens") != EXPECTED_TOKENS:
            sys.exit(f"[ERREUR] manifest total_tokens {man.get('total_tokens')} "
                     f"!= {EXPECTED_TOKENS}")
        self.shard_files = [e["file"] for e in man["shards"]]
        counts = np.array([e["tokens"] for e in man["shards"]], dtype=np.uint64)
        self.cum = np.concatenate(([0], np.cumsum(counts)))   # len n_shards+1
        if int(self.cum[-1]) != EXPECTED_TOKENS:
            sys.exit(f"[ERREUR] somme shards {int(self.cum[-1])} != {EXPECTED_TOKENS}")
        self.maps = {}

    def _shard(self, si):
        m = self.maps.get(si)
        if m is None:
            path = os.path.join(SHARD_DIR, self.shard_files[si])
            m = np.memmap(path, dtype=np.uint16, mode="r")
            self.maps[si] = m
        return m

    def doc_tokens(self, eos, i):
        """Tokens du doc i, EOS INCLUS (start..end inclus), gere le chevauchement
        inter-shard. eos[i] = position absolue de l'EOS du doc i."""
        start = 0 if i == 0 else int(eos[i - 1]) + 1
        end = int(eos[i]) + 1     # +1 pour INCLURE l'EOS (separateur de doc)
        parts = []
        pos = start
        while pos < end:
            si = int(np.searchsorted(self.cum, pos, side="right")) - 1
            base = int(self.cum[si])
            lo = pos - base
            hi = min(end - base, int(self.cum[si + 1]) - base)
            parts.append(np.asarray(self._shard(si)[lo:hi], dtype=np.uint16))
            pos = base + hi
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


# ======================================================================
# Planification du blend
# ======================================================================
def normalize_weights(plan):
    s = sum(p["weight"] for p in plan.values())
    return {k: p["weight"] / s for k, p in plan.items()}


def build_source_plan(target_tokens, code_of, docs_by_code, lengths, rng):
    """Pour chaque source, selectionne un ordre de docs (passes reshuffle)
    couvrant le budget-tokens, plafonne par max_epochs. Retourne :
      plan_docs[name]  = np.array des indices de docs (dans l'ordre a ecrire)
      report[name]     = dict (nominal/effectif/deficit/docs/epochs)."""
    weights = normalize_weights(SOURCE_PLAN)
    plan_docs = {}
    report = {}
    for name, w in weights.items():
        code = code_of[name]
        doc_ids = docs_by_code[code]                 # indices de docs de la source
        avail = int(lengths[doc_ids].sum())          # tokens dispo (1 epoch)
        budget = int(round(w * target_tokens))
        max_ep = SOURCE_PLAN[name]["max_epochs"]
        if max_ep is not None:
            budget = min(budget, max_ep * avail)

        picked = []
        acc = 0
        epochs = 0
        order = doc_ids.copy()
        while acc < budget:
            rng.shuffle(order)                        # reshuffle a chaque passe
            for d in order:
                picked.append(d)
                acc += int(lengths[d])
                if acc >= budget:
                    break
            epochs += 1
            if max_ep is not None and epochs >= max_ep and acc < budget:
                break                                 # plafond atteint -> deficit
        plan_docs[name] = np.array(picked, dtype=np.int64)
        report[name] = {
            "weight_nominal": round(w, 4),
            "tokens_target": budget,
            "tokens_effective": acc,
            "deficit": budget - acc,
            "docs_selected": len(picked),
            "docs_unique_avail": len(doc_ids),
            "avail_tokens_1epoch": avail,
            "epochs_approx": round(acc / max(avail, 1), 3),
        }
    return plan_docs, report


def interleave(plan_docs, rng):
    """Melange global : entrelace les sources par segments de INTERLEAVE_CHUNK
    docs, dans un ordre de sources tire aleatoirement a chaque tour. Evite les
    blocs mono-source longs sans tout charger en RAM."""
    cursors = {n: 0 for n in plan_docs}
    arrs = {n: plan_docs[n] for n in plan_docs}
    out = []
    names = list(plan_docs.keys())
    remaining = sum(len(a) for a in arrs.values())
    while remaining > 0:
        rng.shuffle(names)
        for n in names:
            c = cursors[n]
            a = arrs[n]
            if c >= len(a):
                continue
            take = min(INTERLEAVE_CHUNK, len(a) - c)
            out.append(a[c:c + take])
            cursors[n] = c + take
            remaining -= take
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


# ======================================================================
# Sanity decode — relit des docs DEPUIS train.bin, compare au JSONL (script 17)
# ======================================================================
def sanity_decode(out_doc_order, val_set, n_samples=3):
    import orjson
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    line_off = np.fromfile(LINE_OFFSETS, dtype=np.uint64)
    eos = np.fromfile(EOS_OFFSETS, dtype=np.uint64)
    reader = ShardReader()
    rng = np.random.default_rng(SEED + 999)

    # doc i -> ligne JSONL (decalage ghost)
    def line_of_doc(i):
        return i if i < GHOST_LINE else i + 1

    def line_text(line_no):
        off = int(line_off[line_no])
        ln = int(line_off[line_no + 1]) - off
        with open(IN_PATH, "rb") as fh:
            fh.seek(off)
            raw = fh.read(ln).strip()
        obj = orjson.loads(raw)
        for f in ("text", "content", "raw_content", "body", "page_content"):
            v = obj.get(f)
            if isinstance(v, str) and v:
                return v[:200_000]
        return None

    candidates = [int(d) for d in out_doc_order[:50000] if int(d) not in val_set]
    picks = rng.choice(candidates, size=min(n_samples, len(candidates)),
                       replace=False)
    print("\n[SANITY DECODE] 3 docs relus depuis le plan, compares au JSONL :")
    all_ok = True
    for i in picks:
        i = int(i)
        ids = reader.doc_tokens(eos, i)
        ids_no_eos = ids[:-1] if len(ids) and int(ids[-1]) == EOS_ID else ids
        decoded = sp.decode(ids_no_eos.tolist())
        txt = line_text(line_of_doc(i))
        ok = (txt is not None and decoded == txt)
        all_ok &= ok
        print(f"  doc {i:>11,} -> {'MATCH' if ok else 'MISMATCH'} "
              f"(ligne {line_of_doc(i):,}, {len(ids):,} tok)")
    if not all_ok:
        sys.exit("[ERREUR] sanity decode KO -> ne pas utiliser ce blend.")
    print("[OK] sanity decode : le blend est fidele aux shards.")


# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS_DEFAULT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--no-sanity", action="store_true",
                    help="saute le sanity decode (deconseille)")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)

    if not args.dry_run:
        os.makedirs(OUT_DIR, exist_ok=True)
        for p in (OUT_TRAIN, OUT_VAL):
            if os.path.exists(p) and not args.clean:
                sys.exit(f"[ERREUR] {p} existe deja. Relancer avec --clean.")

    # --- chargement index ---
    print("[INFO] chargement de l'index...")
    eos = np.fromfile(EOS_OFFSETS, dtype=np.uint64)
    lengths = np.fromfile(DOC_LENGTHS, dtype=np.uint32).astype(np.int64)
    doc_src = np.fromfile(DOC_SOURCES, dtype=np.uint8)
    for arr, name, n in ((eos, "eos", N_DOCS), (lengths, "lengths", N_DOCS),
                         (doc_src, "doc_src", N_DOCS)):
        if arr.shape[0] != n:
            sys.exit(f"[ERREUR] {name}: {arr.shape[0]:,} != {n:,}")
    if int(eos[-1]) + 1 != EXPECTED_TOKENS:
        sys.exit(f"[ERREUR] eos[-1]+1 {int(eos[-1]) + 1:,} != {EXPECTED_TOKENS:,}")
    if int(lengths.sum()) != EXPECTED_TOKENS:
        sys.exit(f"[ERREUR] sum(lengths) {int(lengths.sum()):,} != {EXPECTED_TOKENS:,}")

    with open(SOURCES_MAP, "r", encoding="utf-8") as f:
        smap = json.load(f)["sources"]
    code_of = {name: smap[name]["code"] for name in smap}
    for name in SOURCE_PLAN:
        if name not in code_of:
            sys.exit(f"[ERREUR] source '{name}' du blend absente de sources_map.json")

    # docs par code (indices de docs)
    print("[INFO] indexation des docs par source...")
    docs_by_code = {c: np.where(doc_src == c)[0] for c in set(code_of.values())}

    # --- planification ---
    print(f"[INFO] budget cible : {args.target_tokens:,} tokens "
          f"(+ {VAL_TOKENS:,} val)")
    plan_docs, report = build_source_plan(args.target_tokens, code_of,
                                          docs_by_code, lengths, rng)

    total_eff = sum(r["tokens_effective"] for r in report.values())
    total_def = sum(r["deficit"] for r in report.values())
    print("\n[PLAN PAR SOURCE]")
    for name in SOURCE_PLAN:
        r = report[name]
        print(f"  {name:<12} | nominal {r['weight_nominal']:.3f} | "
              f"cible {r['tokens_target']/1e9:6.3f}B | "
              f"effectif {r['tokens_effective']/1e9:6.3f}B | "
              f"deficit {r['deficit']/1e6:7.2f}M | "
              f"docs {r['docs_selected']:>10,} | "
              f"~{r['epochs_approx']}ep")
    print(f"\n[TOTAL] effectif {total_eff/1e9:.3f}B "
          f"(cible {args.target_tokens/1e9:.3f}B, deficit {total_def/1e6:.2f}M)")

    # --- ordre global melange ---
    out_doc_order = interleave(plan_docs, rng)
    print(f"[INFO] ordre global : {len(out_doc_order):,} docs (interleave)")

    # --- val set DISJOINT : tire des docs NON utilises au train ---
    used = set(int(d) for d in out_doc_order)
    all_docs = np.arange(N_DOCS, dtype=np.int64)
    free_mask = np.ones(N_DOCS, dtype=bool)
    free_mask[list(used)] = False
    free_docs = all_docs[free_mask]
    if free_docs.size == 0:
        sys.exit("[ERREUR] aucun doc libre pour val (train couvre tout le corpus)")
    rng.shuffle(free_docs)
    val_docs = []
    val_acc = 0
    for d in free_docs:
        val_docs.append(int(d))
        val_acc += int(lengths[d])
        if val_acc >= VAL_TOKENS:
            break
    val_set = set(val_docs)
    print(f"[INFO] val : {len(val_docs):,} docs, {val_acc:,} tokens (disjoint)")

    if args.dry_run:
        print("\n[DRY-RUN] rien ecrit.")
        _write_manifest(args, report, total_eff, total_def, len(out_doc_order),
                        val_acc, len(val_docs), dry=True)
        print(f"[FIN] {(time.time()-t0)/60:.1f} min")
        return

    # --- materialisation ---
    reader = ShardReader()

    def materialize(doc_order, out_path, label):
        tmp = out_path + ".tmp"
        buf = np.empty(WRITE_BUF_TOKENS, dtype=np.uint16)
        n_buf = 0
        written = 0
        n_eos = 0
        t = time.time()
        nxt = t
        with open(tmp, "wb") as out:
            for k, i in enumerate(doc_order):
                i = int(i)
                ids = reader.doc_tokens(eos, i)          # uint16, EOS inclus
                # VERROU 2 : longueur exacte attendue
                if ids.shape[0] != int(lengths[i]):
                    sys.exit(f"[ERREUR] doc {i}: {ids.shape[0]} tok != "
                             f"doc_lengths {int(lengths[i])}")
                if int(ids[-1]) == EOS_ID:
                    n_eos += 1
                m = ids.shape[0]
                if n_buf + m > buf.shape[0]:
                    buf[:n_buf].tofile(out)
                    written += n_buf
                    n_buf = 0
                    while m > buf.shape[0]:               # doc plus gros que le buffer
                        out.write(ids.tobytes())
                        written += m
                        ids = ids[:0]
                        m = 0
                if m:
                    buf[n_buf:n_buf + m] = ids
                    n_buf += m
                now = time.time()
                if now - nxt >= 30:
                    nxt = now
                    rate = (written + n_buf) / (now - t)
                    print(f"  [{label}] {k+1:,}/{len(doc_order):,} docs | "
                          f"{(written+n_buf)/1e9:.3f}B tok | "
                          f"{rate/1e6:.1f}M tok/s")
            if n_buf:
                buf[:n_buf].tofile(out)
                written += n_buf
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, out_path)
        print(f"[OK] {label} : {written:,} tokens, {n_eos:,} EOS "
              f"({written / max(n_eos,1):.1f} tok/doc) -> {out_path}")
        return written, n_eos

    print("\n[MATERIALISATION] val.bin...")
    val_written, _ = materialize(np.array(val_docs, dtype=np.int64), OUT_VAL, "val")
    print("[MATERIALISATION] train.bin...")
    train_written, train_eos = materialize(out_doc_order, OUT_TRAIN, "train")

    _write_manifest(args, report, total_eff, total_def, len(out_doc_order),
                    val_written, len(val_docs), dry=False,
                    train_tokens=train_written, train_docs=train_eos)

    # VERROU 3
    if not args.no_sanity:
        sanity_decode(out_doc_order, val_set)

    print(f"\n[TERMINE] blend materialise en {(time.time()-t0)/60:.1f} min.")
    print(f"  train.bin : {train_written:,} tokens "
          f"({train_written*2/1024**3:.2f} Go)")
    print(f"  val.bin   : {val_written:,} tokens "
          f"({val_written*2/1024**2:.1f} Mo)")


def _write_manifest(args, report, total_eff, total_def, n_docs_train,
                    val_tokens, val_docs, dry, train_tokens=None, train_docs=None):
    man = {
        "blend": "ancre-moderne" if SOURCE_PLAN["hplt"]["weight"] == 0.30
                 else "custom",
        "target_tokens": args.target_tokens,
        "total_effective_tokens": total_eff,
        "total_deficit": total_def,
        "train_docs_planned": n_docs_train,
        "val_tokens": val_tokens,
        "val_docs": val_docs,
        "seed": SEED,
        "dtype": "uint16",
        "eos_id": EOS_ID,
        "dry_run": dry,
        "per_source": report,
    }
    if train_tokens is not None:
        man["train_tokens_written"] = train_tokens
        man["train_docs_written"] = train_docs
    path = OUT_MANIFEST if not dry else OUT_MANIFEST + ".dryrun"
    os.makedirs(os.path.dirname(path), exist_ok=True) if not dry else None
    target = path if not dry else os.path.join(IDX_DIR, os.path.basename(path))
    with open(target, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print(f"[OK] manifest -> {target}")


if __name__ == "__main__":
    main()
