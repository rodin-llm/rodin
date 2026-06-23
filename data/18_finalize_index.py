# 18_finalize_index.py
# Finalise l'index doc<->source<->tokens en integrant l'anomalie connue :
# ligne 219,077,862 = 2 docs hplt fusionnes (\n perdu Phase 1), 0 token ecrit.
# Decalage : doc j <- ligne j si j < GHOST_LINE, sinon ligne j+1.
# Aucun rescan : reutilise .tmp du 15, line_sources.u8, manifest.
#
# Usage :
#   python -u .\scripts\18_finalize_index.py --dry-run
#   python -u .\scripts\18_finalize_index.py

import argparse
import json
import os
import sys

import numpy as np

IDX_DIR = r"D:\rodin_index"
SHARD_DIR = r"G:\data\rodin\tokenized"
MANIFEST = os.path.join(SHARD_DIR, "manifest.json")
SOURCES_MAP = os.path.join(IDX_DIR, "sources_map.json")
LINE_SOURCES = os.path.join(IDX_DIR, "line_sources.u8")

EOS_TMP = os.path.join(IDX_DIR, "doc_eos_offsets.u64.tmp")
EOS_FINAL = os.path.join(IDX_DIR, "doc_eos_offsets.u64")

OUT_DOC_SOURCES = os.path.join(IDX_DIR, "doc_sources.u8")
OUT_DOC_LENGTHS = os.path.join(IDX_DIR, "doc_lengths.u32")
OUT_STATS = os.path.join(IDX_DIR, "corpus_index_stats.json")

N_LINES = 307_379_232
N_DOCS = 307_379_231
EXPECTED_TOKENS = 339_519_697_529
GHOST_LINE = 219_077_862            # ligne JSONL fusionnee (2 docs hplt perdus)
GHOST_IDS = ["hplt_45067", "hplt_29411"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if os.path.exists(EOS_FINAL) and not args.dry_run and not args.clean:
        print(f"[ERREUR] {EOS_FINAL} existe deja. Relancer avec --clean.")
        sys.exit(1)

    # --- charge les offsets EOS (depuis .tmp si pas encore finalise) ---
    src_eos = EOS_FINAL if os.path.exists(EOS_FINAL) else EOS_TMP
    print(f"[INFO] offsets EOS : {src_eos}")
    eos = np.fromfile(src_eos, dtype=np.uint64)
    if eos.shape[0] != N_DOCS:
        print(f"[ERREUR] {eos.shape[0]:,} offsets EOS != {N_DOCS:,} attendus")
        sys.exit(1)

    # --- verif tokens totaux ---
    if int(eos[-1]) + 1 != EXPECTED_TOKENS:
        print(f"[ERREUR] dernier EOS+1 ({int(eos[-1]) + 1:,}) != "
              f"{EXPECTED_TOKENS:,}")
        sys.exit(1)
    print(f"[OK] {N_DOCS:,} docs, {EXPECTED_TOKENS:,} tokens confirmes")

    # --- longueurs de docs (EOS inclus) ---
    lengths = np.empty(N_DOCS, dtype=np.uint64)
    lengths[0] = eos[0] + 1
    lengths[1:] = np.diff(eos)
    if lengths.max() > 0xFFFFFFFF:
        print(f"[ERREUR] un doc depasse uint32 ({int(lengths.max()):,} tok)")
        sys.exit(1)
    # plafond attendu ~52k (MAX_TEXT_LEN/fertilite) + EOS ; on log le max
    print(f"[INFO] longueur doc max : {int(lengths.max()):,} tokens "
          f"(plafond troncature 200k chars attendu)")

    # --- mapping doc -> source avec decalage ghost ---
    line_src = np.fromfile(LINE_SOURCES, dtype=np.uint8)
    if line_src.shape[0] != N_LINES:
        print(f"[ERREUR] {line_src.shape[0]:,} codes source != {N_LINES:,}")
        sys.exit(1)

    # doc j <- ligne (j si j < GHOST_LINE sinon j+1)
    doc_lines = np.arange(N_DOCS, dtype=np.int64)
    doc_lines[GHOST_LINE:] += 1
    assert doc_lines[-1] == N_LINES - 1, "le dernier doc doit pointer la derniere ligne"
    doc_src = line_src[doc_lines]

    # garde-fou : la ligne fusionnee etait hplt -> les deux docs voisins aussi
    with open(SOURCES_MAP, "r", encoding="utf-8") as f:
        smap = json.load(f)["sources"]
    code_to_name = {v["code"]: k for k, v in smap.items()}
    hplt_code = smap["hplt"]["code"]
    around = doc_src[GHOST_LINE - 2: GHOST_LINE + 2]
    if not np.all(around == hplt_code):
        print(f"[ALERTE] voisinage du ghost pas 100% hplt : {around} "
              f"-> inspecter avant de continuer")

    # --- passe C : tokens par source ---
    n_codes = max(code_to_name) + 1
    tok_per_src = np.bincount(doc_src, weights=lengths.astype(np.float64),
                              minlength=n_codes)
    doc_per_src = np.bincount(doc_src, minlength=n_codes)

    total_tok = int(lengths.sum())
    stats = {
        "total_docs": N_DOCS,
        "total_lines_jsonl": N_LINES,
        "total_tokens": total_tok,
        "max_doc_tokens": int(lengths.max()),
        "anomaly": {
            "ghost_line": GHOST_LINE,
            "cause": "deux objets JSON hplt concatenes (\\n perdu en Phase 1), "
                     "rejetes par le parseur du script 12 -> 0 token ecrit",
            "lost_doc_ids": GHOST_IDS,
            "doc_to_line_shift": f"doc j <- ligne j (j<{GHOST_LINE}) sinon j+1",
        },
        "per_source": {},
    }
    print(f"\n[TOKENS PAR SOURCE]")
    order = sorted(range(n_codes), key=lambda c: -tok_per_src[c])
    for code in order:
        name = code_to_name[code]
        nt, nd = int(tok_per_src[code]), int(doc_per_src[code])
        pct = 100.0 * nt / total_tok
        stats["per_source"][name] = {
            "code": code, "docs": nd, "tokens": nt, "pct": round(pct, 3),
            "mean_doc_tokens": round(nt / max(nd, 1), 1)}
        print(f"  {name:<12} | {nd:>11,} docs | {nt:>15,} tok | "
              f"{pct:6.2f} % | moy {nt / max(nd, 1):>8,.0f} tok/doc")

    editorial = sum(int(tok_per_src[smap[s]["code"]])
                    for s in ("legifrance", "wikipedia", "wikisource",
                              "pleia_books"))
    print(f"\n[EDITORIAL] legifrance+wikipedia+wikisource+pleia_books = "
          f"{editorial:,} tok ({100.0 * editorial / total_tok:.2f} %)")
    print(f"[WEB]       hplt+cc100+pleia_news = "
          f"{total_tok - editorial:,} tok "
          f"({100.0 * (total_tok - editorial) / total_tok:.2f} %)")

    if args.dry_run:
        print("\n[DRY-RUN] rien ecrit.")
        return

    # --- ecritures (atomiques) ---
    if src_eos == EOS_TMP:
        os.replace(EOS_TMP, EOS_FINAL)
        print(f"\n[OK] offsets EOS finalises -> {EOS_FINAL}")
    doc_src.astype(np.uint8).tofile(OUT_DOC_SOURCES + ".tmp")
    os.replace(OUT_DOC_SOURCES + ".tmp", OUT_DOC_SOURCES)
    lengths.astype(np.uint32).tofile(OUT_DOC_LENGTHS + ".tmp")
    os.replace(OUT_DOC_LENGTHS + ".tmp", OUT_DOC_LENGTHS)
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] doc_sources.u8 ({N_DOCS:,} octets)")
    print(f"[OK] doc_lengths.u32 ({4 * N_DOCS:,} octets)")
    print(f"[OK] stats -> {OUT_STATS}")
    print("\n[TERMINE] index doc<->source<->tokens finalise.")


if __name__ == "__main__":
    main()