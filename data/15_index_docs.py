# 15_index_docs.py
# Passe B : scan EOS (id=3) des shards -> offsets globaux des documents
# Passe C : jointure avec line_sources.u8 -> tokens par source
# Lecture : G:\data\rodin\tokenized\ | Ecriture : D:\rodin_index\
# Usage :
#   python -u .\scripts\15_index_docs.py --dry-run   (shard 0000 uniquement)
#   python -u .\scripts\15_index_docs.py             (run complet)

import argparse
import json
import os
import sys
import time

import numpy as np

SHARD_DIR = r"G:\data\rodin\tokenized"
MANIFEST = os.path.join(SHARD_DIR, "manifest.json")
IDX_DIR = r"D:\rodin_index"
LINE_SOURCES = os.path.join(IDX_DIR, "line_sources.u8")
SOURCES_MAP = os.path.join(IDX_DIR, "sources_map.json")
OUT_OFFSETS = os.path.join(IDX_DIR, "doc_eos_offsets.u64")
OUT_STATS = os.path.join(IDX_DIR, "corpus_index_stats.json")

EOS_ID = 3
EXPECTED_DOCS = 307_379_232
EXPECTED_TOKENS = 339_519_697_529
CHUNK = 128 * 1024 * 1024  # 128M tokens par chunk de scan (256 Mo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="ne scanne que le shard 0000, n'ecrit rien de definitif")
    ap.add_argument("--clean", action="store_true",
                    help="autorise l'ecrasement d'un index existant")
    args = ap.parse_args()

    if os.path.exists(OUT_OFFSETS) and not args.dry_run and not args.clean:
        print(f"[ERREUR] {OUT_OFFSETS} existe deja. Relancer avec --clean.")
        sys.exit(1)

    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    shards = manifest["shards"]
    if args.dry_run:
        shards = shards[:1]

    print(f"[INFO] shards a scanner : {len(shards)}")
    print(f"[INFO] sortie offsets   : {OUT_OFFSETS}")
    print(f"[INFO] mode             : "
          f"{'DRY-RUN (shard 0000)' if args.dry_run else 'COMPLET'}")

    tmp_offsets = OUT_OFFSETS + ".tmp"
    global_pos = 0          # position absolue dans le flux de tokens
    total_eos = 0
    shard_mismatch = False
    shards_ending_on_eos = 0
    t0 = time.time()

    with open(tmp_offsets, "wb") as out:
        for si, entry in enumerate(shards):
            path = os.path.join(SHARD_DIR, entry["file"])
            arr = np.memmap(path, dtype=np.uint16, mode="r")
            n_tok = arr.shape[0]

            # verif 3 : tokens du shard vs manifest
            if n_tok != entry["tokens"]:
                print(f"[ALERTE] {entry['file']} : {n_tok:,} tokens "
                      f"vs manifest {entry['tokens']:,}")
                shard_mismatch = True

            shard_eos = 0
            for start in range(0, n_tok, CHUNK):
                chunk = arr[start:start + CHUNK]
                idx = np.flatnonzero(chunk == EOS_ID)
                if idx.size:
                    (idx.astype(np.uint64) + np.uint64(global_pos + start)) \
                        .tofile(out)
                    shard_eos += idx.size

            ends_on_eos = bool(arr[n_tok - 1] == EOS_ID)
            shards_ending_on_eos += ends_on_eos
            del arr

            global_pos += n_tok
            total_eos += shard_eos
            elapsed = time.time() - t0
            rate = global_pos * 2 / elapsed / 1024**2
            print(f"[SHARD {si:02d}] {entry['file']} | {n_tok:>13,} tok | "
                  f"{shard_eos:>11,} EOS | fin-sur-EOS={ends_on_eos} | "
                  f"{rate:6.0f} Mo/s")

        out.flush()
        os.fsync(out.fileno())

    elapsed = time.time() - t0
    print(f"\n[FIN SCAN] {global_pos:,} tokens | {total_eos:,} EOS "
          f"| {elapsed / 60:.1f} min")
    print(f"[INFO] shards finissant sur un EOS : "
          f"{shards_ending_on_eos}/{len(shards)}")

    # ---- verifications croisees (run complet uniquement) ----
    if not args.dry_run:
        ok = True
        if total_eos != EXPECTED_DOCS:
            print(f"[ECHEC] EOS ({total_eos:,}) != docs attendus "
                  f"({EXPECTED_DOCS:,})")
            ok = False
        if global_pos != EXPECTED_TOKENS:
            print(f"[ECHEC] tokens ({global_pos:,}) != manifest "
                  f"({EXPECTED_TOKENS:,})")
            ok = False
        if shard_mismatch:
            print("[ECHEC] au moins un shard ne matche pas le manifest")
            ok = False
        if not ok:
            print("[ABANDON] index NON finalise, fichier .tmp conserve "
                  "pour inspection.")
            sys.exit(1)
        os.replace(tmp_offsets, OUT_OFFSETS)
        print("[OK] verifications croisees PASSEES, offsets finalises.")
    else:
        os.remove(tmp_offsets)
        print("[DRY-RUN] fichier temporaire supprime.")
        return

    # ---- passe C : tokens par source ----
    print("\n[PASSE C] jointure offsets x sources...")
    offsets = np.fromfile(OUT_OFFSETS, dtype=np.uint64)
    codes = np.fromfile(LINE_SOURCES, dtype=np.uint8)
    assert offsets.shape[0] == codes.shape[0] == EXPECTED_DOCS

    # longueur du doc i (EOS inclus) = offsets[i] - offsets[i-1]
    lengths = np.empty(EXPECTED_DOCS, dtype=np.uint64)
    lengths[0] = offsets[0] + 1
    lengths[1:] = np.diff(offsets)

    tok_per_src = np.bincount(codes, weights=lengths.astype(np.float64))

    with open(SOURCES_MAP, "r", encoding="utf-8") as f:
        smap = json.load(f)["sources"]
    code_to_name = {v["code"]: k for k, v in smap.items()}

    stats = {"total_tokens": int(global_pos),
             "total_docs": int(total_eos),
             "shards_ending_on_eos": int(shards_ending_on_eos),
             "per_source": {}}
    print(f"\n[TOKENS PAR SOURCE]")
    for code in range(len(tok_per_src)):
        name = code_to_name[code]
        n_tok = int(tok_per_src[code])
        n_docs = int((codes == code).sum())
        pct = 100.0 * n_tok / global_pos
        stats["per_source"][name] = {
            "code": code, "docs": n_docs, "tokens": n_tok, "pct": round(pct, 3),
            "mean_doc_tokens": round(n_tok / max(n_docs, 1), 1),
        }
        print(f"  {name:<14} | {n_docs:>11,} docs | {n_tok:>15,} tok "
              f"| {pct:6.2f} % | moy {n_tok / max(n_docs, 1):>9,.0f} tok/doc")

    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] stats ecrites : {OUT_STATS}")


if __name__ == "__main__":
    main()