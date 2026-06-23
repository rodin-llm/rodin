# 14_index_sources.py
# Passe A : index ligne -> source depuis merged_final.jsonl
# Lecture : G: (sequentielle) | Ecriture : D: (petite, ~300 Mo)
# Usage :
#   python -u .\scripts\14_index_sources.py --dry-run   (1M lignes, verif)
#   python -u .\scripts\14_index_sources.py             (run complet)

import argparse
import json
import os
import sys
import time

import orjson

IN_PATH = r"G:\data\rodin\deduped\merged_final.jsonl"
OUT_DIR = r"D:\rodin_index"
OUT_ARRAY = os.path.join(OUT_DIR, "line_sources.u8")
OUT_MAP = os.path.join(OUT_DIR, "sources_map.json")

EXPECTED_LINES = 307_379_232  # lines_processed du manifest Phase 3

LOG_EVERY_BYTES = 2 * 1024**3  # log tous les 2 Go lus
PATTERN = b'"source":'


def extract_source_fast(line: bytes):
    """Extrait la valeur du champ source par recherche d'octets depuis la fin.
    Retourne None si ambigu (fallback orjson par l'appelant)."""
    try:
        i = line.rindex(PATTERN)
    except ValueError:
        return None
    j = i + len(PATTERN)
    # sauter espaces eventuels
    while j < len(line) and line[j] in b" \t":
        j += 1
    if j >= len(line) or line[j] != ord('"'):
        return None
    j += 1
    k = line.find(b'"', j)
    if k == -1:
        return None
    val = line[j:k]
    # garde-fou : une vraie valeur de source est courte, ASCII, sans echappement
    if not val or len(val) > 64 or b"\\" in val:
        return None
    return val.decode("ascii", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="ne traite que 1M lignes, n'ecrit rien de definitif")
    ap.add_argument("--clean", action="store_true",
                    help="autorise l'ecrasement d'un index existant")
    args = ap.parse_args()

    if os.path.exists(OUT_ARRAY) and not args.dry_run and not args.clean:
        print(f"[ERREUR] {OUT_ARRAY} existe deja. Relancer avec --clean.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    in_size = os.path.getsize(IN_PATH)
    print(f"[INFO] entree  : {IN_PATH} ({in_size / 1024**4:.2f} To)")
    print(f"[INFO] sortie  : {OUT_ARRAY}")
    print(f"[INFO] mode    : {'DRY-RUN (1M lignes)' if args.dry_run else 'COMPLET'}")

    src_to_code = {}          # nom source -> uint8
    counts = {}               # nom source -> nb lignes
    fallback_count = 0
    lines_done = 0
    bytes_done = 0
    next_log = LOG_EVERY_BYTES
    t0 = time.time()

    limit = 1_000_000 if args.dry_run else None
    tmp_array = OUT_ARRAY + ".tmp"

    buf = bytearray()
    BUF_FLUSH = 8 * 1024 * 1024  # flush du buffer de codes tous les 8 Mo

    with open(IN_PATH, "rb") as fh, open(tmp_array, "wb") as out:
        for line in fh:
            bytes_done += len(line)

            src = extract_source_fast(line)
            if src is None:
                # fallback : parse JSON complet (rare)
                doc = orjson.loads(line)
                src = str(doc["source"])
                fallback_count += 1

            code = src_to_code.get(src)
            if code is None:
                code = len(src_to_code)
                if code > 255:
                    print("[ERREUR] plus de 256 sources distinctes ?!")
                    sys.exit(1)
                src_to_code[src] = code
                print(f"[NOUVELLE SOURCE] '{src}' -> code {code} "
                      f"(ligne {lines_done})")
            counts[src] = counts.get(src, 0) + 1

            buf.append(code)
            if len(buf) >= BUF_FLUSH:
                out.write(buf)
                buf.clear()

            lines_done += 1
            if limit and lines_done >= limit:
                break

            if bytes_done >= next_log:
                next_log += LOG_EVERY_BYTES
                elapsed = time.time() - t0
                rate = bytes_done / elapsed
                eta_s = (in_size - bytes_done) / rate if rate > 0 else 0
                print(f"[PROGRESS] {bytes_done / 1024**3:7.1f} Go | "
                      f"{lines_done:>11,} lignes | "
                      f"{rate / 1024**2:6.0f} Mo/s | "
                      f"ETA {eta_s / 3600:5.2f} h | "
                      f"fallbacks {fallback_count}")

        out.write(buf)
        out.flush()
        os.fsync(out.fileno())

    elapsed = time.time() - t0
    print(f"\n[FIN LECTURE] {lines_done:,} lignes en {elapsed / 3600:.2f} h "
          f"| fallbacks orjson : {fallback_count}")

    # verification du compte
    if not args.dry_run:
        if lines_done != EXPECTED_LINES:
            print(f"[ALERTE] lignes lues ({lines_done:,}) != manifest "
                  f"({EXPECTED_LINES:,}). NE PAS utiliser cet index "
                  f"avant d'avoir compris l'ecart.")
            sys.exit(1)
        os.replace(tmp_array, OUT_ARRAY)
    else:
        os.remove(tmp_array)
        print("[DRY-RUN] fichier temporaire supprime.")

    # mapping + stats (ecrit meme en dry-run, pour inspection)
    map_payload = {
        "input": IN_PATH,
        "lines": lines_done,
        "dry_run": args.dry_run,
        "fallback_parses": fallback_count,
        "sources": {
            src: {"code": code, "lines": counts.get(src, 0)}
            for src, code in sorted(src_to_code.items(), key=lambda x: x[1])
        },
    }
    map_path = OUT_MAP if not args.dry_run else OUT_MAP + ".dryrun"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] mapping ecrit : {map_path}")

    print("\n[REPARTITION]")
    total = max(lines_done, 1)
    for src, code in sorted(src_to_code.items(), key=lambda x: x[1]):
        n = counts.get(src, 0)
        print(f"  code {code:3d} | {src:<14} | {n:>11,} lignes | "
              f"{100.0 * n / total:5.2f} %")


if __name__ == "__main__":
    main()