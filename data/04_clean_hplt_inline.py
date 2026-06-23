#!/usr/bin/env python3
"""
RODIN Phase 1 — Clean HPLT parquet-par-parquet avec suppression raw inline
===========================================================================
Problème : 247Go raw HPLT + expansion x3-x5 à la décompression = impossible
de tout cleaner d'un coup avec ~128Go libres sur D:.

Solution : traiter 1 parquet → écrire jsonl → supprimer le raw immédiatement.
Espace net consommé = taille d'1 parquet décompressé (~2-4Go max) + jsonl accumulé.

Architecture identique à 03_clean_pleia_books.py (subprocess isolation PyArrow).
Reprise automatique via .progress (résiste aux crashes).

Estimation finale :
  - 1006 parquets × ~245Mo = 247Go raw
  - Rétention web crawl : ~40-60% après filtres
  - jsonl final estimé : ~80-130Go

Usage :
    python scripts/04_clean_hplt_inline.py
    python scripts/04_clean_hplt_inline.py --dry-run     # simule sans supprimer
    python scripts/04_clean_hplt_inline.py --force       # repart de zéro
    python scripts/04_clean_hplt_inline.py --no-delete   # garde les raw (debug)
"""

import argparse
import json
import sys
import subprocess
import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(r"D:\data\rodin")
RAW_DIR   = BASE_DIR / "raw"  / "hplt" / "fra_Latn"
OUT_DIR   = BASE_DIR / "cleaned" / "hplt"
SCRIPTS   = BASE_DIR / "scripts"

# ─── Filtres qualité HPLT (web crawl) ────────────────────────────────────────
# HPLT cleaned est déjà pré-filtré par HPLT mais reste du web bruité.
# On assouplit légèrement vs les defaults (min_chars 300→200, min_words 50→30)
# pour ne pas sur-filtrer du contenu déjà sélectionné, tout en virant
# le vrai spam (short, trop de chiffres, pas français).

HPLT_PARAMS = {
    "min_chars":               200,
    "min_words":                30,
    "min_doc_lines":             1,
    "max_special_char_ratio":  0.15,
    "max_digit_ratio":         0.18,
    "max_uppercase_ratio":     0.20,
    "fr_ratio_threshold":      0.05,   # Un peu plus strict : HPLT devrait être du vrai FR
}

# ─── Worker subprocess ────────────────────────────────────────────────────────
# Tournera dans un process isolé → RAM libérée entre chaque parquet.
# La colonne texte de HPLT cleaned s'appelle "text".

WORKER_CODE = r'''
import sys
import json
import re

def main():
    pf_path                = sys.argv[1]
    min_chars              = int(sys.argv[2])
    min_words              = int(sys.argv[3])
    min_doc_lines          = int(sys.argv[4])
    max_special_char_ratio = float(sys.argv[5])
    max_digit_ratio        = float(sys.argv[6])
    max_uppercase_ratio    = float(sys.argv[7])
    fr_ratio_threshold     = float(sys.argv[8])

    FR_WORDS = frozenset([
        "le","la","les","de","du","des","un","une","et","en","est","que",
        "qui","dans","sur","par","il","elle","ils","elles","nous","vous",
        "son","sa","ses","ce","cet","cette","ces","aussi","mais","ou",
        "donc","or","ni","car","pas","plus","très","bien","avec","pour",
        "comme","tout","faire","être","avoir","au","aux","dont","où",
        "quand","même","leur","leurs","y","lui","on","se","ne","si","je",
        "tu","me","te","nous","vous","mon","ton","ma","ta","nos","vos",
        "cette","entre","après","avant","lors","dont","puis","sans","sous",
    ])

    def is_french(text):
        words = re.findall(r'\b[a-z\xe0-\xff]+\b', text[:3000].lower())
        if len(words) < 10:
            return True
        ratio = sum(1 for w in words if w in FR_WORDS) / len(words)
        return ratio >= fr_ratio_threshold

    def quality_ok(text):
        text = text.strip()
        if len(text) < min_chars:
            return False
        words = text.split()
        if len(words) < min_words:
            return False
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < min_doc_lines:
            return False
        n = len(text)
        alpha  = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        other  = n - alpha - digits - spaces
        if n > 0:
            if digits / n > max_digit_ratio:
                return False
            if other / n > max_special_char_ratio:
                return False
            if alpha > 0 and sum(c.isupper() for c in text) / alpha > max_uppercase_ratio:
                return False
        if not is_french(text):
            return False
        return True

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("FAIL: pyarrow non disponible", file=sys.stderr)
        sys.exit(1)

    count   = 0
    skipped = 0

    # HPLT cleaned : colonne principale = "text"
    # Fallback sur "content" si absent (versions anciennes)
    TEXT_COLS = ["text", "content", "raw_text"]

    try:
        schema = pq.read_schema(pf_path)
        col = next((c for c in TEXT_COLS if c in schema.names), None)
        if col is None:
            print(f"FAIL: aucune colonne texte dans {pf_path} — colonnes: {schema.names}", file=sys.stderr)
            sys.exit(1)

        pf       = pq.ParquetFile(pf_path)
        n_groups = pf.metadata.num_row_groups

        for rg_idx in range(n_groups):
            try:
                tbl   = pf.read_row_group(rg_idx, columns=[col])
                texts = tbl.column(col).to_pylist()
                del tbl

                for text in texts:
                    if not text or not isinstance(text, str):
                        skipped += 1
                        continue
                    text = text.strip()
                    if not quality_ok(text):
                        skipped += 1
                        continue
                    if len(text) > 500_000:
                        text = text[:500_000]
                    count += 1
                    print(json.dumps({
                        "text":   text,
                        "source": "hplt",
                        "id":     f"hplt_{count}",
                    }, ensure_ascii=True))

            except Exception as e:
                print(f"WARN rg{rg_idx}: {e}", file=sys.stderr)
                continue

        del pf
        print(f"OK {count} skipped={skipped}", file=sys.stderr)

    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

main()
'''

# ─── Helpers ──────────────────────────────────────────────────────────────────

def free_gb(path: Path) -> float:
    return shutil.disk_usage(path.anchor).free / 1024**3

def get_progress(progress_file: Path) -> set[str]:
    if progress_file.exists():
        return set(progress_file.read_text(encoding="utf-8").splitlines())
    return set()

def mark_progress(progress_file: Path, filename: str):
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN — Clean HPLT inline (suppression raw par parquet)"
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Simule sans écrire ni supprimer")
    p.add_argument("--no-delete", action="store_true",
                   help="Ne supprime pas les raw après clean (debug)")
    p.add_argument("--force",     action="store_true",
                   help="Ignore .progress et repart de zéro")
    p.add_argument("--timeout",   type=int, default=600,
                   help="Timeout subprocess par parquet en secondes (défaut: 600)")
    p.add_argument("--raw-dir",   default=None,
                   help=f"Dossier raw override (défaut: {RAW_DIR})")
    p.add_argument("--out-dir",   default=None,
                   help=f"Dossier output override (défaut: {OUT_DIR})")
    return p.parse_args()


def main():
    args    = parse_args()
    raw_dir = Path(args.raw_dir) if args.raw_dir else RAW_DIR
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    out_file      = out_dir / "hplt_cleaned.jsonl"
    done_flag     = out_dir / ".done"
    progress_file = out_dir / ".progress"
    worker_script = SCRIPTS / "_worker_hplt.py"

    # ── Vérifications préalables
    if done_flag.exists() and not args.force:
        print("[hplt] Déjà terminé (.done présent). Utiliser --force pour relancer.")
        return

    if not raw_dir.exists():
        print(f"ERREUR : dossier raw introuvable : {raw_dir}")
        sys.exit(1)

    parquet_files = sorted(raw_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERREUR : aucun .parquet dans {raw_dir}")
        sys.exit(1)

    done_files = set() if args.force else get_progress(progress_file)
    if args.force and progress_file.exists():
        progress_file.unlink()
    todo = [p for p in parquet_files if p.name not in done_files]

    print(f"[hplt] {len(parquet_files)} parquets total | "
          f"{len(done_files)} déjà traités | {len(todo)} restants")
    print(f"  Raw dir  : {raw_dir}")
    print(f"  Output   : {out_file}")
    print(f"  Filtres  : min_chars={HPLT_PARAMS['min_chars']} "
          f"min_words={HPLT_PARAMS['min_words']} "
          f"fr_ratio>={HPLT_PARAMS['fr_ratio_threshold']}")
    print(f"  Mode     : {'DRY-RUN' if args.dry_run else 'SUPPRESSION RAW ACTIVÉE' if not args.no_delete else 'NO-DELETE (debug)'}")
    print()

    if not args.dry_run and not args.no_delete:
        print("⚠️  ATTENTION : ce script supprime les parquets raw après chaque clean.")
        print("   Les données raw HPLT seront détruites au fur et à mesure.")
        print("   C'est voulu pour économiser l'espace disque.")
        confirm = input("   Confirmer ? (oui/non) : ").strip().lower()
        if confirm not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            return
        print()

    # ── Écrire le worker
    if not args.dry_run:
        SCRIPTS.mkdir(parents=True, exist_ok=True)
        worker_script.write_text(WORKER_CODE, encoding="utf-8")

    total_docs    = 0
    total_skipped = 0
    total_deleted = 0
    errors        = []

    p = HPLT_PARAMS

    mode = "a" if not args.force else "w"

    fh = None if args.dry_run else open(out_file, mode, encoding="utf-8")

    try:
        for i, pf in enumerate(todo):
            free = free_gb(BASE_DIR)
            print(f"  [{i+1}/{len(todo)}] {pf.name} | D: libre={free:.1f}Go ...",
                  end=" ", flush=True)

            if args.dry_run:
                print("DRY-RUN skip")
                continue

            # ── Lancer le worker subprocess
            cmd = [
                sys.executable, str(worker_script),
                str(pf),
                str(p["min_chars"]),
                str(p["min_words"]),
                str(p["min_doc_lines"]),
                str(p["max_special_char_ratio"]),
                str(p["max_digit_ratio"]),
                str(p["max_uppercase_ratio"]),
                str(p["fr_ratio_threshold"]),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=args.timeout,
                )
                stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
                stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            except subprocess.TimeoutExpired:
                print(f"TIMEOUT ({args.timeout}s) — skip")
                errors.append(pf.name)
                continue
            except Exception as e:
                print(f"ERREUR subprocess : {e}")
                errors.append(pf.name)
                continue

            if "FAIL" in stderr and result.returncode != 0:
                print(f"FAIL")
                for line in stderr.splitlines():
                    if line.strip():
                        print(f"    {line}")
                errors.append(pf.name)
                continue

            # ── Écrire les docs valides
            n_written   = 0
            n_skip_json = 0
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    fh.write(line + "\n")
                    n_written += 1
                except json.JSONDecodeError:
                    n_skip_json += 1

            fh.flush()

            # ── Extraire stats stderr
            n_skipped = 0
            for s in stderr.splitlines():
                if s.startswith("OK "):
                    parts = s.split()
                    try:
                        n_skipped = int(parts[2].split("=")[1])
                    except (IndexError, ValueError):
                        pass
                    break

            total_docs    += n_written
            total_skipped += n_skipped
            mark_progress(progress_file, pf.name)

            pct = n_written / max(n_written + n_skipped, 1) * 100

            # ── Supprimer le raw immédiatement
            deleted_str = ""
            if not args.no_delete:
                try:
                    raw_size_mb = pf.stat().st_size / 1024**2
                    pf.unlink()
                    total_deleted += 1
                    deleted_str = f" | raw supprimé ({raw_size_mb:.0f}Mo)"
                except Exception as e:
                    deleted_str = f" | ERREUR suppression: {e}"

            free_after = free_gb(BASE_DIR)
            print(f"{n_written:,} docs ({pct:.0f}%) | total={total_docs:,}{deleted_str} | libre={free_after:.1f}Go")

    finally:
        if fh:
            fh.close()
        if worker_script.exists():
            worker_script.unlink(missing_ok=True)

    # ── Stats finales
    print()
    print("=" * 65)
    print(f"[hplt] Clean inline terminé")
    print(f"  Parquets traités  : {len(todo) - len(errors)}/{len(todo)}")
    print(f"  Parquets supprimés: {total_deleted}")
    print(f"  Documents gardés  : {total_docs:,}")
    print(f"  Documents rejetés : {total_skipped:,}")
    if total_docs + total_skipped > 0:
        print(f"  Taux rétention    : {total_docs/(total_docs+total_skipped)*100:.1f}%")
    if out_file.exists():
        print(f"  Taille jsonl      : {out_file.stat().st_size/1024**3:.2f} Go")
    if errors:
        print(f"  Erreurs ({len(errors)}) : {errors[:10]}{'...' if len(errors)>10 else ''}")
    print("=" * 65)

    if not errors:
        done_flag.touch()
        print(f"  .done créé ✅")
        # Vérifier si le dossier raw est vide → le supprimer proprement
        remaining = list(raw_dir.glob("*.parquet"))
        if not remaining:
            try:
                raw_dir.rmdir()
                raw_dir.parent.rmdir()   # hplt/ si vide aussi
                print(f"  Dossier raw vide supprimé : {raw_dir}")
            except Exception:
                pass
    else:
        print(f"  {len(errors)} erreurs — .done NON créé")
        print(f"  Relancer sans --force pour retry uniquement les erreurs")


if __name__ == "__main__":
    main()
