#!/usr/bin/env python3
"""
RODIN Phase 1 — Cleaning pleia_books via subprocess isolation
=============================================================
Problème : pyarrow alloue ~1 Go RAM par parquet lors de la décompression,
indépendamment du batch_size. Avec 145 parquets × ~1 Go = OOM garanti
si tout tourne dans le même process.

Solution : chaque parquet est traité dans un subprocess Python séparé.
Le subprocess meurt après chaque fichier → RAM libérée à 100% entre chaque.

Features :
  - Reprise automatique via .progress (résiste aux crashes)
  - ensure_ascii=True sur stdout → pas de pb encoding pipe Windows
  - Filtres qualité intégrés (miroir SOURCE_OVERRIDES pleia_books)
  - Détection automatique colonne texte (complete_text, text, full_text...)
  - Timeout 300s par parquet
  - Log progression + stats finales

Usage :
    python scripts/03_clean_pleia_books.py
    python scripts/03_clean_pleia_books.py --force   # ignore .progress et repart de zéro
    python scripts/03_clean_pleia_books.py --raw-dir D:\\data\\rodin\\raw\\pleia_books
    python scripts/03_clean_pleia_books.py --source pleia_news  # réutilisable pour pleia_news
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\data\rodin")

# ─── Filtres qualité — miroir SOURCE_OVERRIDES du script principal ────────────

SOURCE_PARAMS = {
    "pleia_books": {
        "min_chars":              200,
        "min_words":               30,
        "min_doc_lines":            1,
        "max_special_char_ratio": 0.18,
        "max_digit_ratio":        0.20,
        "max_uppercase_ratio":    0.25,
        "fr_ratio_threshold":     0.04,
    },
    "pleia_news": {
        "min_chars":              150,
        "min_words":               20,
        "min_doc_lines":            1,
        "max_special_char_ratio": 0.20,
        "max_digit_ratio":        0.20,
        "max_uppercase_ratio":    0.30,
        "fr_ratio_threshold":     0.04,
    },
}

# ─── Code du worker (exécuté dans le subprocess) ─────────────────────────────
# IMPORTANT : ensure_ascii=True sur tous les json.dumps → stdout 100% ASCII
# → aucun pb d'encoding sur le pipe Windows, quel que soit le contenu du texte.

WORKER_CODE = r'''
import sys
import json
import re

def main():
    pf_path   = sys.argv[1]
    text_col  = sys.argv[2]
    source    = sys.argv[3]
    min_chars             = int(sys.argv[4])
    min_words             = int(sys.argv[5])
    min_doc_lines         = int(sys.argv[6])
    max_special_char_ratio = float(sys.argv[7])
    max_digit_ratio       = float(sys.argv[8])
    max_uppercase_ratio   = float(sys.argv[9])
    fr_ratio_threshold    = float(sys.argv[10])

    # Mots FR courants — détection heuristique rapide sans langdetect
    FR_WORDS = frozenset([
        "le","la","les","de","du","des","un","une","et","en","est","que",
        "qui","dans","sur","par","il","elle","ils","elles","nous","vous",
        "son","sa","ses","ce","cet","cette","ces","aussi","mais","ou",
        "donc","or","ni","car","pas","plus","très","bien","avec","pour",
        "comme","tout","faire","être","avoir","au","aux","dont","où",
        "quand","même","leur","leurs","y","lui","on","se","ne","si","je",
        "tu","me","te","nous","vous","mon","ton","ma","ta","nos","vos",
    ])

    def is_french(text):
        words = re.findall(r'\b[a-z\xe0-\xff]+\b', text[:3000].lower())
        if len(words) < 10:
            return True   # trop court pour décider → laisser passer
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

    count = 0
    skipped = 0

    try:
        pf       = pq.ParquetFile(pf_path)
        n_groups = pf.metadata.num_row_groups

        for rg_idx in range(n_groups):
            try:
                tbl   = pf.read_row_group(rg_idx, columns=[text_col])
                texts = tbl.column(text_col).to_pylist()
                del tbl

                for text in texts:
                    if not text or not isinstance(text, str):
                        skipped += 1
                        continue
                    text = text.strip()
                    if not quality_ok(text):
                        skipped += 1
                        continue
                    # Tronquer à 500K chars max
                    if len(text) > 500_000:
                        text = text[:500_000]
                    count += 1
                    # ensure_ascii=True : stdout 100% ASCII → pas d'erreur encoding pipe
                    print(json.dumps({
                        "text":   text,
                        "source": source,
                        "id":     f"{source}_{count}",
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

# ─── Helpers ─────────────────────────────────────────────────────────────────

def detect_text_col(pf_path: Path) -> str | None:
    """Détecte la colonne texte dans un parquet sans le charger."""
    try:
        import pyarrow.parquet as pq
        cols = pq.read_schema(pf_path).names
        candidates = ["complete_text", "text", "full_text", "raw_text",
                      "plain_text", "content", "body", "article",
                      "texte", "contenu", "document"]
        col = next((c for c in candidates if c in cols), None)
        if col is None:
            col = next((c for c in cols if "text" in c.lower()), None)
        return col
    except Exception as e:
        print(f"  WARN schema {pf_path.name} : {e}")
        return None


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
        description="RODIN — Cleaning parquet PleIAs via subprocess isolation"
    )
    p.add_argument("--source",   default="pleia_books",
                   choices=list(SOURCE_PARAMS.keys()),
                   help="Source à cleaner (défaut: pleia_books)")
    p.add_argument("--raw-dir",  default=None,
                   help="Dossier raw override (défaut: raw/<source>)")
    p.add_argument("--out-dir",  default=None,
                   help="Dossier output override (défaut: cleaned/<source>)")
    p.add_argument("--timeout",  type=int, default=300,
                   help="Timeout subprocess par parquet en secondes (défaut: 300)")
    p.add_argument("--force",    action="store_true",
                   help="Ignore .progress et retraite tous les parquets")
    return p.parse_args()


def main():
    args   = parse_args()
    source = args.source
    params = SOURCE_PARAMS[source]

    raw_dir  = Path(args.raw_dir)  if args.raw_dir  else BASE_DIR / "raw"     / source
    out_dir  = Path(args.out_dir)  if args.out_dir  else BASE_DIR / "cleaned" / source
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file      = out_dir / f"{source}_cleaned.jsonl"
    done_flag     = out_dir / ".done"
    progress_file = out_dir / ".progress"
    worker_script = BASE_DIR / "scripts" / "_worker_parquet.py"

    # Vérifier si déjà terminé
    if done_flag.exists() and not args.force:
        print(f"[{source}] Déjà terminé (.done présent). Utiliser --force pour relancer.")
        return

    # Écrire le worker sur disque
    worker_script.write_text(WORKER_CODE, encoding="utf-8")

    # Lister les parquets
    parquet_files = sorted(raw_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"ERREUR : aucun .parquet dans {raw_dir}")
        return

    # Reprendre là où on s'est arrêté
    done_files = set() if args.force else get_progress(progress_file)
    if args.force and progress_file.exists():
        progress_file.unlink()
    todo = [p for p in parquet_files if p.name not in done_files]

    print(f"[{source}] {len(parquet_files)} parquets total | "
          f"{len(done_files)} déjà traités | {len(todo)} restants")
    print(f"  Output   : {out_file}")
    print(f"  Filtres  : min_chars={params['min_chars']} min_words={params['min_words']} "
          f"fr_ratio>={params['fr_ratio_threshold']}")
    print()

    total_docs    = 0
    total_skipped = 0
    errors        = []

    # Ouvrir en append — reprend sans écraser si crash
    with open(out_file, "a", encoding="utf-8") as out_fh:
        for i, pf in enumerate(todo):

            # Détecter colonne texte
            text_col = detect_text_col(pf)
            if text_col is None:
                print(f"  [{i+1}/{len(todo)}] SKIP {pf.name} — colonne texte introuvable")
                mark_progress(progress_file, pf.name)
                continue

            print(f"  [{i+1}/{len(todo)}] {pf.name} (col={text_col}) ...",
                  end=" ", flush=True)

            # Lancer le subprocess
            cmd = [
                sys.executable, str(worker_script),
                str(pf), text_col, source,
                str(params["min_chars"]),
                str(params["min_words"]),
                str(params["min_doc_lines"]),
                str(params["max_special_char_ratio"]),
                str(params["max_digit_ratio"]),
                str(params["max_uppercase_ratio"]),
                str(params["fr_ratio_threshold"]),
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=args.timeout,
                )
                # Décoder en utf-8 avec remplacement — évite tout crash d'encoding
                stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
                stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""

            except subprocess.TimeoutExpired:
                print(f"TIMEOUT ({args.timeout}s)")
                errors.append(pf.name)
                continue
            except Exception as e:
                print(f"ERREUR subprocess : {e}")
                errors.append(pf.name)
                continue

            # Traiter la sortie
            if "FAIL" in stderr and result.returncode != 0:
                print(f"ERREUR")
                for line in stderr.splitlines():
                    print(f"    {line}")
                errors.append(pf.name)
                continue

            # Écrire les docs valides
            n_written = 0
            n_skip_json = 0
            for line in stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)   # validation JSON
                    out_fh.write(line + "\n")
                    n_written += 1
                except json.JSONDecodeError:
                    n_skip_json += 1

            out_fh.flush()

            # Extraire stats du stderr
            n_skipped = 0
            for s in stderr.splitlines():
                if s.startswith("OK "):
                    parts = s.split()
                    # "OK 1234 skipped=5678"
                    try:
                        n_skipped = int(parts[2].split("=")[1])
                    except (IndexError, ValueError):
                        pass
                    break
                if "WARN" in s:
                    print(f"\n    {s}", end="")

            total_docs    += n_written
            total_skipped += n_skipped
            mark_progress(progress_file, pf.name)

            pct = n_written / max(n_written + n_skipped, 1) * 100
            print(f"{n_written:,} docs ({pct:.0f}% rétention) | total={total_docs:,}")

    # Nettoyage worker
    worker_script.unlink(missing_ok=True)

    # Stats finales
    print()
    print("=" * 60)
    print(f"[{source}] Cleaning terminé")
    print(f"  Parquets traités : {len(todo) - len(errors)}/{len(todo)}")
    print(f"  Documents gardés : {total_docs:,}")
    print(f"  Documents rejetés: {total_skipped:,}")
    if total_docs + total_skipped > 0:
        print(f"  Taux rétention   : {total_docs/(total_docs+total_skipped)*100:.1f}%")
    print(f"  Output           : {out_file}")
    if out_file.exists():
        print(f"  Taille fichier   : {out_file.stat().st_size/1024**3:.2f} Go")
    if errors:
        print(f"  Erreurs ({len(errors)}) : {errors}")
    print("=" * 60)

    # Marquer done seulement si pas d'erreurs bloquantes
    if not errors:
        done_flag.touch()
        print(f"  .done créé → prêt pour déduplication globale")
    else:
        print(f"  ATTENTION : {len(errors)} parquets en erreur — .done NON créé")
        print(f"  Relancer sans --force pour retry uniquement les erreurs")


if __name__ == "__main__":
    main()
