#!/usr/bin/env python3
"""
RODIN Phase 1 — Download + Clean HPLT extra (2596 parquets restants) sur G:
=============================================================================
Problème :
  - snapshot_download() télécharge TOUT le repo avant de filtrer → on obtiendrait
    les 3602 parquets d'un coup, dont les 1006 déjà traités.
  - Les 1006 parquets déjà cleanés ont leurs raw supprimés → la détection de
    "déjà fait" doit se baser sur le .progress du clean, pas sur la présence du raw.

Solution :
  - Lister les parquets disponibles sur HF via list_repo_tree()
  - Filtrer ceux déjà dans le .progress du clean existant (G:\\cleaned\\hplt\\.progress)
  - Télécharger un parquet → clean inline → supprimer raw → suivant

Architecture :
  - Download : huggingface_hub.hf_hub_download() fichier par fichier
  - Clean    : subprocess isolé (même worker que 04_clean_hplt_inline.py)
  - Output   : G:\\data\\rodin\\cleaned\\hplt\\hplt_cleaned.jsonl (mode append)
  - Progress : G:\\data\\rodin\\cleaned\\hplt\\.progress (partagé avec script 04)

Usage :
    python scripts\\05_download_clean_hplt_extra.py
    python scripts\\05_download_clean_hplt_extra.py --dry-run
    python scripts\\05_download_clean_hplt_extra.py --no-delete   # garde les raw
    python scripts\\05_download_clean_hplt_extra.py --workers 2   # dl en parallèle (expérimental)
"""

import argparse
import json
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# ─── Paths (tout sur G:) ──────────────────────────────────────────────────────

BASE_DIR  = Path(r"G:\data\rodin")
RAW_DIR   = BASE_DIR / "raw"  / "hplt" / "fra_Latn"
OUT_DIR   = BASE_DIR / "cleaned" / "hplt"
SCRIPTS   = BASE_DIR / "scripts"
LOG_DIR   = BASE_DIR / "logs"

HF_REPO   = "HPLT/HPLT2.0_cleaned"
HF_SUBDIR = "fra_Latn"   # sous-dossier du repo HF contenant les parquets FR
HF_TOKEN = os.environ.get("HF_TOKEN")

# ─── Filtres qualité HPLT (identiques à 04_clean_hplt_inline.py) ─────────────

HPLT_PARAMS = {
    "min_chars":               200,
    "min_words":                30,
    "min_doc_lines":             1,
    "max_special_char_ratio":  0.15,
    "max_digit_ratio":         0.18,
    "max_uppercase_ratio":     0.20,
    "fr_ratio_threshold":      0.05,
}

# ─── Worker subprocess (identique à 04_clean_hplt_inline.py) ─────────────────

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

    TEXT_COLS = ["text", "content", "raw_text"]

    try:
        schema = pq.read_schema(pf_path)
        col = next((c for c in TEXT_COLS if c in schema.names), None)
        if col is None:
            print(f"FAIL: aucune colonne texte — colonnes: {schema.names}", file=sys.stderr)
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
    """Retourne l'ensemble des noms de fichiers déjà traités."""
    if progress_file.exists():
        return set(progress_file.read_text(encoding="utf-8").splitlines())
    return set()


def mark_progress(progress_file: Path, filename: str):
    with open(progress_file, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def list_hplt_parquets(token: str) -> list[str]:
    """
    Liste tous les parquets fra_Latn disponibles sur HuggingFace.
    Retourne une liste de chemins relatifs type 'fra_Latn/train-00000-of-03602.parquet'.
    Utilise list_repo_tree() (huggingface_hub >= 0.20).
    """
    try:
        from huggingface_hub import list_repo_tree
    except ImportError:
        # Fallback : list_repo_files (huggingface_hub < 0.20)
        from huggingface_hub import list_repo_files
        all_files = list(list_repo_files(
            HF_REPO, repo_type="dataset", token=token
        ))
        return sorted(f for f in all_files
                      if f.startswith(f"{HF_SUBDIR}/") and f.endswith(".parquet"))

    items = list_repo_tree(
        HF_REPO,
        path_in_repo=HF_SUBDIR,
        repo_type="dataset",
        token=token,
        recursive=False,
    )
    return sorted(
        item.path for item in items
        if hasattr(item, "path") and item.path.endswith(".parquet")
    )


def download_one(hf_path: str, dest_dir: Path, token: str, max_retries: int = 5) -> Path | None:
    """
    Télécharge un seul parquet depuis HF vers dest_dir.
    Retourne le Path local, ou None si échec.
    hf_path : chemin relatif dans le repo, ex: 'fra_Latn/train-00000-of-03602.parquet'
    """
    from huggingface_hub import hf_hub_download
    filename = Path(hf_path).name
    dest_file = dest_dir / filename

    if dest_file.exists() and dest_file.stat().st_size > 10_000:
        return dest_file  # déjà présent (reprise crash mid-download)

    for attempt in range(1, max_retries + 1):
        try:
            local = hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=hf_path,
                local_dir=str(dest_dir),
                local_dir_use_symlinks=False,
                token=token,
            )
            return Path(local)
        except Exception as e:
            wait = min(30 * attempt, 180)
            print(f"    Erreur download (tentative {attempt}/{max_retries}) : {e}")
            if attempt < max_retries:
                print(f"    Retry dans {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Abandon après {max_retries} tentatives.")
                return None
    return None


def clean_one(pf: Path, worker_script: Path, out_fh, timeout: int) -> tuple[int, int, list[str]]:
    """
    Lance le worker subprocess sur un parquet, écrit le jsonl dans out_fh.
    Retourne (n_written, n_skipped, errors).
    """
    p = HPLT_PARAMS
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
        result = subprocess.run(cmd, capture_output=True, timeout=timeout)
        stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    except subprocess.TimeoutExpired:
        return 0, 0, [f"TIMEOUT {pf.name}"]
    except Exception as e:
        return 0, 0, [f"SUBPROCESS_ERR {pf.name}: {e}"]

    if "FAIL" in stderr and result.returncode != 0:
        errs = [l for l in stderr.splitlines() if l.strip()]
        return 0, 0, errs

    n_written = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            out_fh.write(line + "\n")
            n_written += 1
        except json.JSONDecodeError:
            pass
    out_fh.flush()

    n_skipped = 0
    for s in stderr.splitlines():
        if s.startswith("OK "):
            try:
                n_skipped = int(s.split()[2].split("=")[1])
            except (IndexError, ValueError):
                pass
            break

    return n_written, n_skipped, []


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN — Download + Clean HPLT extra (2596 parquets) sur G:"
    )
    p.add_argument("--dry-run",   action="store_true",
                   help="Simule sans télécharger ni écrire")
    p.add_argument("--no-delete", action="store_true",
                   help="Ne supprime pas les raw après clean (debug)")
    p.add_argument("--timeout",   type=int, default=600,
                   help="Timeout subprocess clean par parquet (défaut: 600s)")
    p.add_argument("--hf-token",  default=HF_TOKEN,
                   help="Token HuggingFace (défaut: token embarqué)")
    p.add_argument("--max-retries", type=int, default=5,
                   help="Tentatives download par parquet (défaut: 5)")
    p.add_argument("--limit",     type=int, default=0,
                   help="Limiter à N parquets (debug, 0 = tous)")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    token = args.hf_token

    # Créer les dossiers
    for d in [RAW_DIR, OUT_DIR, SCRIPTS, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    out_file      = OUT_DIR / "hplt_cleaned.jsonl"
    done_flag     = OUT_DIR / ".done"
    progress_file = OUT_DIR / ".progress"
    worker_script = SCRIPTS / "_worker_hplt_extra.py"

    # Bloquer si .done présent (le cleaning 04 est terminé ET ce script aussi)
    if done_flag.exists():
        print("[hplt-extra] .done présent — HPLT déjà entièrement terminé.")
        print("  Si tu veux relancer, supprime G:\\data\\rodin\\cleaned\\hplt\\.done")
        return

    print("[hplt-extra] Listage des parquets disponibles sur HuggingFace...")
    print(f"  Repo : {HF_REPO}/{HF_SUBDIR}")

    if args.dry_run:
        print("  [DRY-RUN] Pas de connexion HF en dry-run — estimation seule.")
        print("  2596 parquets restants estimés (~1.1To raw, ~143B tokens nets)")
        return

    try:
        all_hf_paths = list_hplt_parquets(token)
    except Exception as e:
        print(f"ERREUR listing HuggingFace : {e}")
        print("  Vérifier le token HF et la connexion réseau (ProtonVPN fixe ?)")
        sys.exit(1)

    print(f"  {len(all_hf_paths)} parquets FR disponibles sur HF au total")

    # Charger le .progress existant (partagé avec 04_clean_hplt_inline.py)
    done_names = get_progress(progress_file)
    print(f"  {len(done_names)} parquets déjà dans .progress (traités par script 04)")

    # Filtrer les parquets à traiter : ceux dont le nom n'est pas dans .progress
    todo = [p for p in all_hf_paths if Path(p).name not in done_names]

    if args.limit > 0:
        todo = todo[:args.limit]
        print(f"  Limité à {args.limit} parquets (--limit)")

    print(f"  {len(todo)} parquets restants à télécharger + cleaner")
    print(f"  Raw dir  : {RAW_DIR}")
    print(f"  Output   : {out_file} (mode append)")
    print(f"  Espace G: libre : {free_gb(BASE_DIR):.1f} Go")
    print()

    if not todo:
        print("[hplt-extra] Rien à faire — tous les parquets HF sont déjà dans .progress.")
        done_flag.touch()
        print("  .done créé ✅")
        return

    if not args.no_delete:
        print("⚠️  ATTENTION : les parquets raw seront supprimés après chaque clean.")
        print("   C'est voulu pour économiser l'espace disque sur G:.")
        confirm = input("   Confirmer ? (oui/non) : ").strip().lower()
        if confirm not in ("oui", "o", "yes", "y"):
            print("Annulé.")
            return
        print()

    # Écrire le worker
    worker_script.write_text(WORKER_CODE, encoding="utf-8")

    total_docs    = 0
    total_skipped = 0
    total_deleted = 0
    dl_errors     = []
    clean_errors  = []

    fh = open(out_file, "a", encoding="utf-8")

    try:
        for i, hf_path in enumerate(todo):
            filename = Path(hf_path).name
            free = free_gb(BASE_DIR)
            print(f"  [{i+1}/{len(todo)}] {filename} | G: libre={free:.1f}Go")

            # ── Vérification espace disque minimal (1 parquet ~ 500Mo raw + marge)
            if free < 2.0:
                print(f"    ⚠️  STOP : moins de 2Go libres sur G: — risque de corruption.")
                print(f"    Libérer de l'espace et relancer.")
                break

            # ── Download
            print(f"    Téléchargement...", end=" ", flush=True)
            t_dl = time.time()
            local_pf = download_one(hf_path, RAW_DIR, token, args.max_retries)
            if local_pf is None:
                print(f"ÉCHEC DOWNLOAD")
                dl_errors.append(filename)
                continue
            dl_sec = time.time() - t_dl
            dl_mb  = local_pf.stat().st_size / 1024**2
            print(f"OK ({dl_mb:.0f}Mo en {dl_sec:.0f}s)")

            # ── Clean
            print(f"    Cleaning...", end=" ", flush=True)
            t_cl = time.time()
            n_written, n_skipped, errs = clean_one(local_pf, worker_script, fh, args.timeout)
            cl_sec = time.time() - t_cl

            if errs:
                print(f"ÉCHEC CLEAN")
                for e in errs:
                    print(f"      {e}")
                clean_errors.append(filename)
                # Ne pas supprimer le raw en cas d'erreur
                continue

            pct = n_written / max(n_written + n_skipped, 1) * 100
            total_docs    += n_written
            total_skipped += n_skipped

            # ── Supprimer raw
            deleted_str = ""
            if not args.no_delete:
                try:
                    local_pf.unlink()
                    total_deleted += 1
                    deleted_str = " | raw ✓ supprimé"
                except Exception as e:
                    deleted_str = f" | ERREUR suppression: {e}"

            free_after = free_gb(BASE_DIR)
            print(
                f"    → {n_written:,} docs ({pct:.0f}%) en {cl_sec:.0f}s"
                f"{deleted_str} | total={total_docs:,} | G: libre={free_after:.1f}Go"
            )

            mark_progress(progress_file, filename)

    finally:
        fh.close()
        if worker_script.exists():
            worker_script.unlink(missing_ok=True)

    # ── Stats finales
    print()
    print("=" * 70)
    print(f"[hplt-extra] Terminé")
    print(f"  Parquets traités   : {len(todo) - len(dl_errors) - len(clean_errors)}/{len(todo)}")
    print(f"  Parquets supprimés : {total_deleted}")
    print(f"  Documents gardés   : {total_docs:,}")
    print(f"  Documents rejetés  : {total_skipped:,}")
    if total_docs + total_skipped > 0:
        print(f"  Taux rétention     : {total_docs/(total_docs+total_skipped)*100:.1f}%")
    if out_file.exists():
        print(f"  Taille jsonl       : {out_file.stat().st_size/1024**3:.2f} Go")
    if dl_errors:
        print(f"  Erreurs download ({len(dl_errors)}) : {dl_errors[:5]}{'...' if len(dl_errors)>5 else ''}")
    if clean_errors:
        print(f"  Erreurs clean    ({len(clean_errors)}) : {clean_errors[:5]}{'...' if len(clean_errors)>5 else ''}")
    print("=" * 70)

    all_errors = dl_errors + clean_errors
    if not all_errors:
        done_flag.touch()
        print("  .done créé ✅ — HPLT complet")
    else:
        print(f"  {len(all_errors)} erreurs — .done NON créé")
        print("  Relancer le script pour retry (les erreurs ne sont pas dans .progress)")


if __name__ == "__main__":
    main()
