#!/usr/bin/env python3
"""
RODIN Phase 1 — Téléchargement Gallica BNF via API SRU
=======================================================
Télécharge les textes en domaine public depuis Gallica (BNF)
via l'API SRU publique. Pas de compte ni de token requis.

Rate limit BNF : 1 requête/sec max — ce script respecte ce seuil.
Durée estimée : plusieurs heures à plusieurs jours selon le volume.
À lancer en overnight. Reprend là où il s'est arrêté.

Corpus couverts :
  - Littérature (romans, poèmes, théâtre)
  - Histoire
  - Philosophie
  - Sciences
  - Médecine
  - Géographie
  - Religion
  → ~5-8B tokens estimés, ~30-40Go texte brut

Output :
  - Fichiers .txt individuels dans RAW_DIR/
  - progress.json pour reprise automatique
  - Logs console + fichier

Usage :
    python scripts\\06_download_gallica.py
    python scripts\\06_download_gallica.py --out-dir D:\\data\\rodin\\raw\\gallica
    python scripts\\06_download_gallica.py --dry-run    # liste les requêtes sans DL
    python scripts\\06_download_gallica.py --max-docs 10000  # limiter pour test
    python scripts\\06_download_gallica.py --resume    # forcer reprise (défaut déjà)
"""

import argparse
import io
import json
import logging
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ─── Chemins par défaut ───────────────────────────────────────────────────────
# Gallica est petit (~30-40Go) → D: convient une fois libéré.
# Override avec --out-dir si besoin.

DEFAULT_OUT_DIR = Path(r"G:\data\rodin\raw\gallica")
DEFAULT_LOG_DIR = Path(r"G:\data\rodin\logs")

# ─── Corpus de requêtes SRU ───────────────────────────────────────────────────
# Chaque requête est une facette thématique du fonds BNF en français.
# Plafond à 10 000 résultats par requête (limite API SRU Gallica).
# Avec 50 docs/page → 200 pages max par requête.
# On déroule toutes les pages jusqu'à épuisement ou max_docs.

QUERIES = [
    # Littérature
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "littérature"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "roman"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "poésie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "théâtre"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "contes"',
    # Histoire
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "histoire"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "révolution française"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "guerre"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "biographie"',
    # Sciences & philosophie
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "philosophie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "sciences"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "mathématiques"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "physique"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "chimie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "botanique"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "astronomie"',
    # Médecine & santé
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "médecine"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "chirurgie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "pharmacie"',
    # Géographie & voyages
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "géographie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "voyage"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "exploration"',
    # Droit & économie
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "droit"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "économie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "politique"',
    # Religion & culture
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "religion"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "musique"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "art"',
    # Périodiques & encyclopédies (grands volumes)
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "encyclopédie"',
    'dc.language all "fre" and gallica.type all "text" and dc.subject all "dictionnaire"',
]

# ─── Setup logging ────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"gallica_{datetime.now():%Y%m%d_%H%M%S}.log"
    stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(stdout_utf8),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return logging.getLogger("rodin.gallica")


# ─── Core download ────────────────────────────────────────────────────────────

def download_gallica(out_dir: Path, log: logging.Logger,
                     dry_run: bool = False,
                     max_docs: int = 0) -> int:
    """
    Télécharge les textes Gallica via API SRU.
    Retourne le nombre total de docs téléchargés (inclus les sessions précédentes).
    """
    try:
        import requests
    except ImportError:
        log.error("pip install requests")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    progress_file = out_dir / "progress.json"

    # Charger la progression existante
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            done_arks = set(data.get("done", []))
        except Exception:
            done_arks = set()
    else:
        done_arks = set()

    log.info(f"Gallica — {len(done_arks)} docs déjà téléchargés (reprise automatique)")
    log.info(f"Output : {out_dir}")
    log.info(f"Requêtes SRU : {len(QUERIES)}")

    if dry_run:
        log.info("[DRY-RUN] Requêtes qui seraient exécutées :")
        for i, q in enumerate(QUERIES, 1):
            log.info(f"  {i:02d}. {q[:90]}...")
        return len(done_arks)

    session = requests.Session()
    session.headers["User-Agent"] = "RODIN-Research/1.0 (academic, open-source LLM FR)"
    SRU_BASE = "https://gallica.bnf.fr/SRU"

    new_docs      = 0
    total_errors  = 0
    last_save     = time.time()
    SAVE_INTERVAL = 500  # sauvegarder le progress toutes les N docs

    def save_progress():
        progress_file.write_text(
            json.dumps({"done": list(done_arks), "total": len(done_arks)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for q_idx, query in enumerate(QUERIES):
        log.info(f"[{q_idx+1}/{len(QUERIES)}] {query[:80]}...")
        start        = 1
        q_docs       = 0
        consec_errors = 0

        while start <= 10_000:
            # Vérifier le plafond global
            if max_docs > 0 and len(done_arks) >= max_docs:
                log.info(f"  Plafond --max-docs {max_docs} atteint — arrêt.")
                save_progress()
                return len(done_arks)

            params = {
                "operation":      "searchRetrieve",
                "version":        "1.2",
                "query":          query,
                "startRecord":    start,
                "maximumRecords": 50,
                "recordSchema":   "dc",
            }

            try:
                resp = session.get(SRU_BASE, params=params, timeout=30)
                resp.raise_for_status()
                consec_errors = 0
            except Exception as e:
                consec_errors += 1
                total_errors  += 1
                log.warning(f"  Erreur API (consécutif={consec_errors}) : {e}")
                if consec_errors > 5:
                    log.warning("  Trop d'erreurs consécutives — passage à la requête suivante.")
                    break
                time.sleep(15)
                continue

            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as e:
                log.warning(f"  XML invalide à start={start} : {e}")
                start += 50
                time.sleep(1.1)
                continue

            ns = {
                "srw": "http://www.loc.gov/zing/srw/",
                "dc":  "http://purl.org/dc/elements/1.1/",
            }
            records = root.findall(".//srw:record", ns)
            if not records:
                break  # Pas plus de résultats pour cette requête

            for record in records:
                id_el = record.find(".//dc:identifier", ns)
                if id_el is None or not id_el.text:
                    continue
                ark_url = id_el.text.strip()
                if "gallica.bnf.fr" not in ark_url:
                    continue

                ark_id = ark_url.rstrip("/").split("/")[-1]
                if ark_id in done_arks:
                    continue

                # Télécharger le texte brut
                try:
                    r = session.get(f"{ark_url}.texteBrut", timeout=30)
                    if r.status_code == 200 and len(r.content) > 500:
                        dest = out_dir / f"{ark_id}.txt"
                        dest.write_bytes(r.content)
                        done_arks.add(ark_id)
                        new_docs += 1
                        q_docs   += 1

                        if new_docs % 100 == 0:
                            log.info(
                                f"  {new_docs} nouveaux | {len(done_arks)} total | "
                                f"requête {q_idx+1}/{len(QUERIES)}"
                            )
                        if time.time() - last_save > 60 or new_docs % SAVE_INTERVAL == 0:
                            save_progress()
                            last_save = time.time()

                except Exception as e:
                    # Erreur sur un doc individuel → on skip, on continue
                    pass

                time.sleep(1.1)  # Rate limit BNF strict : 1 req/sec

            start += 50

        log.info(f"  Requête {q_idx+1} terminée — {q_docs} nouveaux docs")

    save_progress()
    log.info(f"[OK] Gallica terminé — {new_docs} nouveaux docs | {len(done_arks)} total")

    # Taille totale
    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.txt"))
    log.info(f"  Taille totale : {total_bytes/1024**3:.2f} Go")

    return len(done_arks)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN — Téléchargement Gallica BNF via API SRU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out-dir",   default=str(DEFAULT_OUT_DIR),
                   help=f"Dossier de sortie (défaut: {DEFAULT_OUT_DIR})")
    p.add_argument("--log-dir",   default=str(DEFAULT_LOG_DIR),
                   help=f"Dossier logs (défaut: {DEFAULT_LOG_DIR})")
    p.add_argument("--dry-run",   action="store_true",
                   help="Liste les requêtes sans télécharger")
    p.add_argument("--max-docs",  type=int, default=0,
                   help="Plafond total de docs (0 = illimité)")
    p.add_argument("--resume",    action="store_true", default=True,
                   help="Reprendre depuis progress.json (défaut: toujours actif)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    log     = setup_logging(log_dir)

    log.info("=" * 60)
    log.info("RODIN — Gallica BNF download")
    log.info("=" * 60)
    log.info(f"Output  : {out_dir}")
    log.info(f"Dry-run : {args.dry_run}")
    if args.max_docs:
        log.info(f"Max docs : {args.max_docs}")

    total = download_gallica(
        out_dir=out_dir,
        log=log,
        dry_run=args.dry_run,
        max_docs=args.max_docs,
    )

    log.info(f"Total docs Gallica : {total:,}")


if __name__ == "__main__":
    main()
