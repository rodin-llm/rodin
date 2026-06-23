#!/usr/bin/env python3
"""
RODIN Phase 1 — Téléchargement sources dataset FR
==================================================
Sources vérifiées, toutes accessibles sans friction :

  wikipedia   -> dump Wikimedia HTTP direct        (~6.5 Go compressé / ~22 Go XML)
  wikisource  -> dump Wikimedia HTTP direct        (~1 Go compressé / ~5 Go XML)
  cc100       -> statmt.org HTTP direct            (~50 Go compressé / ~300 Go texte)
  legifrance  -> DILA OpenData HTTP direct         (~10 Go)
  gallica     -> API SRU BNF HTTP direct           (~50 Go, long)
  pleia_books -> PleIAs/French-PD-Books HF libre   (~100 Go, 289K livres BNF)
  pleia_news  -> PleIAs/French-PD-Newspapers HF   (~200 Go, 3M journaux BNF)
  hplt        -> HPLT/HPLT2.0_cleaned HF libre    (~100 Go FR web crawl)
  mc4         -> allenai/c4 config=fr via datasets (~35 Go)

Total : ~570 Go brut -> ~100B tokens propres apres cleaning

Ordre recommande :
  1. wikipedia   (deja fait normalement)
  2. cc100       (gros volume FR pur, HTTP direct, sans compte)
  3. pleia_books (BNF livres, HF libre, sans gate)
  4. pleia_news  (BNF journaux, HF libre, sans gate)
  5. hplt        (web crawl FR, HF libre, sans gate)
  6. mc4         (Common Crawl FR via datasets, sans gate)
  7. wikisource  (petit, HTTP direct)
  8. legifrance  (juridique, HTTP direct)
  9. gallica     (lent, API BNF, lancer la nuit)

Usage:
    python 01_download.py --list
    python 01_download.py --source wikipedia
    python 01_download.py --source cc100
    python 01_download.py --source pleia_books,pleia_news,hplt
    python 01_download.py --source mc4
    python 01_download.py --source all
    python 01_download.py --dry-run
"""

import io
import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(r"D:\data\rodin")
RAW_DIR  = BASE_DIR / "raw"
LOG_DIR  = BASE_DIR / "logs"

for _d in [RAW_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Logging UTF-8 Windows-safe ───────────────────────────────────────────────

_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(_stdout_utf8),
        logging.FileHandler(
            LOG_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("rodin.download")

# ─── Sources figees et verifiees ──────────────────────────────────────────────

SOURCES = {

    "wikipedia": {
        "description": "Wikipedia FR — dump Wikimedia officiel",
        "method": "http",
        "dir": RAW_DIR / "wikipedia",
        "url": "https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-pages-articles.xml.bz2",
        "filename": "frwiki-latest-pages-articles.xml.bz2",
        "size_gb": 6.5,
    },

    "wikisource": {
        "description": "Wikisource FR — dump Wikimedia officiel",
        "method": "http",
        "dir": RAW_DIR / "wikisource",
        "url": "https://dumps.wikimedia.org/frwikisource/latest/frwikisource-latest-pages-articles.xml.bz2",
        "filename": "frwikisource-latest-pages-articles.xml.bz2",
        "size_gb": 1.0,
    },

    "cc100": {
        "description": "CC-100 FR — Common Crawl monolingue FR (~300 Go texte)",
        "method": "http",
        "dir": RAW_DIR / "cc100",
        "url": "http://data.statmt.org/cc-100/fr.txt.xz",
        "filename": "fr.txt.xz",
        "size_gb": 50,
    },

    # FIX : mc4 utilise load_dataset (config "fr") au lieu de snapshot_download.
    # snapshot_download avec allow_patterns=["fr/*"] doit lister TOUT le repo
    # avant de filtrer -> timeout apres 7000+ requetes de pagination.
    # load_dataset(repo, config="fr") cible directement les shards FR
    # via les metadata du dataset sans enumerer le reste.
    "mc4": {
        "description": "mC4 FR — Common Crawl filtre par Google (~35 Go)",
        "method": "hf_datasets",
        "dir": RAW_DIR / "mc4",
        "hf_repo": "allenai/c4",
        "hf_config": "fr",
        "size_gb": 35,
    },

    "legifrance": {
        "description": "Legifrance LEGI+JORF+JADE+KALI — stocks globaux DILA OpenData (mars 2026)",
        "method": "legi",
        "dir": RAW_DIR / "legifrance",
        "size_gb": 5,
        # IMPORTANT : DILA ne publie plus d'archives multi-periodes.
        # Il n'existe qu'UN SEUL stock global par base, regenere periodiquement.
        # Nommage : Freemium_<base>_global_<date>-<heure>.tar.gz
        # Pour trouver les URLs exactes du moment :
        #   https://echanges.dila.gouv.fr/OPENDATA/LEGI/
        #   https://echanges.dila.gouv.fr/OPENDATA/JORF/
        #   https://echanges.dila.gouv.fr/OPENDATA/JADE/
        #   https://echanges.dila.gouv.fr/OPENDATA/KALI/
        # Les archives sont conservees 62 jours — les URLs ci-dessous
        # EXPIRERONT. Verifier le listing HTTP si 404.
        "urls": [
            # LEGI : codes de loi consolides (~1.1 Go compresse)
            "https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20250713-140000.tar.gz",
            # JORF : Journal Officiel de la Republique Francaise
            "https://echanges.dila.gouv.fr/OPENDATA/JORF/Freemium_jorf_global_20250713-140000.tar.gz",
            # JADE : jurisprudence administrative (~1.2 Go compresse)
            "https://echanges.dila.gouv.fr/OPENDATA/JADE/Freemium_jade_global_20250713-140000.tar.gz",
            # KALI : conventions collectives
            "https://echanges.dila.gouv.fr/OPENDATA/KALI/Freemium_kali_global_20250713-140000.tar.gz",
            # CASS : jurisprudence Cour de cassation (~248 Mo)
            "https://echanges.dila.gouv.fr/OPENDATA/CASS/Freemium_cass_global_20250713-140000.tar.gz",
        ],
    },

    "gallica": {
        "description": "Gallica BNF — textes domaine public via API SRU (lent)",
        "method": "gallica",
        "dir": RAW_DIR / "gallica",
        "size_gb": 50,
    },

    "pleia_books": {
        "description": "PleIAs/French-PD-Books — 289K livres BNF domaine public",
        "method": "huggingface",
        "dir": RAW_DIR / "pleia_books",
        "hf_repo": "PleIAs/French-PD-Books",
        "size_gb": 100,
    },

    "pleia_news": {
        "description": "PleIAs/French-PD-Newspapers — 3M journaux BNF domaine public",
        "method": "huggingface",
        "dir": RAW_DIR / "pleia_news",
        "hf_repo": "PleIAs/French-PD-Newspapers",
        "size_gb": 200,
    },

    # FIX : HPLT2.0_cleaned structure reelle = "data/fr/..." pas "fr/..."
    # On essaie plusieurs patterns pour couvrir les deux structures possibles.
    # Si les deux echouent, fallback sans pattern (tout le repo, filtrage manuel).
    "hplt": {
        "description": "HPLT 2.0 FR cleaned — web crawl FR deduplique et nettoye",
        "method": "huggingface",
        "dir": RAW_DIR / "hplt",
        "hf_repo": "HPLT/HPLT2.0_cleaned",
        "hf_patterns": ["fra_Latn/*"],
        "size_gb": 10,
    },
}

# ─── Utilitaires ──────────────────────────────────────────────────────────────

def check_disk_space(path: Path, required_gb: float) -> bool:
    import shutil
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    log.info(f"Espace libre : {free_gb:.1f} Go — requis : {required_gb:.1f} Go")
    if free_gb < required_gb * 1.05:
        log.error(f"Espace insuffisant ! {free_gb:.1f} Go libre, {required_gb:.1f} Go requis.")
        return False
    return True


def http_file_done(dest_dir: Path, filename: str, size_gb: float) -> bool:
    """Retourne True si le fichier est deja telecharge (>90% taille attendue)."""
    f = dest_dir / filename
    if f.exists():
        actual = f.stat().st_size / (1024 ** 3)
        if actual >= size_gb * 0.9:
            log.info(f"Deja present : {filename} ({actual:.2f} Go) — skip")
            return True
        log.info(f"Fichier partiel ({actual:.2f} / {size_gb:.1f} Go) — reprise")
    return False


def hf_done(dest_dir: Path) -> bool:
    """Retourne True si des fichiers HF sont deja presents."""
    for ext in ["*.parquet", "*.jsonl", "*.arrow", "*.jsonl.gz", "*.json.gz"]:
        files = list(dest_dir.rglob(ext))
        if files:
            total_gb = sum(f.stat().st_size for f in files) / (1024 ** 3)
            log.info(f"Deja present : {len(files)} fichiers ({total_gb:.1f} Go) — skip")
            return True
    return False

# ─── Downloader HTTP avec reprise ─────────────────────────────────────────────

def download_http(cfg: dict) -> bool:
    """
    Telechargement HTTP direct avec reprise automatique via Range header.
    Zero dependance externe hormis requests.
    """
    try:
        import requests
    except ImportError:
        log.error("pip install requests")
        return False

    dest_dir  = cfg["dir"]
    filename  = cfg["filename"]
    url       = cfg["url"]
    size_gb   = cfg["size_gb"]

    dest_dir.mkdir(parents=True, exist_ok=True)

    if http_file_done(dest_dir, filename, size_gb):
        return True

    if not check_disk_space(dest_dir, size_gb):
        return False

    dest_file = dest_dir / filename
    CHUNK     = 8 * 1024 * 1024  # 8 Mo
    MAX_RETRY = 10
    session   = requests.Session()
    session.headers["User-Agent"] = "RODIN-Research/1.0 (academic)"

    log.info(f"URL  : {url}")
    log.info(f"Dest : {dest_file}")

    for attempt in range(1, MAX_RETRY + 1):
        resume_pos = dest_file.stat().st_size if dest_file.exists() else 0
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

        try:
            resp = session.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 416:
                log.info("Fichier deja complet (416 Range Not Satisfiable)")
                return True

            resp.raise_for_status()

            content_len = int(resp.headers.get("Content-Length", 0))
            total_bytes = (resume_pos + content_len) if resp.status_code == 206 else content_len
            total_gb    = total_bytes / (1024 ** 3) if total_bytes else size_gb

            mode       = "ab" if (resume_pos > 0 and resp.status_code == 206) else "wb"
            downloaded = resume_pos
            last_log   = time.time()

            with open(dest_file, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if time.time() - last_log >= 30:
                        pct = downloaded / max(total_bytes, 1) * 100
                        log.info(
                            f"  {downloaded/(1024**3):.2f} Go"
                            + (f" / {total_gb:.1f} Go ({pct:.1f}%)" if total_bytes else "")
                        )
                        last_log = time.time()

            final_gb = dest_file.stat().st_size / (1024 ** 3)
            log.info(f"[OK] {filename} — {final_gb:.2f} Go")
            return True

        except (requests.ConnectionError, requests.Timeout,
                requests.ChunkedEncodingError) as e:
            wait = min(30 * attempt, 300)
            log.warning(f"Erreur reseau (tentative {attempt}/{MAX_RETRY}) : {e}")
            if attempt < MAX_RETRY:
                log.info(f"Reprise dans {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Abandon apres {MAX_RETRY} tentatives.")
                return False
        except Exception as e:
            log.error(f"Erreur inattendue : {e}")
            return False

    return False

# ─── Downloader HuggingFace snapshot ─────────────────────────────────────────

def download_huggingface(cfg: dict, hf_token: str | None = None) -> bool:
    """
    Telechargement via huggingface_hub snapshot_download.
    Utilise pour les repos sans config HF (PleIAs, HPLT, etc.).
    NE PAS utiliser pour allenai/c4 — utiliser download_hf_datasets.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("pip install huggingface_hub")
        return False

    dest_dir = cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    repo_id  = cfg["hf_repo"]

    if hf_done(dest_dir):
        return True

    if not check_disk_space(dest_dir, cfg["size_gb"]):
        return False

    log.info(f"HuggingFace : {repo_id}")
    log.info(f"Destination : {dest_dir}")

    patterns = cfg.get("hf_patterns")
    if patterns:
        log.info(f"Filtre patterns : {patterns}")

    kwargs = dict(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    if hf_token:
        kwargs["token"] = hf_token
    if patterns:
        kwargs["allow_patterns"] = patterns

    try:
        snapshot_download(**kwargs)
        # Verifier qu'on a bien des fichiers apres le download
        files = list(dest_dir.rglob("*.parquet")) + list(dest_dir.rglob("*.jsonl"))
        if not files and patterns:
            # Patterns n'ont rien matche — retenter sans filtre
            log.warning(
                f"Aucun fichier telecharge avec patterns {patterns}.\n"
                f"  -> Tentative sans filtre (tout le repo sera telecharge).\n"
                f"  -> Les fichiers non-FR seront ignores au cleaning."
            )
            kwargs.pop("allow_patterns", None)
            snapshot_download(**kwargs)
        log.info(f"[OK] {repo_id}")
        return True
    except Exception as e:
        log.error(f"Echec HuggingFace ({repo_id}) : {e}")
        if "403" in str(e) or "gated" in str(e).lower():
            log.error(f"  -> Accepter les conditions sur : https://huggingface.co/datasets/{repo_id}")
        return False


# ─── Downloader HuggingFace datasets (pour repos avec configs) ───────────────

def download_hf_datasets(cfg: dict, hf_token: str | None = None) -> bool:
    """
    Telechargement via datasets.load_dataset pour les repos HF avec configs.

    Pourquoi pas snapshot_download ici :
      allenai/c4 contient ~7000 fichiers (EN, multilingual, FR).
      snapshot_download avec allow_patterns=["fr/*"] doit lister TOUT le repo
      avant de filtrer -> timeout apres 7000+ requetes de pagination.
      load_dataset(repo, config="fr") cible directement les shards FR
      via les metadata du dataset sans enumerer le reste.

    Le dataset est sauvegarde en parquet via ds.to_parquet() dans dest_dir.
    Cache HF dans dest_dir/.hf_cache/ pour eviter de polluer le cache global.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("pip install datasets")
        return False

    dest_dir = cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    if hf_done(dest_dir):
        return True

    if not check_disk_space(dest_dir, cfg["size_gb"]):
        return False

    repo   = cfg["hf_repo"]
    config = cfg.get("hf_config", "default")
    cache  = dest_dir / ".hf_cache"

    log.info(f"datasets.load_dataset({repo!r}, {config!r})")
    log.info(f"Destination  : {dest_dir}")
    log.info(f"Cache HF     : {cache}")
    log.info(f"Taille estimee : {cfg['size_gb']} Go")
    log.info("Le telechargement peut prendre plusieurs heures — soyez patient.")

    try:
        ds = load_dataset(
            repo,
            config,
            cache_dir=str(cache),
            token=hf_token,
            num_proc=4,
        )
        log.info(f"Dataset charge : {ds}")

        # Exporter en parquet dans dest_dir pour compatibilite avec le reste du pipeline
        # (le cleaning attend des .parquet ou .jsonl dans raw/{source}/)
        for split_name, split_ds in ds.items():
            out_file = dest_dir / f"{config}_{split_name}.parquet"
            log.info(f"Export parquet : {out_file} ({len(split_ds):,} exemples)")
            split_ds.to_parquet(str(out_file))
            size_gb = out_file.stat().st_size / (1024 ** 3)
            log.info(f"  -> {size_gb:.2f} Go")

        log.info(f"[OK] {repo} config={config}")
        return True

    except Exception as e:
        log.error(f"Echec load_dataset ({repo}, {config}) : {e}")
        return False


# ─── Downloader Legifrance DILA ───────────────────────────────────────────────

def _dila_discover_latest(session, base: str) -> str | None:
    """
    Tente de recuperer l'URL du dernier Freemium_<base>_global_*.tar.gz
    en parsant le listing HTTP du repertoire DILA.
    Retourne l'URL complete ou None si echec.
    """
    import re as _re
    dir_url = f"https://echanges.dila.gouv.fr/OPENDATA/{base.upper()}/"
    try:
        resp = session.get(dir_url, timeout=20)
        resp.raise_for_status()
        # Cherche les noms de fichiers Freemium_<base>_global_*.tar.gz
        pattern = _re.compile(
            rf'href="(Freemium_{base.lower()}_global_[\d\-]+\.tar\.gz)"',
            _re.IGNORECASE,
        )
        matches = pattern.findall(resp.text)
        if not matches:
            log.warning(f"  -> Aucune archive Freemium trouvee dans le listing {dir_url}")
            return None
        # Prendre la plus recente (tri lexicographique suffisant sur les timestamps)
        latest = sorted(matches)[-1]
        url = dir_url + latest
        log.info(f"  -> Auto-discovery {base.upper()} : {latest}")
        return url
    except Exception as e:
        log.warning(f"  -> Auto-discovery {base.upper()} impossible : {e}")
        return None


def download_legi(cfg: dict) -> bool:
    """
    Legifrance via DILA OpenData — archives tar.gz publiques.

    DILA ne publie plus d'archives multi-periodes.
    Il n'existe qu'UN stock global par base, regenere periodiquement.
    Les archives sont conservees 62 jours — URLs a mettre a jour si 404.

    Si une URL retourne 404, le script tente de decouvrir automatiquement
    la derniere archive disponible via le listing HTTP du repertoire DILA.
    """
    try:
        import requests
    except ImportError:
        log.error("pip install requests")
        return False

    dest_dir = cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    session  = requests.Session()
    session.headers["User-Agent"] = "RODIN-Research/1.0 (academic)"
    ok_count = 0
    CHUNK    = 8 * 1024 * 1024
    MAX_RETRY = 5

    for url in cfg["urls"]:
        filename  = url.split("/")[-1]
        dest_file = dest_dir / filename

        if dest_file.exists() and dest_file.stat().st_size > 500_000:
            log.info(f"Deja present : {filename} ({dest_file.stat().st_size/(1024**3):.2f} Go) — skip")
            ok_count += 1
            continue

        log.info(f"Verification : {url}")

        # ── HEAD check + auto-discovery si 404 ────────────────────────────
        resolved_url = url
        try:
            head = session.head(url, timeout=20, allow_redirects=True)
            if head.status_code == 404:
                log.warning(f"  404 : {filename} — tentative auto-discovery...")
                # Extraire le nom de base : LEGI, JORF, JADE, KALI, CASS...
                # Pattern filename : Freemium_<base>_global_<timestamp>.tar.gz
                import re as _re
                m = _re.match(r"Freemium_([a-z]+)_global_", filename, _re.IGNORECASE)
                if m:
                    base_name = m.group(1)
                    discovered = _dila_discover_latest(session, base_name)
                    if discovered:
                        resolved_url = discovered
                        filename     = discovered.split("/")[-1]
                        dest_file    = dest_dir / filename
                        log.info(f"  -> URL resolue : {resolved_url}")
                        if dest_file.exists() and dest_file.stat().st_size > 500_000:
                            log.info(f"  -> Deja present : {filename} — skip")
                            ok_count += 1
                            continue
                    else:
                        log.error(
                            f"  -> Auto-discovery echoue.\n"
                            f"  -> Lister manuellement : https://echanges.dila.gouv.fr/OPENDATA/{base_name.upper()}/\n"
                            f"  -> Telecharger le fichier Freemium_{base_name}_global_*.tar.gz\n"
                            f"  -> Le placer dans : {dest_dir}"
                        )
                        continue
                else:
                    log.error(f"  -> Impossible de determiner la base depuis : {filename}")
                    continue
        except Exception as e:
            log.warning(f"  -> HEAD impossible : {e} — tentative download direct")

        # ── Telechargement avec reprise ───────────────────────────────────
        log.info(f"Telechargement : {filename}")
        for attempt in range(1, MAX_RETRY + 1):
            resume_pos = dest_file.stat().st_size if dest_file.exists() else 0
            headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
            try:
                resp = session.get(resolved_url, headers=headers, stream=True, timeout=120)
                if resp.status_code == 416:
                    log.info(f"  Fichier deja complet (416) — ok")
                    ok_count += 1
                    break
                resp.raise_for_status()

                content_len = int(resp.headers.get("Content-Length", 0))
                total_bytes = (resume_pos + content_len) if resp.status_code == 206 else content_len
                mode        = "ab" if (resume_pos > 0 and resp.status_code == 206) else "wb"
                downloaded  = resume_pos
                last_log    = time.time()

                with open(dest_file, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if time.time() - last_log >= 30:
                                pct = downloaded / max(total_bytes, 1) * 100
                                mb  = downloaded / (1024 ** 2)
                                tot = total_bytes / (1024 ** 3) if total_bytes else 0
                                log.info(
                                    f"  {mb:.0f} Mo"
                                    + (f" / {tot:.2f} Go ({pct:.1f}%)" if total_bytes else "")
                                )
                                last_log = time.time()

                final_gb = dest_file.stat().st_size / (1024 ** 3)
                log.info(f"[OK] {filename} — {final_gb:.2f} Go")
                ok_count += 1
                break

            except (requests.ConnectionError, requests.Timeout,
                    requests.ChunkedEncodingError) as e:
                wait = min(30 * attempt, 300)
                log.warning(f"  Erreur reseau (tentative {attempt}/{MAX_RETRY}) : {e}")
                if attempt < MAX_RETRY:
                    log.info(f"  Reprise dans {wait}s...")
                    time.sleep(wait)
                else:
                    log.error(f"  Abandon : {filename}")
                    if dest_file.exists() and dest_file.stat().st_size < 50_000:
                        dest_file.unlink()

            except Exception as e:
                log.error(f"  Echec : {filename} — {e}")
                if dest_file.exists() and dest_file.stat().st_size < 50_000:
                    dest_file.unlink()
                break

    if ok_count == 0:
        log.error(
            "Legifrance : aucune archive telechargee.\n"
            "Diagnostic :\n"
            "  1. Verifier les listings DILA :\n"
            "     https://echanges.dila.gouv.fr/OPENDATA/LEGI/\n"
            "     https://echanges.dila.gouv.fr/OPENDATA/JORF/\n"
            "     https://echanges.dila.gouv.fr/OPENDATA/JADE/\n"
            "  2. Telecharger manuellement les Freemium_*_global_*.tar.gz\n"
            f"  3. Placer dans : {dest_dir}"
        )
        return False

    log.info(f"Legifrance : {ok_count}/{len(cfg['urls'])} archives OK")
    return True

# ─── Downloader Gallica BNF ───────────────────────────────────────────────────

def download_gallica(cfg: dict) -> bool:
    """
    Gallica BNF via API SRU — textes plein domaine public.
    Rate limit respecte : 1 requete/seconde.
    Reprise via fichier .progress.json.
    Source longue — lancer la nuit.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
    except ImportError:
        log.error("pip install requests")
        return False

    dest_dir      = cfg["dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    progress_file = dest_dir / ".progress.json"

    # Charger progression precedente
    done_arks: set = set()
    if progress_file.exists():
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            done_arks = set(data.get("done", []))
            log.info(f"Gallica reprise : {len(done_arks)} documents deja traites")
        except Exception:
            pass

    # Requetes par sujet — 10K resultats max par requete (limite API Gallica)
    QUERIES = [
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "litterature"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "histoire"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "philosophie"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "sciences"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "droit"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "medecine"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "geographie"',
        'dc.language all "fre" and gallica.type all "text" and dc.subject all "religion"',
    ]

    session = requests.Session()
    session.headers["User-Agent"] = "RODIN-Research/1.0 (academic)"
    SRU_BASE = "https://gallica.bnf.fr/SRU"
    total    = 0
    errors   = 0

    for query in QUERIES:
        log.info(f"Gallica requete : {query[:80]}...")
        start = 1

        while start <= 10000:
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
                errors = 0
            except Exception as e:
                errors += 1
                log.warning(f"Erreur API ({errors}) : {e}")
                if errors > 5:
                    log.warning("Trop d'erreurs consecutives — passage requete suivante")
                    break
                time.sleep(15)
                continue

            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                start += 50
                continue

            ns = {
                "srw": "http://www.loc.gov/zing/srw/",
                "dc":  "http://purl.org/dc/elements/1.1/",
            }
            records = root.findall(".//srw:record", ns)
            if not records:
                break  # Plus de resultats

            for record in records:
                id_el = record.find(".//dc:identifier", ns)
                if id_el is None or not id_el.text:
                    continue
                ark_url = id_el.text
                if "gallica.bnf.fr" not in ark_url:
                    continue

                ark_id = ark_url.rstrip("/").split("/")[-1]
                if ark_id in done_arks:
                    continue

                try:
                    r = session.get(f"{ark_url}.texteBrut", timeout=30)
                    if r.status_code == 200 and len(r.content) > 500:
                        (dest_dir / f"{ark_id}.txt").write_bytes(r.content)
                        done_arks.add(ark_id)
                        total += 1
                        if total % 500 == 0:
                            log.info(f"  Gallica : {total} docs telecharges")
                            progress_file.write_text(
                                json.dumps({"done": list(done_arks)}),
                                encoding="utf-8",
                            )
                except Exception:
                    pass

                time.sleep(1.1)  # Respect BNF rate limit

            start += 50

    # Sauvegarde finale
    progress_file.write_text(
        json.dumps({"done": list(done_arks)}),
        encoding="utf-8",
    )
    log.info(f"[OK] Gallica : {total} nouveaux docs, {len(done_arks)} total")
    return len(done_arks) > 0

# ─── Dispatcher ───────────────────────────────────────────────────────────────

def run_source(name: str, cfg: dict, hf_token: str | None = None) -> bool:
    log.info("")
    log.info("=" * 60)
    log.info(f"  {name.upper()}")
    log.info(f"  {cfg['description']}")
    log.info("=" * 60)

    cfg["dir"].mkdir(parents=True, exist_ok=True)
    method = cfg["method"]

    if method == "http":
        return download_http(cfg)
    elif method == "huggingface":
        return download_huggingface(cfg, hf_token=hf_token)
    elif method == "hf_datasets":
        return download_hf_datasets(cfg, hf_token=hf_token)
    elif method == "legi":
        return download_legi(cfg)
    elif method == "gallica":
        return download_gallica(cfg)
    else:
        log.error(f"Methode inconnue : {method}")
        return False

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN Phase 1 — Telechargement sources FR verifiees",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 01_download.py --list
  python 01_download.py --source wikipedia
  python 01_download.py --source cc100
  python 01_download.py --source pleia_books,pleia_news,hplt
  python 01_download.py --source mc4
  python 01_download.py --source all
  python 01_download.py --dry-run

Sources : wikipedia, wikisource, cc100, mc4, legifrance, gallica,
          pleia_books, pleia_news, hplt
        """,
    )
    p.add_argument("--source", default="all",
                   help="Source(s) separees par virgules, ou 'all'")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Token HuggingFace (optionnel — ces sources sont sans gate)")
    p.add_argument("--list", action="store_true",
                   help="Lister les sources disponibles et quitter")
    p.add_argument("--dry-run", action="store_true",
                   help="Simuler sans telecharger")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list:
        print()
        print("Sources RODIN — verifiees et accessibles :")
        print("-" * 65)
        total_gb = 0
        for name, cfg in SOURCES.items():
            gb = cfg["size_gb"]
            total_gb += gb
            method = cfg["method"]
            print(f"  {name:<15} {gb:>6.0f} Go   [{method:<12}]  {cfg['description'][:36]}")
        print("-" * 65)
        print(f"  {'TOTAL':<15} {total_gb:>6.0f} Go brut estime")
        print()
        return

    # Selectionner les sources
    if args.source == "all":
        selected = list(SOURCES.keys())
    else:
        selected = [s.strip() for s in args.source.split(",")]
        unknown  = [s for s in selected if s not in SOURCES]
        if unknown:
            log.error(f"Sources inconnues : {unknown}")
            log.error(f"Sources valides   : {list(SOURCES.keys())}")
            sys.exit(1)

    if args.dry_run:
        log.info("[DRY RUN] Sources qui seraient telechargees :")
        for name in selected:
            cfg = SOURCES[name]
            log.info(f"  {name:<15} {cfg['size_gb']:>6.0f} Go   {cfg['description']}")
        return

    log.info(f"Sources selectionnees : {selected}")
    results = {}
    t0 = time.time()

    for name in selected:
        cfg   = SOURCES[name]
        t_src = time.time()
        ok    = run_source(name, cfg, hf_token=args.hf_token)
        elapsed = time.time() - t_src
        results[name] = ok
        log.info(f"{'[OK]' if ok else '[ECHEC]'} {name} — {elapsed/60:.1f} min")

    # Bilan
    elapsed_total = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("BILAN")
    log.info("=" * 60)
    for name, ok in results.items():
        status = "OK   " if ok else "ECHEC"
        log.info(f"  [{status}]  {name:<15} {SOURCES[name]['description'][:40]}")
    ok_n = sum(results.values())
    log.info(f"\n  {ok_n}/{len(results)} sources OK — {elapsed_total/3600:.1f}h total")
    log.info("=" * 60)

    if ok_n < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
