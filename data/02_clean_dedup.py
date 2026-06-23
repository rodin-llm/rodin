#!/usr/bin/env python3
"""
RODIN Phase 1 — Cleaning + Déduplication MinHash
=================================================
Pipeline :
  raw/ → [parsing XML/JSONL/TXT/Parquet/XZ]
       → [cleaning texte]
       → cleaned/{source}/{source}_cleaned.jsonl
       → [MinHash global inter-sources]
       → deduped/merged/merged_deduped.jsonl
       → [scoring qualité heuristique]
       → deduped/merged/merged_final.jsonl

Corrections v2 :
  - Bug langdetect : logique inversée corrigée, seuil abaissé,
    détection sur texte post-nettoyage uniquement
  - Bug stats workers : agrégation corrigée (futures traités en fin de batch)
  - Bug tronquage max_chars : texte tronqué correctement propagé dans quality_filter
  - Parsers ajoutés : cc100 (xz), pleia/hplt (parquet HF), gallica (.txt)
  - Legifrance : lxml remplacé par xml.etree.ElementTree (stdlib, stable Windows)
  - Nettoyage wikitext amélioré : tableaux, balises math, infobox supprimés
  - Filtre qualité renforcé : détection spam, ratio ponctuation, densité mots FR
  - Sources list mise à jour : oscar/roots/croissant retirés

Usage :
    python 02_clean_dedup.py --source wikipedia
    python 02_clean_dedup.py --source all --workers 4
    python 02_clean_dedup.py --stage clean_only --source wikipedia,wikisource
    python 02_clean_dedup.py --stage dedup_only
    python 02_clean_dedup.py --stage quality_only
    python 02_clean_dedup.py --source wikipedia --force  # relance même si .done

Prérequis :
    pip install datasketch ftfy langdetect tqdm requests pyarrow datasets zstandard
"""

import io
import os
import re
import sys
import json
import gzip
import bz2
import lzma
import time
import hashlib
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Generator
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(r"D:\data\rodin")
RAW_DIR     = BASE_DIR / "raw"
CLEANED_DIR = BASE_DIR / "cleaned"
DEDUPED_DIR = BASE_DIR / "deduped"
MERGED_DIR  = DEDUPED_DIR / "merged"
LOG_DIR     = BASE_DIR / "logs"

for _d in [CLEANED_DIR, DEDUPED_DIR, MERGED_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Logging UTF-8 Windows-safe ───────────────────────────────────────────────

_stdout_utf8 = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(_stdout_utf8),
        logging.FileHandler(
            LOG_DIR / f"clean_{datetime.now():%Y%m%d_%H%M%S}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("rodin.clean")

# ─── Configuration qualité ────────────────────────────────────────────────────

@dataclass
class CleanConfig:
    # Longueur texte
    min_chars: int       = 300        # Minimum absolu post-nettoyage
    max_chars: int       = 500_000    # Tronquer (pas rejeter) les très longs docs

    # Mots
    min_words: int       = 50         # Minimum mots
    min_avg_word_len: float = 3.5     # Evite les textes tokenisés/codés
    max_avg_word_len: float = 12.0    # Evite les URLs/hashes sans espace

    # Ratios caractères
    max_special_char_ratio: float = 0.12  # Ponctuation + symboles hors alpha/digit/espace
    max_digit_ratio: float        = 0.15  # Textes trop numériques (tableaux, logs)
    max_uppercase_ratio: float    = 0.15  # Evite les textes ALL CAPS

    # Répétitions
    max_line_repetition_ratio: float      = 0.25  # Lignes dupliquées (spam)
    max_paragraph_repetition_ratio: float = 0.20

    # Lignes minimales
    min_doc_lines: int = 3

    # Langue — appliqué APRÈS nettoyage complet du texte
    require_french: bool        = True
    lang_sample_chars: int      = 3000   # Taille de l'échantillon pour langdetect
    lang_min_sample_chars: int  = 200    # Pas de détection si texte trop court
    lang_confidence_threshold: float = 0.75  # Seuil confiance langdetect

CFG = CleanConfig()

# ─── Overrides par source ─────────────────────────────────────────────────────
# Certaines sources ont un style spécifique incompatible avec les seuils généraux.
# Chaque override remplace uniquement les clés listées, le reste vient de CFG.

SOURCE_OVERRIDES: dict[str, dict] = {

    # Légifrance : textes juridiques.
    # - Articles de loi souvent très courts (50-200 chars) → min_chars abaissé
    # - 1-2 lignes par article dans LEGI → min_doc_lines = 1
    # - Nombreuses références numériques (n°, dates, alinéas) → digit_ratio souple
    # - Style formel peu de mots courants FR → seuil heuristique fr_ratio abaissé
    # - Ponctuation dense (points-virgules, tirets, parenthèses) → special souple
    "legifrance": {
        "min_chars":            80,
        "min_words":            10,
        "min_doc_lines":         1,
        "max_digit_ratio":      0.25,
        "max_special_char_ratio": 0.20,
        "max_uppercase_ratio":  0.30,   # Titres de section en majuscules (JORF)
        "fr_ratio_threshold":   0.03,   # Heuristique assouplie pour le style juridique
    },

    # Wikisource : textes littéraires domaine public.
    # - Poèmes, pièces de théâtre : peu de lignes, courts → assouplir
    # - Textes anciens : orthographe pre-1900, mots FR courants différents
    # - Listes de vers : répétition de structure normale
    "wikisource": {
        "min_chars":            150,
        "min_words":            20,
        "min_doc_lines":         1,
        "max_line_repetition_ratio": 0.40,   # Vers/strophes répétitifs
        "fr_ratio_threshold":   0.04,
    },

    # Gallica : OCR ancien parfois bruité
    # - Erreurs OCR augmentent special_char_ratio
    # - Textes XVIIe-XIXe : orthographe variable
    "gallica": {
        "min_chars":            200,
        "max_special_char_ratio": 0.18,
        "max_digit_ratio":      0.20,
        "fr_ratio_threshold":   0.04,
    },
    
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
        "fr_ratio_threshold":     0.04,
    },
}

def get_source_param(source: str, param: str):
    """
    Retourne la valeur d'un paramètre pour une source donnée.
    Priorité : SOURCE_OVERRIDES[source][param] > getattr(CFG, param).
    """
    override = SOURCE_OVERRIDES.get(source, {})
    if param in override:
        return override[param]
    return getattr(CFG, param, None)

# ─── Expressions régulières ───────────────────────────────────────────────────

# Wikitext
RE_WIKI_COMMENT   = re.compile(r'<!--.*?-->', re.DOTALL)
RE_WIKI_REFS      = re.compile(r'<ref[^>]*>.*?</ref>', re.DOTALL | re.IGNORECASE)
RE_WIKI_REFS_OPEN = re.compile(r'<ref[^/]*/>', re.IGNORECASE)
RE_WIKI_FILE      = re.compile(r'\[\[(?:File|Image|Fichier|Imagen|Media):[^\]]*\]\]', re.IGNORECASE)
RE_WIKI_CATEGORY  = re.compile(r'\[\[(?:Catégorie|Category|Kategoría|Kategorie):[^\]]*\]\]', re.IGNORECASE)
RE_WIKI_TEMPLATE  = re.compile(r'\{\{[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)?\}\}')  # 2 niveaux
RE_WIKI_TABLE     = re.compile(r'\{\|.*?\|\}', re.DOTALL)   # Tableaux wiki
RE_WIKI_MATH      = re.compile(r'<math[^>]*>.*?</math>', re.DOTALL | re.IGNORECASE)
RE_WIKI_GALLERY   = re.compile(r'<gallery[^>]*>.*?</gallery>', re.DOTALL | re.IGNORECASE)
RE_WIKI_LINK      = re.compile(r'\[\[(?:[^\]|]*?\|)?([^\]]*?)\]\]')   # Garde texte visible
RE_WIKI_EXT_LINK  = re.compile(r'\[https?://[^\s\]]*(?:\s([^\]]+))?\]')  # [url texte]
RE_WIKI_HEADING   = re.compile(r'={2,}([^=\n]+)={2,}')    # == Titre == → Titre

# Général
RE_HTML_TAG    = re.compile(r'<[^>]{1,200}>')
RE_URL         = re.compile(r'https?://\S+|www\.\S+')
RE_EMAIL       = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
RE_MULTI_NL    = re.compile(r'\n{3,}')
RE_MULTI_SP    = re.compile(r'[ \t]{2,}')
RE_DASH_LINE   = re.compile(r'^[-=*#]{5,}\s*$', re.MULTILINE)

# Boilerplate FR — lignes à supprimer
BOILERPLATE_RE = re.compile(
    r'tous droits réservés'
    r'|politique de confidentialité'
    r'|mentions légales'
    r'|cliquez ici pour'
    r'|abonnez-vous.{0,20}newsletter'
    r'|suivez-nous sur'
    r'|partagez? sur (facebook|twitter|linkedin)'
    r'|javascript (est )?désactivé'
    r'|veuillez activer javascript'
    r'|error 404|page not found'
    r'|cookie[s]? (nécessaire|obligatoire|analytique)'
    r'|accepter les cookies'
    r'|en savoir plus sur les cookies'
    r'|ce site utilise des cookies'
    r'|retour en haut'
    r'|imprimer cet article'
    r'|publié le \d',
    re.IGNORECASE,
)

# Mots français courants — pour vérifier présence FR sans langdetect
FR_COMMON_WORDS = frozenset([
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en",
    "est", "que", "qui", "dans", "sur", "par", "il", "elle", "ils",
    "elles", "nous", "vous", "son", "sa", "ses", "ce", "cet", "cette",
    "ces", "aussi", "mais", "ou", "donc", "or", "ni", "car", "pas",
    "plus", "très", "bien", "avec", "pour", "comme", "tout", "faire",
    "être", "avoir", "au", "aux", "dont", "où", "quand", "même",
])

# ─── Parsers ──────────────────────────────────────────────────────────────────

def parse_wikipedia_dump(xml_path: Path) -> Generator[dict, None, None]:
    """
    Parse dump Wikipedia/Wikisource .xml.bz2.
    Namespace détecté automatiquement.
    Filtre : namespace 0 uniquement, pas de redirects.
    Log toutes les 10 000 pages.
    """
    import xml.etree.ElementTree as ET

    log.info(f"Parsing dump XML bz2 : {xml_path.name}")

    # Détecter namespace
    ns_str = "http://www.mediawiki.org/xml/export-0.11/"
    try:
        with bz2.open(xml_path, "rb") as f:
            header = f.read(8192).decode("utf-8", errors="replace")
        m = re.search(r'xmlns="(http://www\.mediawiki\.org/xml/export-[^"]+)"', header)
        if m:
            ns_str = m.group(1)
    except Exception:
        pass
    log.info(f"  Namespace : {ns_str}")

    TAG_PAGE  = f"{{{ns_str}}}page"
    TAG_TITLE = f"{{{ns_str}}}title"
    TAG_ID    = f"{{{ns_str}}}id"
    TAG_TEXT  = f"{{{ns_str}}}text"
    TAG_NS    = f"{{{ns_str}}}ns"

    page_count  = 0
    yield_count = 0
    current: dict = {}
    in_page = False

    with bz2.open(xml_path, "rt", encoding="utf-8", errors="replace") as f:
        for event, elem in ET.iterparse(f, events=("start", "end")):
            tag = elem.tag
            if event == "start":
                if tag == TAG_PAGE:
                    in_page = True
                    current = {"ns": "0", "title": "", "id": "", "text": ""}
            elif event == "end":
                if not in_page:
                    elem.clear()
                    continue
                if tag == TAG_NS:
                    current["ns"] = elem.text or "0"
                elif tag == TAG_TITLE:
                    current["title"] = elem.text or ""
                elif tag == TAG_ID and not current["id"]:
                    current["id"] = elem.text or ""
                elif tag == TAG_TEXT:
                    current["text"] = elem.text or ""
                elif tag == TAG_PAGE:
                    in_page = False
                    page_count += 1
                    if page_count % 10_000 == 0:
                        log.info(
                            f"  {page_count:,} pages | {yield_count:,} articles gardés"
                        )
                    # Filtres parseur : namespace 0, pas redirect
                    if current["ns"] != "0":
                        elem.clear()
                        continue
                    text = current["text"].strip()
                    if not text or text.lower().startswith("#redirect"):
                        elem.clear()
                        continue
                    yield_count += 1
                    yield {
                        "text":   text,
                        "title":  current["title"],
                        "id":     f"wp_{current['id']}",
                        "source": "wikipedia",
                    }
                elem.clear()

    log.info(f"  Parsing terminé : {page_count:,} pages, {yield_count:,} articles")


def parse_wikisource_dump(xml_path: Path) -> Generator[dict, None, None]:
    """Wikisource — même format XML bz2 que Wikipedia."""
    for doc in parse_wikipedia_dump(xml_path):
        yield {**doc, "source": "wikisource", "id": doc["id"].replace("wp_", "ws_")}


def parse_cc100(cc100_dir: Path) -> Generator[dict, None, None]:
    """
    CC-100 FR — fichier fr.txt.xz.
    Format : texte brut, documents séparés par lignes vides.
    Parsing streaming via lzma pour éviter de tout charger en RAM.
    """
    xz_files = list(cc100_dir.glob("*.xz")) + list(cc100_dir.glob("*.txt"))
    if not xz_files:
        log.warning(f"CC-100 : aucun fichier .xz trouvé dans {cc100_dir}")
        return

    log.info(f"CC-100 : {len(xz_files)} fichier(s)")
    doc_count = 0

    for xz_file in xz_files:
        log.info(f"  Parsing : {xz_file.name}")
        buf: list[str] = []

        if xz_file.suffix == ".xz":
            fh = lzma.open(xz_file, "rt", encoding="utf-8", errors="replace")
        else:
            fh = open(xz_file, "r", encoding="utf-8", errors="replace")

        with fh as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if buf:
                        text = "\n".join(buf).strip()
                        if text:
                            doc_count += 1
                            yield {
                                "text":   text,
                                "id":     f"cc100_{doc_count}",
                                "source": "cc100",
                            }
                        buf = []
                else:
                    buf.append(line)
            # Dernier document sans ligne vide finale
            if buf:
                text = "\n".join(buf).strip()
                if text:
                    doc_count += 1
                    yield {
                        "text":   text,
                        "id":     f"cc100_{doc_count}",
                        "source": "cc100",
                    }

    log.info(f"  CC-100 : {doc_count:,} documents parsés")


def parse_parquet_hf(src_dir: Path, source_name: str,
                     text_col: str = "text") -> Generator[dict, None, None]:
    """
    Parser générique pour datasets HuggingFace au format parquet.
    Utilisé pour : pleia_books, pleia_news, hplt, mc4.
    Cherche récursivement tous les .parquet dans src_dir.
    text_col : nom de la colonne texte (auto-détecté si absent).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        log.error("pyarrow requis : pip install pyarrow")
        return

    parquet_files = sorted(src_dir.rglob("*.parquet"))
    if not parquet_files:
        log.warning(f"{source_name} : aucun fichier .parquet dans {src_dir}")
        return

    log.info(f"{source_name} : {len(parquet_files)} fichiers parquet")
    doc_count = 0

    for pf in parquet_files:
        try:
            schema = pq.read_schema(pf)
            col_names = schema.names

            # Auto-détecter la colonne texte
            candidates = ["text", "complete_text", "content", "passage", "body",
                          "article", "texte", "contenu", "document", "full_text",
                          "raw_text", "plain_text"]
            # Fallback : première colonne dont le nom contient "text"
            text_col_actual = next((c for c in candidates if c in col_names), None)
            if text_col_actual is None:
                text_col_actual = next((c for c in col_names if "text" in c.lower()), None)
            if text_col_actual is None:
                log.warning(f"  Colonne texte introuvable dans {pf.name} — colonnes : {col_names}")
                continue
            if text_col_actual != text_col:
                log.info(f"  {pf.name} : colonne '{text_col_actual}' utilisée (demandée: '{text_col}')")

            parquet_file = pq.ParquetFile(pf)
            for batch in parquet_file.iter_batches(batch_size=10_000,
                                                    columns=[text_col_actual]):
                for text in batch.column(text_col_actual).to_pylist():
                    if text and isinstance(text, str) and text.strip():
                        doc_count += 1
                        yield {
                            "text":   text,
                            "id":     f"{source_name}_{doc_count}",
                            "source": source_name,
                        }
            del parquet_file

            if doc_count % 100_000 == 0 and doc_count > 0:
                log.info(f"  {source_name} : {doc_count:,} docs")

        except Exception as e:
            log.warning(f"  Erreur {pf.name} : {e}")
            continue

    log.info(f"  {source_name} : {doc_count:,} documents parsés au total")


def parse_gallica(gallica_dir: Path) -> Generator[dict, None, None]:
    """
    Gallica BNF — fichiers .txt téléchargés via API SRU.
    Un document par fichier. Encodage variable → fallback latin-1.
    """
    txt_files = list(gallica_dir.glob("*.txt"))
    if not txt_files:
        log.warning(f"Gallica : aucun fichier .txt dans {gallica_dir}")
        return

    log.info(f"Gallica : {len(txt_files)} fichiers .txt")

    for tf in txt_files:
        try:
            try:
                text = tf.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = tf.read_text(encoding="latin-1", errors="replace")

            text = text.strip()
            if text:
                yield {
                    "text":   text,
                    "id":     f"gallica_{tf.stem}",
                    "source": "gallica",
                }
        except Exception as e:
            log.debug(f"  Gallica erreur {tf.name} : {e}")


def parse_legifrance(legi_dir: Path) -> Generator[dict, None, None]:
    """
    Légifrance DILA — archives tar.gz contenant des XML article par article.
    Utilise xml.etree.ElementTree (stdlib) — pas de dépendance lxml.
    Extrait les balises CONTENU, TEXTE, BLOC_TEXTUEL.
    """
    import tarfile
    import xml.etree.ElementTree as ET

    tar_files = list(legi_dir.glob("*.tar.gz")) + list(legi_dir.glob("*.tgz"))
    if not tar_files:
        log.warning(f"Légifrance : aucune archive tar.gz dans {legi_dir}")
        return

    log.info(f"Légifrance : {len(tar_files)} archives")
    doc_count = 0

    for tar_path in tar_files:
        log.info(f"  Archive : {tar_path.name}")
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.name.endswith(".xml"):
                        continue
                    try:
                        f = tar.extractfile(member)
                        if f is None:
                            continue
                        content = f.read()
                        # Essayer UTF-8 puis latin-1
                        try:
                            content_str = content.decode("utf-8")
                        except UnicodeDecodeError:
                            content_str = content.decode("latin-1", errors="replace")

                        root = ET.fromstring(content_str)
                        texts: list[str] = []
                        for tag in ["CONTENU", "TEXTE", "BLOC_TEXTUEL", "NOTA"]:
                            for el in root.iter(tag):
                                if el.text and el.text.strip():
                                    texts.append(el.text.strip())

                        full_text = "\n\n".join(texts).strip()
                        if len(full_text) >= 100:
                            doc_count += 1
                            yield {
                                "text":   full_text,
                                "id":     f"legi_{hashlib.md5(member.name.encode()).hexdigest()[:12]}",
                                "source": "legifrance",
                            }
                    except ET.ParseError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            log.warning(f"  Erreur archive {tar_path.name} : {e}")

    log.info(f"  Légifrance : {doc_count:,} articles parsés")


def get_parser(source_name: str):
    """Retourne la fonction de parsing pour une source donnée."""
    raw = RAW_DIR / source_name

    def _wiki():
        files = list(raw.glob("*.xml.bz2"))
        if not files:
            log.warning(f"wikipedia : aucun dump .xml.bz2 dans {raw}")
            return iter([])
        return parse_wikipedia_dump(files[0])

    def _wikisource():
        files = list(raw.glob("*.xml.bz2"))
        if not files:
            log.warning(f"wikisource : aucun dump .xml.bz2 dans {raw}")
            return iter([])
        return parse_wikisource_dump(files[0])

    parsers = {
        "wikipedia":   _wiki,
        "wikisource":  _wikisource,
        "cc100":       lambda: parse_cc100(raw),
        "pleia_books": lambda: parse_parquet_hf(raw, "pleia_books"),
        "pleia_news":  lambda: parse_parquet_hf(raw, "pleia_news"),
        "hplt":        lambda: parse_parquet_hf(raw, "hplt"),
        "gallica":     lambda: parse_gallica(raw),
        "legifrance":  lambda: parse_legifrance(raw),
    }
    fn = parsers.get(source_name)
    if fn is None:
        log.error(f"Pas de parser pour source : {source_name}")
        return lambda: iter([])
    return fn

# ─── Nettoyage wikitext ───────────────────────────────────────────────────────

def clean_wiki_markup(text: str) -> str:
    """
    Nettoyage wikitext complet.
    Ordre important : supprimer les structures imbriquées en premier.
    """
    # 1. Commentaires et refs
    text = RE_WIKI_COMMENT.sub("", text)
    text = RE_WIKI_REFS.sub("", text)
    text = RE_WIKI_REFS_OPEN.sub("", text)

    # 2. Blocs structurés (tableaux, gallery, math)
    text = RE_WIKI_TABLE.sub("", text)
    text = RE_WIKI_GALLERY.sub("", text)
    text = RE_WIKI_MATH.sub("", text)

    # 3. Fichiers et catégories
    text = RE_WIKI_FILE.sub("", text)
    text = RE_WIKI_CATEGORY.sub("", text)

    # 4. Templates — plusieurs passes pour les imbriqués
    for _ in range(4):
        new = RE_WIKI_TEMPLATE.sub("", text)
        if new == text:
            break
        text = new

    # 5. Liens internes [[lien|texte]] → texte, [[lien]] → lien
    text = RE_WIKI_LINK.sub(r"\1", text)

    # 6. Liens externes [url texte] → texte, [url] → ""
    text = RE_WIKI_EXT_LINK.sub(lambda m: m.group(1) or "", text)

    # 7. HTML résiduel
    text = RE_HTML_TAG.sub("", text)

    # 8. Titres de sections == Titre == → Titre\n
    text = RE_WIKI_HEADING.sub(r"\1\n", text)

    # 9. Lignes de séparation
    text = RE_DASH_LINE.sub("", text)

    return text

# ─── Nettoyage général ────────────────────────────────────────────────────────

def normalize_text(text: str, source: str = "") -> str:
    # Détecter double encodage latin-1/UTF-8 (symptôme : présence de "Ã")
    if "Ã" in text[:100]:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass

    # Guillemets FR → standard
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')   # « »
    text = text.replace("\u2019", "'").replace("\u2018", "'")   # ' '
    text = text.replace("\u201c", '"').replace("\u201d", '"')   # " "
    text = text.replace("\u2013", "-").replace("\u2014", "-")   # – —
    text = text.replace("\u00a0", " ")                          # espace insécable

    # Espaces et retours à la ligne
    text = RE_MULTI_SP.sub(" ", text)
    text = RE_MULTI_NL.sub("\n\n", text)

    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Supprime les lignes boilerplate détectées."""
    lines = text.split("\n")
    kept  = [l for l in lines if not BOILERPLATE_RE.search(l)]
    return "\n".join(kept)

# ─── Détection langue ─────────────────────────────────────────────────────────

def is_french(text: str, fr_ratio_threshold: float = 0.08) -> bool:
    """
    Détecte si le texte est français.
    Stratégie à 2 niveaux :
    1. Heuristique rapide : présence de mots FR courants (pas de librairie)
    2. langdetect sur échantillon si heuristique insuffisante
    Retourne True si français, False sinon.
    fr_ratio_threshold : seuil heuristique mots FR courants (défaut 0.08,
      abaissé pour sources au style formel : légifrance, wikisource, gallica).
    """
    if len(text) < CFG.lang_min_sample_chars:
        return True  # Trop court pour décider → on garde

    # Niveau 1 : heuristique mots FR courants
    words = re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text[:2000].lower())
    if len(words) >= 20:
        fr_words = sum(1 for w in words if w in FR_COMMON_WORDS)
        fr_ratio = fr_words / len(words)
        # Si > seuil de mots FR courants → clairement français
        if fr_ratio >= fr_ratio_threshold:
            return True
        # Si < 1% → clairement pas français
        if fr_ratio < 0.01:
            return False

    # Niveau 2 : langdetect
    try:
        from langdetect import detect_langs
        sample = text[:CFG.lang_sample_chars]
        results = detect_langs(sample)
        if not results:
            return True  # Indécis → garder
        best = results[0]
        # Accepter fr, mais aussi "ca" (catalan) et "pt" qui faussent souvent sur le FR
        if best.lang == "fr":
            return True
        # Si la confiance est faible → garder (éviter faux positifs)
        if best.prob < CFG.lang_confidence_threshold:
            return True
        # Langue clairement non-FR avec haute confiance → rejeter
        return False
    except Exception:
        return True  # En cas d'erreur → garder

# ─── Filtre qualité ───────────────────────────────────────────────────────────

def quality_filter(text: str, source: str) -> tuple[bool, str]:
    """
    Filtre qualité complet.
    Retourne (keep: bool, rejection_reason: str).
    Le texte passé ici est déjà nettoyé.
    Les seuils sont adaptés par source via SOURCE_OVERRIDES.
    """
    # Récupérer les seuils effectifs pour cette source
    min_chars              = get_source_param(source, "min_chars")
    min_words              = get_source_param(source, "min_words")
    min_doc_lines          = get_source_param(source, "min_doc_lines")
    max_digit_ratio        = get_source_param(source, "max_digit_ratio")
    max_special_char_ratio = get_source_param(source, "max_special_char_ratio")
    max_uppercase_ratio    = get_source_param(source, "max_uppercase_ratio")
    max_line_rep_ratio     = get_source_param(source, "max_line_repetition_ratio")
    fr_ratio_threshold     = get_source_param(source, "fr_ratio_threshold") or 0.08

    # Tronquer les très longs docs (pas rejeter)
    if len(text) > CFG.max_chars:
        text = text[:CFG.max_chars]

    n_chars = len(text)
    if n_chars < min_chars:
        return False, f"trop_court:{n_chars}"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < min_doc_lines:
        return False, f"trop_peu_lignes:{len(lines)}"

    words = text.split()
    if len(words) < min_words:
        return False, f"trop_peu_mots:{len(words)}"

    # Longueur moyenne des mots alpha
    alpha_words = [w for w in words if w.isalpha()]
    if len(alpha_words) >= 10:
        avg_wl = sum(len(w) for w in alpha_words) / len(alpha_words)
        if avg_wl < CFG.min_avg_word_len:
            return False, f"mots_courts:{avg_wl:.1f}"
        if avg_wl > CFG.max_avg_word_len:
            return False, f"mots_longs:{avg_wl:.1f}"

    # Ratio caractères alpha
    alpha_chars = sum(c.isalpha() for c in text)
    space_chars = sum(c.isspace() for c in text)
    digit_chars = sum(c.isdigit() for c in text)
    other_chars = n_chars - alpha_chars - space_chars - digit_chars

    alpha_ratio   = alpha_chars / n_chars
    digit_ratio   = digit_chars / n_chars
    special_ratio = other_chars / n_chars

    if alpha_ratio < (1 - max_special_char_ratio - max_digit_ratio - 0.15):
        return False, f"trop_special:{special_ratio:.2f}"

    if digit_ratio > max_digit_ratio:
        return False, f"trop_chiffres:{digit_ratio:.2f}"

    if special_ratio > max_special_char_ratio:
        return False, f"trop_ponctuation:{special_ratio:.2f}"

    # Majuscules excessives
    upper_chars = sum(c.isupper() for c in text)
    if alpha_chars > 0:
        upper_ratio = upper_chars / alpha_chars
        if upper_ratio > max_uppercase_ratio:
            return False, f"trop_majuscules:{upper_ratio:.2f}"

    # Lignes répétées (spam / boilerplate)
    if len(lines) >= 10:
        unique_ratio = len(set(lines)) / len(lines)
        rep_ratio = 1 - unique_ratio
        if rep_ratio > max_line_rep_ratio:
            return False, f"lignes_repetees:{rep_ratio:.2f}"

    # Détecter les tables des matières / listes d'index
    list_lines = sum(1 for l in lines if l.startswith(('*', '#', '-')) )
    if len(lines) >= 10 and list_lines / len(lines) > 0.60:
        return False, "liste_index"

    # Détection langue — en dernier (coûteux)
    if CFG.require_french:
        if not is_french(text, fr_ratio_threshold=fr_ratio_threshold):
            return False, "langue_non_fr"

    return True, ""

# ─── Nettoyage document ───────────────────────────────────────────────────────

def clean_document(doc: dict) -> dict | None:
    """
    Nettoie un document complet.
    Retourne None si rejeté par les filtres qualité.
    """
    text   = doc.get("text", "")
    source = doc.get("source", "unknown")

    if not text or not text.strip():
        return None

    # 1. Nettoyage spécifique à la source
    if source in ("wikipedia", "wikisource"):
        text = clean_wiki_markup(text)

    # 2. Nettoyage général
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_HTML_TAG.sub("", text)
    text = remove_boilerplate(text)
    text = normalize_text(text)

    # 3. Filtre qualité (sur texte nettoyé)
    keep, reason = quality_filter(text, source)
    if not keep:
        return None

    # Tronquer après validation (max_chars peut avoir été appliqué dans quality_filter)
    if len(text) > CFG.max_chars:
        text = text[:CFG.max_chars]

    return {
        **doc,
        "text":  text,
        "chars": len(text),
    }

# ─── Worker ───────────────────────────────────────────────────────────────────

def _clean_worker(args: tuple) -> tuple[list[dict], dict]:
    """Worker threadé : nettoie un batch de documents."""
    batch, source_name = args
    cleaned = []
    stats   = defaultdict(int)
    for doc in batch:
        stats["total"] += 1
        result = clean_document(doc)
        if result is not None:
            cleaned.append(result)
            stats["kept"] += 1
        else:
            stats["rejected"] += 1
    return cleaned, dict(stats)

# ─── Pipeline cleaning ────────────────────────────────────────────────────────

def run_cleaning(source_name: str, workers: int = 4, batch_size: int = 200,
                 force: bool = False):
    """
    Nettoie une source complète.
    Output : cleaned/{source_name}/{source_name}_cleaned.jsonl
    """
    out_dir   = CLEANED_DIR / source_name
    out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = out_dir / ".done"

    if done_flag.exists() and not force:
        size_mb = (out_dir / f"{source_name}_cleaned.jsonl").stat().st_size / 1024**2 \
            if (out_dir / f"{source_name}_cleaned.jsonl").exists() else 0
        log.info(f"Cleaning {source_name} déjà effectué ({size_mb:.0f} Mo) — skip")
        return

    log.info(f"━━━ Cleaning : {source_name} ━━━")
    parser   = get_parser(source_name)
    out_file = out_dir / f"{source_name}_cleaned.jsonl"

    stats_total = defaultdict(int)
    batch: list = []
    doc_count   = 0
    start       = time.time()

    with open(out_file, "w", encoding="utf-8") as fout:
        futures = []
        executor = ThreadPoolExecutor(max_workers=workers)

        def flush_futures(all_futures: bool = False):
            nonlocal doc_count
            target = futures[:] if all_futures else [f for f in futures if f.done()]
            for future in target:
                futures.remove(future)
                try:
                    cleaned_batch, batch_stats = future.result()
                    for d in cleaned_batch:
                        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                        doc_count += 1
                    for k, v in batch_stats.items():
                        stats_total[k] += v
                except Exception as e:
                    log.error(f"Erreur worker : {e}")

        for doc in parser():
            batch.append(doc)
            if len(batch) >= batch_size:
                futures.append(executor.submit(_clean_worker, (batch, source_name)))
                batch = []
                flush_futures(all_futures=False)

                total_seen = stats_total["total"]
                if total_seen > 0 and total_seen % 50_000 == 0:
                    kept_pct = stats_total["kept"] / total_seen * 100
                    elapsed  = time.time() - start
                    log.info(
                        f"  {source_name} : {total_seen:,} lus | "
                        f"{stats_total['kept']:,} gardés ({kept_pct:.1f}%) | "
                        f"{elapsed/60:.1f} min"
                    )

        # Dernier batch
        if batch:
            futures.append(executor.submit(_clean_worker, (batch, source_name)))

        # Attendre tous les workers
        flush_futures(all_futures=True)
        executor.shutdown(wait=True)

    elapsed    = time.time() - start
    total      = max(stats_total["total"], 1)
    kept       = stats_total["kept"]
    keep_ratio = kept / total * 100

    log.info(f"✓ {source_name} : {elapsed/60:.1f} min")
    log.info(f"  Lus : {total:,} | Gardés : {kept:,} ({keep_ratio:.1f}%) | "
             f"Rejetés : {stats_total['rejected']:,}")
    log.info(f"  Output : {out_file}")

    # Stats JSON
    stats_path = out_dir / "cleaning_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "source":          source_name,
            "total_read":      int(stats_total["total"]),
            "kept":            int(kept),
            "rejected":        int(stats_total["rejected"]),
            "keep_ratio_pct":  round(keep_ratio, 2),
            "elapsed_seconds": round(elapsed, 1),
            "output_file":     str(out_file),
            "output_size_mb":  round(out_file.stat().st_size / 1024**2, 1)
                               if out_file.exists() else 0,
        }, f, indent=2, ensure_ascii=False)

    if kept == 0:
        log.error(
            f"ERREUR : 0 document gardé pour {source_name} !\n"
            f"  Vérifier les paramètres CleanConfig et le contenu du dump."
        )
        # Ne pas créer le flag .done si 0 docs → permet relance
        return

    done_flag.touch()

# ─── MinHash Déduplication ────────────────────────────────────────────────────

class MinHashDeduplicator:
    """
    Déduplication MinHash LSH inter-sources.
    Shingles 5-grammes caractères, 128 permutations, seuil Jaccard 0.80.
    """
    def __init__(self, num_perm: int = 128, threshold: float = 0.80,
                 shingle_size: int = 5):
        try:
            from datasketch import MinHash, MinHashLSH
            self.MinHash = MinHash
            self.lsh     = MinHashLSH(threshold=threshold, num_perm=num_perm)
        except ImportError:
            log.error("datasketch requis : pip install datasketch")
            raise
        self.num_perm     = num_perm
        self.threshold    = threshold
        self.shingle_size = shingle_size
        self.seen_ids:    set = set()
        self._dup_count   = 0
        self._proc_count  = 0

    def _shinglify(self, text: str) -> set:
        t = re.sub(r'\s+', ' ', text.lower())[:5000]
        n = self.shingle_size
        return {t[i:i+n] for i in range(len(t) - n + 1)}

    def is_duplicate(self, doc_id: str, text: str) -> bool:
        if doc_id in self.seen_ids:
            self._dup_count += 1
            return True
        mh = self.MinHash(num_perm=self.num_perm)
        for sh in self._shinglify(text):
            mh.update(sh.encode("utf-8"))
        try:
            if self.lsh.query(mh):
                self._dup_count += 1
                return True
        except Exception:
            pass
        try:
            self.lsh.insert(doc_id, mh)
            self.seen_ids.add(doc_id)
        except Exception as e:
            log.debug(f"LSH insert error {doc_id}: {e}")
        self._proc_count += 1
        return False

    def stats(self) -> dict:
        total = self._proc_count + self._dup_count
        return {
            "processed":  self._proc_count,
            "duplicates": self._dup_count,
            "dup_ratio":  self._dup_count / max(total, 1),
        }


def run_global_dedup(threshold: float = 0.80, num_perm: int = 128,
                     force: bool = False):
    """
    Déduplication MinHash globale sur tous les fichiers cleaned/.
    Fusionne toutes les sources, élimine les doublons inter-sources.
    Output : deduped/merged/merged_deduped.jsonl
    """
    out_file   = MERGED_DIR / "merged_deduped.jsonl"
    stats_file = MERGED_DIR / "dedup_stats.json"
    done_flag  = MERGED_DIR / ".done"

    if done_flag.exists() and not force:
        log.info("Déduplication déjà effectuée — skip")
        return

    all_cleaned = sorted(CLEANED_DIR.rglob("*_cleaned.jsonl"))
    if not all_cleaned:
        log.error(f"Aucun fichier cleaned trouvé dans {CLEANED_DIR}")
        log.error("Lancer d'abord : python 02_clean_dedup.py --stage clean_only")
        return

    log.info(f"━━━ Déduplication MinHash globale ━━━")
    log.info(f"  Threshold : {threshold} | Permutations : {num_perm}")
    log.info(f"  Fichiers  : {len(all_cleaned)}")
    for f in all_cleaned:
        log.info(f"    {f.parent.name}/{f.name} ({f.stat().st_size/1024**2:.0f} Mo)")

    dedup        = MinHashDeduplicator(num_perm=num_perm, threshold=threshold)
    total_read   = 0
    total_written= 0
    start        = time.time()

    with open(out_file, "w", encoding="utf-8") as fout:
        for cleaned_file in all_cleaned:
            src  = cleaned_file.parent.name
            r, w = 0, 0
            log.info(f"  Dédup {src}...")

            with open(cleaned_file, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text   = doc.get("text", "")
                    doc_id = doc.get("id", f"{src}_{total_read}")
                    if not text:
                        continue
                    r += 1
                    total_read += 1
                    if not dedup.is_duplicate(doc_id, text):
                        fout.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        w += 1
                        total_written += 1
                    if total_read % 100_000 == 0:
                        s = dedup.stats()
                        log.info(
                            f"  {total_read:,} lus | {total_written:,} gardés "
                            f"| dup: {s['dup_ratio']:.1%} | "
                            f"{(time.time()-start)/60:.0f} min"
                        )
            log.info(f"    {src}: {r:,} → {w:,} ({w/max(r,1):.1%})")

    elapsed     = time.time() - start
    final_stats = dedup.stats()
    log.info(f"✓ Déduplication : {elapsed/60:.0f} min")
    log.info(f"  Lu : {total_read:,} | Gardé : {total_written:,} "
             f"({total_written/max(total_read,1):.1%})")
    log.info(f"  Doublons : {final_stats['duplicates']:,} ({final_stats['dup_ratio']:.1%})")
    log.info(f"  Output   : {out_file}")

    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_read":       total_read,
            "total_written":    total_written,
            "duplicates":       final_stats["duplicates"],
            "dup_ratio":        round(final_stats["dup_ratio"], 4),
            "threshold":        threshold,
            "num_perm":         num_perm,
            "elapsed_seconds":  round(elapsed, 1),
        }, f, indent=2)

    done_flag.touch()

# ─── Scoring qualité ──────────────────────────────────────────────────────────

def run_quality_scoring(input_file: Path | None = None,
                        output_file: Path | None = None,
                        percentile_keep: float = 0.85,
                        force: bool = False):
    """
    Filtrage qualité par score heuristique composite.
    Score basé sur : entropie Shannon + richesse vocabulaire + densité FR.
    Conserve le top percentile_keep% des documents.
    Output : deduped/merged/merged_final.jsonl
    """
    import math

    if input_file is None:
        input_file  = MERGED_DIR / "merged_deduped.jsonl"
    if output_file is None:
        output_file = MERGED_DIR / "merged_final.jsonl"

    done_flag = MERGED_DIR / ".quality_done"
    if done_flag.exists() and not force:
        log.info("Scoring qualité déjà effectué — skip")
        return

    if not input_file.exists():
        log.error(f"Fichier input introuvable : {input_file}")
        return

    log.info(f"━━━ Scoring qualité (top {percentile_keep:.0%}) ━━━")

    def entropy_score(text: str) -> float:
        """Entropie Shannon sur les caractères (0-8, idéal ~4.0-4.8 pour FR)."""
        freq: dict[str, int] = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        n = len(text)
        if n == 0:
            return 0.0
        return -sum((v/n) * math.log2(v/n) for v in freq.values())

    def vocab_richness(text: str) -> float:
        """Type-Token Ratio (mots uniques / mots totaux)."""
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def fr_density(text: str) -> float:
        """Ratio de mots FR courants dans le texte."""
        words = re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text[:3000].lower())
        if len(words) < 5:
            return 0.5
        return sum(1 for w in words if w in FR_COMMON_WORDS) / len(words)

    def quality_score(text: str) -> float:
        """Score composite [0, 1]."""
        ent  = entropy_score(text)
        voc  = vocab_richness(text)
        frd  = fr_density(text)
        # Entropie normalisée : 3.5-5.0 = optimal pour le français
        ent_norm = max(0.0, min(1.0, (ent - 2.5) / 2.5))
        # Score : 50% entropie + 30% vocab + 20% densité FR
        return 0.50 * ent_norm + 0.30 * voc + 0.20 * frd

    # Phase 1 : calculer tous les scores (lecture complète)
    log.info("Phase 1 : calcul des scores...")
    scores: list[float] = []
    count = 0

    with open(input_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc  = json.loads(line)
                sc   = quality_score(doc.get("text", ""))
                scores.append(sc)
            except Exception:
                scores.append(0.0)
            count += 1
            if count % 200_000 == 0:
                log.info(f"  Scoré : {count:,} docs")

    if not scores:
        log.error("Aucun document à scorer.")
        return

    # Percentile seuil
    sorted_scores = sorted(scores)
    cutoff_idx    = int(len(sorted_scores) * (1 - percentile_keep))
    threshold_sc  = sorted_scores[min(cutoff_idx, len(sorted_scores) - 1)]
    log.info(f"  Seuil : {threshold_sc:.3f} (top {percentile_keep:.0%} = {int(len(scores)*percentile_keep):,} docs)")

    # Phase 2 : filtrer
    log.info("Phase 2 : filtrage...")
    kept_mask = [s >= threshold_sc for s in scores]
    written   = 0

    with open(input_file, encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            if idx < len(kept_mask) and kept_mask[idx]:
                fout.write(line)
                written += 1

    log.info(f"✓ Qualité : {written:,}/{count:,} docs conservés ({written/max(count,1):.1%})")
    log.info(f"  Output : {output_file}")
    log.info(f"  Taille : {output_file.stat().st_size/1024**3:.1f} Go")
    done_flag.touch()

# ─── CLI ──────────────────────────────────────────────────────────────────────

SOURCES_AVAILABLE = [
    "wikipedia", "wikisource", "cc100",
    "pleia_books", "pleia_news", "hplt",
    "gallica", "legifrance",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN Phase 1 — Cleaning + Déduplication MinHash v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Sources disponibles : {', '.join(SOURCES_AVAILABLE)}

Exemples :
  # Cleaning wikipedia uniquement
  python 02_clean_dedup.py --stage clean_only --source wikipedia

  # Cleaning toutes les sources téléchargées
  python 02_clean_dedup.py --stage clean_only --source all --workers 4

  # Déduplication globale (après cleaning)
  python 02_clean_dedup.py --stage dedup_only

  # Scoring qualité (après dédup)
  python 02_clean_dedup.py --stage quality_only

  # Pipeline complet
  python 02_clean_dedup.py --stage all --source all

  # Relancer wikipedia en ignorant le flag .done
  python 02_clean_dedup.py --stage clean_only --source wikipedia --force
        """,
    )
    p.add_argument("--source", default="all",
                   help="Source(s) séparées par virgules, ou 'all'")
    p.add_argument("--stage",
                   choices=["all", "clean_only", "dedup_only", "quality_only"],
                   default="all")
    p.add_argument("--workers", type=int, default=4,
                   help="Workers threads pour le cleaning (défaut: 4)")
    p.add_argument("--dedup-threshold", type=float, default=0.80,
                   help="Seuil similarité Jaccard MinHash (défaut: 0.80)")
    p.add_argument("--num-perm", type=int, default=128,
                   help="Permutations MinHash (défaut: 128)")
    p.add_argument("--quality-percentile", type=float, default=0.85,
                   help="Percentile qualité à conserver (défaut: 0.85)")
    p.add_argument("--force", action="store_true",
                   help="Ignorer les flags .done et réexécuter")
    return p.parse_args()


def main():
    args = parse_args()

    if args.source == "all":
        selected = SOURCES_AVAILABLE
    else:
        selected = [s.strip() for s in args.source.split(",")]
        unknown  = [s for s in selected if s not in SOURCES_AVAILABLE]
        if unknown:
            log.error(f"Sources inconnues : {unknown}")
            log.error(f"Sources valides   : {SOURCES_AVAILABLE}")
            sys.exit(1)

    log.info(f"Stage : {args.stage} | Sources : {selected} | Workers : {args.workers}")

    start_total = time.time()

    # ── Cleaning
    if args.stage in ("all", "clean_only"):
        for source in selected:
            src_raw = RAW_DIR / source
            if not src_raw.exists():
                log.warning(f"Source {source} non trouvée dans {src_raw} — skip")
                continue
            # Vérifier qu'il y a des fichiers
            has_files = any(src_raw.iterdir())
            if not has_files:
                log.warning(f"Source {source} : dossier vide — skip")
                continue
            run_cleaning(source, workers=args.workers, force=args.force)

    # ── Déduplication
    if args.stage in ("all", "dedup_only"):
        run_global_dedup(
            threshold=args.dedup_threshold,
            num_perm=args.num_perm,
            force=args.force,
        )

    # ── Scoring qualité
    if args.stage in ("all", "quality_only"):
        run_quality_scoring(
            percentile_keep=args.quality_percentile,
            force=args.force,
        )

    elapsed = time.time() - start_total
    log.info(f"\n✓ Pipeline terminé en {elapsed/3600:.1f}h")

    # Résumé fichier final
    final_file = MERGED_DIR / "merged_final.jsonl"
    if final_file.exists():
        size_gb = final_file.stat().st_size / 1024**3
        with open(final_file, encoding="utf-8") as f:
            n_docs = sum(1 for _ in f)
        log.info(f"\nFichier final : {final_file}")
        log.info(f"  Taille : {size_gb:.1f} Go")
        log.info(f"  Docs   : {n_docs:,}")
        log.info(f"\n→ Prêt pour Phase 2 (Tokenizer BPE custom FR 64K)")


if __name__ == "__main__":
    main()
