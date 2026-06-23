#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RODIN - Phase 2 - 09_sample_for_bpe.py
=======================================================================
Echantillonnage stratifie des sources nettoyees (.zst) pour entrainer
le tokenizer BPE SentencePiece 64K.

DEUX MODES :
  --inspect   Inspection rapide BORNEE de chaque source (remplace le 08) :
              structure JSON, trigrammes (boilerplate), distribution de
              longueurs, ratio latin, echantillons visuels.
              Lecture plafonnee -> quelques minutes au total.

  (defaut)    Sampling stratifie : extrait UNIQUEMENT le champ texte,
              decoupe en lignes digestes, ecrit bpe_training_sample.txt
              + bpe_sample_stats.json. Reproductible (seed fixe).

PRINCIPE CLE : on ne lit JAMAIS les 1,4 To. Chaque source est lue en
sequentiel borne (cap decompresse) avec downsampling Bernoulli pour etaler
les keeps, et arret des que la cible volume est atteinte.

zstd n'est pas seekable ici (compresse en -3 -T16 simple) -> sampling de
tete borne. Pour un tokenizer le leger biais d'ordre intra-source est
negligeable ; la stratification ENTRE sources est ce qui compte.

Usage :
  python 09_sample_for_bpe.py --inspect
  python 09_sample_for_bpe.py --inspect --source hplt
  python 09_sample_for_bpe.py
  python 09_sample_for_bpe.py --source wikipedia
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

# orjson si dispo, sinon stdlib
try:
    import orjson
    def jloads(b):
        return orjson.loads(b)
except ImportError:
    def jloads(b):
        return json.loads(b)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CLEANED_DIR  = r"G:\data\rodin\cleaned"
BPE_DIR      = r"G:\data\rodin\bpe"
INSPECT_DIR  = r"G:\data\rodin\inspection"

SEED = 42

# Champs texte candidats (defensif : les sources peuvent differer)
TEXT_FIELDS = ("text", "content", "raw_content", "body", "page_content")

# Filtres au niveau document
MIN_DOC_CHARS   = 50        # docs plus courts = bruit
MAX_DOC_CHARS   = 200_000   # garde-fou memoire
LATIN_RATIO_MIN = 0.70      # plancher sanity FR (relache, on log le reste)

# Decoupage en lignes pour SentencePiece
MIN_LINE_CHARS  = 20
MAX_LINE_CHARS  = 4000      # marge sous la limite SP (accents = multibyte)

# Inspection : plafond de lecture decompresse par source
INSPECT_CAP_BYTES = 500 * 1024 * 1024   # 500 Mo -> rapide
INSPECT_TRIGRAMS  = 50
INSPECT_SAMPLES   = 5

# ----------------------------------------------------------------------
# TABLE DE STRATIFICATION
#   target_bytes : volume texte cible dans l'echantillon (octets)
#   read_cap     : plafond de lecture decompresse (borne dure)
#   "all"        : prendre toute la source (sources minuscules)
# read_cap > target_bytes -> on etale les keeps via Bernoulli p=target/cap
# ----------------------------------------------------------------------
GB = 1024 ** 3
SOURCES = {
    "legifrance":  {"file": r"legifrance\legifrance_cleaned.jsonl.zst",   "mode": "all"},
    "wikisource":  {"file": r"wikisource\wikisource_cleaned.jsonl.zst",   "mode": "all"},
    "wikipedia":   {"file": r"wikipedia\wikipedia_cleaned.jsonl.zst",     "target": 3 * GB, "read_cap": 8 * GB},
    "pleia_books": {"file": r"pleia_books\pleia_books_cleaned.jsonl.zst", "target": 5 * GB, "read_cap": 14 * GB},
    "pleia_news":  {"file": r"pleia_news\pleia_news_cleaned.jsonl.zst",   "target": 3 * GB, "read_cap": 10 * GB},
    "cc100":       {"file": r"cc100\cc100_cleaned.jsonl.zst",             "target": 2 * GB, "read_cap": 8 * GB},
    "hplt":        {"file": r"hplt\hplt_cleaned.jsonl.zst",               "target": 5 * GB, "read_cap": 14 * GB},
}

# Ordre de traitement : petites sources d'abord (feedback rapide), HPLT en dernier
SOURCE_ORDER = ["legifrance", "wikisource", "wikipedia", "cc100",
                "pleia_books", "pleia_news", "hplt"]

# ----------------------------------------------------------------------
# Filtres boilerplate (DEFAUT MINIMAL - a etoffer apres inspection)
# Appliques au niveau ligne. On reste prudent pour ne pas sur-filtrer
# avant d'avoir vu les trigrammes reels.
# ----------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*(tous droits r[eé]serv[eé]s|all rights reserved)\s*\.?\s*$", re.I),
    re.compile(r"^\s*creative commons", re.I),
    re.compile(r"^\s*(cookies?|politique de confidentialit[eé])\b", re.I),
    # --- boilerplate web confirme a l'inspection (hplt / crawl) ---
    # lignes COURTES seulement : on ne tue pas une vraie phrase qui
    # contiendrait ces mots, juste le bouton/footer isole.
    re.compile(r"^\s*lire la suite\s*\.{0,3}\s*$", re.I),
    re.compile(r"^\s*(voir|lire) plus\s*\.{0,3}\s*$", re.I),
    re.compile(r"^\s*partager (sur|via)\b.{0,40}$", re.I),
    re.compile(r"^\s*(cliquez ici|en savoir plus)\s*\.{0,3}\s*$", re.I),
]

# Jeu de caracteres "latins FR" pour le ratio
_LATIN_RE = re.compile(
    r"[A-Za-z0-9\u00C0-\u017F\s\.,;:!\?'\"()\[\]\-\u2013\u2014\u2026\u00AB\u00BB%\u20AC]"
)

# ======================================================================
# STREAMING ZSTD
# ======================================================================
def stream_zst_lines(path):
    """
    Decompresse en flux via 'zstd -dc --long=27' et yield les lignes
    decodees (bytes -> str utf-8, errors=replace).
    Yield (line_str, raw_byte_len).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=27", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024,
    )
    try:
        for raw in proc.stdout:
            yield raw.decode("utf-8", errors="replace"), len(raw)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def extract_text(line):
    """Parse une ligne JSONL et retourne le champ texte (ou None)."""
    line = line.strip()
    if not line:
        return None, None
    try:
        obj = jloads(line)
    except Exception:
        return None, "json_error"
    if not isinstance(obj, dict):
        return None, "not_dict"
    for f in TEXT_FIELDS:
        v = obj.get(f)
        if isinstance(v, str) and v:
            return v, f
    return None, "no_text_field"


def latin_ratio(s):
    if not s:
        return 0.0
    matched = len(_LATIN_RE.findall(s))
    return matched / len(s)


def split_to_lines(text):
    """
    Decoupe un document en lignes digestes pour SentencePiece :
    paragraphes -> phrases -> hard-wrap si trop long.
    Filtre les fragments trop courts.
    """
    text = text.replace("\r", "")
    out = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        # split phrase sur ponctuation finale suivie d'espace
        for frag in re.split(r"(?<=[\.!\?\u2026])\s+", para):
            frag = frag.strip()
            if len(frag) < MIN_LINE_CHARS:
                continue
            while len(frag) > MAX_LINE_CHARS:
                cut = frag.rfind(" ", 0, MAX_LINE_CHARS)
                if cut <= 0:
                    cut = MAX_LINE_CHARS
                out.append(frag[:cut].strip())
                frag = frag[cut:].strip()
            if len(frag) >= MIN_LINE_CHARS:
                out.append(frag)
    return out


def is_boilerplate(line):
    for pat in BOILERPLATE_PATTERNS:
        if pat.search(line):
            return True
    return False


def human(n):
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} Po"


# ======================================================================
# MODE INSPECTION
# ======================================================================
def inspect_source(name, cfg, cap_bytes):
    path = os.path.join(CLEANED_DIR, cfg["file"])
    print(f"\n=== INSPECT {name}  ({cfg['file']}) ===")
    if not os.path.isfile(path):
        print(f"  !! introuvable : {path}")
        return None

    rng = random.Random(SEED)
    fields_counter = Counter()
    parse_errors = 0
    empty_or_short = 0
    length_buckets = Counter()
    latin_low = 0
    trigrams = Counter()
    samples = []
    seen = 0
    read_bytes = 0
    docs = 0
    t0 = time.time()

    for line, blen in stream_zst_lines(path):
        read_bytes += blen
        if read_bytes >= cap_bytes:
            break
        seen += 1
        txt, field = extract_text(line)
        if txt is None:
            if field == "json_error":
                parse_errors += 1
            continue
        fields_counter[field] += 1
        docs += 1

        n = len(txt)
        if n < MIN_DOC_CHARS:
            empty_or_short += 1
        if n < 50:
            length_buckets["very_short(<50)"] += 1
        elif n < 200:
            length_buckets["short(50-200)"] += 1
        elif n < 1000:
            length_buckets["medium(200-1k)"] += 1
        elif n < 5000:
            length_buckets["long(1k-5k)"] += 1
        else:
            length_buckets["very_long(>5k)"] += 1

        if latin_ratio(txt[:2000]) < LATIN_RATIO_MIN:
            latin_low += 1

        # trigrammes de mots (sur un extrait pour rester rapide)
        words = re.findall(r"\w+", txt[:1500].lower())
        for i in range(len(words) - 2):
            trigrams[(words[i], words[i + 1], words[i + 2])] += 1

        # reservoir sampling borne
        if len(samples) < INSPECT_SAMPLES:
            samples.append(txt[:600])
        else:
            j = rng.randint(0, docs - 1)
            if j < INSPECT_SAMPLES:
                samples[j] = txt[:600]

    dt = time.time() - t0
    report = {
        "source": name,
        "read_bytes": read_bytes,
        "lines_seen": seen,
        "docs_valid": docs,
        "parse_errors": parse_errors,
        "empty_or_short": empty_or_short,
        "text_fields_used": dict(fields_counter),
        "length_buckets": dict(length_buckets),
        "latin_below_floor": latin_low,
        "top_trigrams": [
            {"ngram": " ".join(k), "count": c}
            for k, c in trigrams.most_common(INSPECT_TRIGRAMS)
        ],
        "elapsed_sec": round(dt, 1),
    }

    print(f"  lu={human(read_bytes)} lignes={seen} docs={docs} "
          f"parse_err={parse_errors} short={empty_or_short} "
          f"latin_low={latin_low} ({dt:.1f}s)")
    print(f"  champs texte : {dict(fields_counter)}")
    print(f"  longueurs    : {dict(length_buckets)}")
    print(f"  top trigrammes :")
    for item in report["top_trigrams"][:15]:
        print(f"     {item['count']:>7}  {item['ngram']}")

    return report, samples


def run_inspect(args):
    os.makedirs(INSPECT_DIR, exist_ok=True)
    cap = args.inspect_cap_mb * 1024 * 1024 if args.inspect_cap_mb else INSPECT_CAP_BYTES
    targets = [args.source] if args.source else SOURCE_ORDER
    full_report = {}
    samples_text = []

    for name in targets:
        if name not in SOURCES:
            print(f"  ?? source inconnue : {name}")
            continue
        res = inspect_source(name, SOURCES[name], cap)
        if res is None:
            continue
        report, samples = res
        full_report[name] = report
        samples_text.append(f"\n{'=' * 70}\nSOURCE : {name}\n{'=' * 70}")
        for i, s in enumerate(samples, 1):
            samples_text.append(f"\n--- echantillon {i} ---\n{s}")

    rpath = os.path.join(INSPECT_DIR, "inspection_report.json")
    spath = os.path.join(INSPECT_DIR, "samples_by_source.txt")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
    with open(spath, "w", encoding="utf-8") as f:
        f.write("\n".join(samples_text))

    print(f"\n[OK] rapport   -> {rpath}")
    print(f"[OK] samples   -> {spath}")


# ======================================================================
# MODE SAMPLING
# ======================================================================
def sample_source(name, cfg, out_fh, rng):
    path = os.path.join(CLEANED_DIR, cfg["file"])
    print(f"\n=== SAMPLE {name}  ({cfg['file']}) ===")
    if not os.path.isfile(path):
        print(f"  !! introuvable : {path}")
        return None

    mode_all = cfg.get("mode") == "all"
    target   = cfg.get("target", 0)
    read_cap = cfg.get("read_cap", 0)
    p_keep   = 1.0 if mode_all else min(1.0, target / read_cap)

    read_bytes = 0
    kept_bytes = 0
    docs_seen = 0
    docs_kept = 0
    lines_out = 0
    skipped_short = 0
    skipped_latin = 0
    skipped_boiler = 0
    t0 = time.time()
    last_log = t0

    for line, blen in stream_zst_lines(path):
        read_bytes += blen

        # arrets
        if not mode_all:
            if kept_bytes >= target:
                break
            if read_bytes >= read_cap:
                break

        txt, _field = extract_text(line)
        if txt is None:
            continue
        docs_seen += 1

        # downsampling Bernoulli (etale les keeps)
        if p_keep < 1.0 and rng.random() > p_keep:
            continue

        if len(txt) < MIN_DOC_CHARS:
            skipped_short += 1
            continue
        if len(txt) > MAX_DOC_CHARS:
            txt = txt[:MAX_DOC_CHARS]
        if latin_ratio(txt[:2000]) < LATIN_RATIO_MIN:
            skipped_latin += 1
            continue

        wrote_any = False
        for ln in split_to_lines(txt):
            if is_boilerplate(ln):
                skipped_boiler += 1
                continue
            out_fh.write(ln)
            out_fh.write("\n")
            kept_bytes += len(ln.encode("utf-8")) + 1
            lines_out += 1
            wrote_any = True
        if wrote_any:
            docs_kept += 1

        now = time.time()
        if now - last_log >= 15:
            rate = read_bytes / (now - t0) / (1024 * 1024)
            print(f"  ... lu={human(read_bytes)} garde={human(kept_bytes)} "
                  f"lignes={lines_out} ({rate:.0f} Mo/s)")
            last_log = now

    dt = time.time() - t0
    stats = {
        "source": name,
        "mode": "all" if mode_all else "stratified",
        "p_keep": round(p_keep, 4),
        "read_bytes": read_bytes,
        "kept_bytes": kept_bytes,
        "docs_seen": docs_seen,
        "docs_kept": docs_kept,
        "lines_out": lines_out,
        "skipped_short": skipped_short,
        "skipped_latin": skipped_latin,
        "skipped_boilerplate": skipped_boiler,
        "elapsed_sec": round(dt, 1),
    }
    print(f"  [{name}] lu={human(read_bytes)} garde={human(kept_bytes)} "
          f"docs={docs_kept} lignes={lines_out} "
          f"(short={skipped_short} latin={skipped_latin} boiler={skipped_boiler}) "
          f"{dt:.1f}s")
    return stats


def run_sample(args):
    os.makedirs(BPE_DIR, exist_ok=True)
    out_path = os.path.join(BPE_DIR, "bpe_training_sample.txt")
    stats_path = os.path.join(BPE_DIR, "bpe_sample_stats.json")
    targets = [args.source] if args.source else SOURCE_ORDER

    rng = random.Random(args.seed)
    all_stats = []
    t0 = time.time()

    # 'a' si on cible une seule source pour ne pas ecraser le reste,
    # 'w' pour un run complet propre.
    open_mode = "a" if args.source else "w"
    with open(out_path, open_mode, encoding="utf-8", newline="\n") as out_fh:
        for name in targets:
            if name not in SOURCES:
                print(f"  ?? source inconnue : {name}")
                continue
            st = sample_source(name, SOURCES[name], out_fh, rng)
            if st:
                all_stats.append(st)

    total_kept = sum(s["kept_bytes"] for s in all_stats)
    total_lines = sum(s["lines_out"] for s in all_stats)
    summary = {
        "seed": args.seed,
        "output": out_path,
        "total_kept_bytes": total_kept,
        "total_kept_human": human(total_kept),
        "total_lines": total_lines,
        "elapsed_sec": round(time.time() - t0, 1),
        "per_source": all_stats,
    }

    # merge stats si run partiel
    if args.source and os.path.isfile(stats_path):
        try:
            with open(stats_path, encoding="utf-8") as f:
                prev = json.load(f)
            prev_sources = {s["source"]: s for s in prev.get("per_source", [])}
            for s in all_stats:
                prev_sources[s["source"]] = s
            summary["per_source"] = list(prev_sources.values())
            summary["total_kept_bytes"] = sum(s["kept_bytes"] for s in summary["per_source"])
            summary["total_kept_human"] = human(summary["total_kept_bytes"])
            summary["total_lines"] = sum(s["lines_out"] for s in summary["per_source"])
        except Exception:
            pass

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TOTAL garde : {summary['total_kept_human']}  "
          f"({summary['total_lines']} lignes)")
    print(f"[OK] sample -> {out_path}")
    print(f"[OK] stats  -> {stats_path}")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="RODIN Phase 2 - sampling BPE")
    ap.add_argument("--inspect", action="store_true",
                    help="mode inspection rapide borne (remplace le 08)")
    ap.add_argument("--source", default=None,
                    help="traiter une seule source (ex: hplt)")
    ap.add_argument("--inspect-cap-mb", type=int, default=None,
                    help="plafond lecture decompresse par source en Mo (inspect)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.inspect:
        run_inspect(args)
    else:
        run_sample(args)


if __name__ == "__main__":
    main()
