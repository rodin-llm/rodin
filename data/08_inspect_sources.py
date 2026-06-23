#!/usr/bin/env python3
"""
RODIN - Phase 2, etape 1 : inspection des sources zstd cleaned.

Objectif : valider la qualite et la structure des donnees avant de generer
l'echantillon stratifie pour l'entrainement du tokenizer BPE.

Pour chaque source zstd :
  - Liste les champs JSON presents (text, meta, score, etc.)
  - Compte les lignes corrompues, vides, trop courtes
  - Top N trigrammes (detection boilerplate residuel)
  - Distribution des longueurs (court/moyen/long)
  - Ratio caracteres latins / autres (sanity FR)
  - Echantillon visuel : 5 docs aleatoires

Strategie : streaming via zstd -dc, reservoir sampling, limite configurable
en octets decompresses par source (defaut 2 Go, suffisant statistiquement).

Usage:
    python 08_inspect_sources.py
    python 08_inspect_sources.py --max-bytes 5000000000  # 5 Go par source
    python 08_inspect_sources.py --source hplt           # 1 source seulement
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

try:
    import orjson  # plus rapide que json stdlib
    JSON_LIB = "orjson"
except ImportError:
    orjson = None
    JSON_LIB = "json"

# ============================================================================
# CONFIGURATION
# ============================================================================

RODIN_ROOT = Path("G:/data/rodin")
CLEANED_DIR = RODIN_ROOT / "cleaned"
OUTPUT_DIR = RODIN_ROOT / "inspection"

SOURCES = [
    "wikipedia",
    "wikisource",
    "legifrance",
    "pleia_books",
    "pleia_news",
    "cc100",
    "hplt",
]

# Limite par defaut : 2 Go decompresses par source
DEFAULT_MAX_BYTES = 2_000_000_000

# Reservoir sampling
N_SAMPLES_FOR_TRIGRAMS = 50_000   # docs gardes pour analyse trigrammes
N_VISUAL_SAMPLES = 5              # docs montres pour inspection humaine

# Buckets de longueur (en chars)
LENGTH_BUCKETS = [
    ("very_short", 0, 100),
    ("short", 100, 500),
    ("medium", 500, 2000),
    ("long", 2000, 10000),
    ("very_long", 10000, float("inf")),
]

# Regex pour trigrammes de mots (3 mots consecutifs en minuscules)
WORD_RE = re.compile(r"\b[a-z\u00e0-\u00ff]+\b")

# Regex pour ratio caracteres latins
LATIN_RE = re.compile(r"[a-zA-Z\u00c0-\u017f\s\d.,;:!?'\"()\-]")

# Seuils de validation
MIN_TEXT_LEN = 50  # chars, en-dessous = trop court
LATIN_RATIO_THRESHOLD = 0.85  # < 85% latin = suspect

# Couleurs console
class C:
    R = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"
    GREY = "\033[90m"

    @staticmethod
    def disable():
        for attr in ("R", "BOLD", "RED", "GREEN", "YELLOW", "BLUE", "CYAN", "GREY"):
            setattr(C, attr, "")


# ============================================================================
# STREAMING DECOMPRESSION ZSTD
# ============================================================================

def stream_zstd_lines(zst_path: Path, max_bytes: int | None = None) -> Iterator[tuple[int, bytes]]:
    """
    Streame les lignes d'un .zst via subprocess zstd -dc.
    Yield (line_number_1based, raw_bytes_without_newline).
    Stop apres max_bytes octets decompresses lus.
    """
    if not zst_path.exists():
        raise FileNotFoundError(f"Source manquante : {zst_path}")

    # zstd doit etre dans le PATH
    proc = subprocess.Popen(
        ["zstd", "-dc", "--long=27", str(zst_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=1024 * 1024,  # 1 Mo buffer
    )
    assert proc.stdout is not None

    bytes_read = 0
    line_no = 0
    try:
        for raw in proc.stdout:
            line_no += 1
            bytes_read += len(raw)
            yield line_no, raw.rstrip(b"\r\n")
            if max_bytes is not None and bytes_read >= max_bytes:
                break
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# ============================================================================
# PARSING JSON
# ============================================================================

def parse_json(raw: bytes) -> dict | None:
    """Parse une ligne JSON. Retourne None si invalide."""
    if not raw:
        return None
    try:
        if orjson is not None:
            return orjson.loads(raw)
        else:
            return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


# ============================================================================
# RESERVOIR SAMPLING
# ============================================================================

class Reservoir:
    """Reservoir sampling de Vitter (algorithme R) - RAM constante."""

    def __init__(self, k: int, seed: int = 42):
        self.k = k
        self.items: list[Any] = []
        self.n_seen = 0
        self.rng = random.Random(seed)

    def add(self, item: Any) -> None:
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
        else:
            j = self.rng.randint(0, self.n_seen - 1)
            if j < self.k:
                self.items[j] = item


# ============================================================================
# ANALYSE DES TRIGRAMMES
# ============================================================================

def extract_word_trigrams(text: str, max_per_doc: int = 100) -> list[str]:
    """Extrait les trigrammes de mots (3 mots consecutifs)."""
    words = WORD_RE.findall(text.lower())
    if len(words) < 3:
        return []
    trigrams = []
    for i in range(min(len(words) - 2, max_per_doc)):
        trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    return trigrams


# ============================================================================
# RATIO LATIN
# ============================================================================

def latin_ratio(text: str, sample_size: int = 2000) -> float:
    """Ratio de caracteres latins/standards. Echantillon les premiers N chars."""
    if not text:
        return 0.0
    sample = text[:sample_size]
    if not sample:
        return 0.0
    matches = LATIN_RE.findall(sample)
    return sum(len(m) for m in matches) / len(sample)


# ============================================================================
# ANALYSE D'UNE SOURCE
# ============================================================================

def analyze_source(source: str, zst_path: Path, max_bytes: int) -> dict:
    """Analyse complete d'une source zstd."""
    print(f"\n{C.BOLD}{C.CYAN}=== {source} ==={C.R}")
    print(f"  Fichier : {zst_path}")
    print(f"  Taille zst : {zst_path.stat().st_size / 1e9:.2f} Go")
    print(f"  Limite lecture : {max_bytes / 1e9:.2f} Go decompresses")

    stats = {
        "source": source,
        "path": str(zst_path),
        "size_zst_bytes": zst_path.stat().st_size,
        "max_bytes_read": max_bytes,
        "lines_read": 0,
        "json_valid": 0,
        "json_invalid": 0,
        "missing_text": 0,
        "empty_text": 0,
        "too_short_text": 0,
        "low_latin_ratio": 0,
        "valid_docs": 0,
        "fields_seen": Counter(),
        "field_types": defaultdict(Counter),
        "length_buckets": Counter(),
        "total_text_chars": 0,
        "min_text_len": float("inf"),
        "max_text_len": 0,
    }

    trigram_counter: Counter[str] = Counter()
    visual_reservoir = Reservoir(N_VISUAL_SAMPLES, seed=42)
    trigram_reservoir_n = 0
    trigram_reservoir_max = N_SAMPLES_FOR_TRIGRAMS

    t0 = time.time()
    last_report = t0

    for line_no, raw in stream_zstd_lines(zst_path, max_bytes=max_bytes):
        stats["lines_read"] += 1

        # Parsing JSON
        obj = parse_json(raw)
        if obj is None:
            stats["json_invalid"] += 1
            continue
        if not isinstance(obj, dict):
            stats["json_invalid"] += 1
            continue
        stats["json_valid"] += 1

        # Champs vus
        for k, v in obj.items():
            stats["fields_seen"][k] += 1
            stats["field_types"][k][type(v).__name__] += 1

        # Champ text
        text = obj.get("text")
        if text is None:
            stats["missing_text"] += 1
            continue
        if not isinstance(text, str):
            stats["missing_text"] += 1
            continue
        if len(text) == 0:
            stats["empty_text"] += 1
            continue
        if len(text) < MIN_TEXT_LEN:
            stats["too_short_text"] += 1
            continue

        # Ratio latin
        lr = latin_ratio(text)
        if lr < LATIN_RATIO_THRESHOLD:
            stats["low_latin_ratio"] += 1

        # Stats longueur
        tl = len(text)
        stats["total_text_chars"] += tl
        stats["min_text_len"] = min(stats["min_text_len"], tl)
        stats["max_text_len"] = max(stats["max_text_len"], tl)

        for name, lo, hi in LENGTH_BUCKETS:
            if lo <= tl < hi:
                stats["length_buckets"][name] += 1
                break

        stats["valid_docs"] += 1

        # Trigrammes (echantillon)
        if trigram_reservoir_n < trigram_reservoir_max:
            trigrams = extract_word_trigrams(text, max_per_doc=50)
            trigram_counter.update(trigrams)
            trigram_reservoir_n += 1

        # Echantillons visuels
        visual_reservoir.add({
            "line_no": line_no,
            "len": tl,
            "preview": text[:500],
            "fields": list(obj.keys()),
        })

        # Progress
        now = time.time()
        if now - last_report >= 5.0:
            elapsed = now - t0
            rate = stats["lines_read"] / elapsed if elapsed > 0 else 0
            print(f"  ... {stats['lines_read']:>10,} lignes lues | "
                  f"{stats['valid_docs']:>10,} valides | "
                  f"{rate:>7.0f} l/s | "
                  f"{elapsed:.0f}s",
                  flush=True)
            last_report = now

    elapsed = time.time() - t0

    # Finaliser stats
    if stats["min_text_len"] == float("inf"):
        stats["min_text_len"] = 0

    stats["fields_seen"] = dict(stats["fields_seen"])
    stats["field_types"] = {k: dict(v) for k, v in stats["field_types"].items()}
    stats["length_buckets"] = dict(stats["length_buckets"])
    stats["avg_text_len"] = (
        stats["total_text_chars"] / stats["valid_docs"]
        if stats["valid_docs"] > 0 else 0
    )
    stats["elapsed_sec"] = round(elapsed, 1)

    # Top trigrammes
    stats["top_trigrams"] = trigram_counter.most_common(50)
    stats["trigrams_doc_sample"] = trigram_reservoir_n

    # Echantillons visuels
    stats["visual_samples"] = visual_reservoir.items

    # Affichage console
    print_source_summary(stats)

    return stats


# ============================================================================
# AFFICHAGE
# ============================================================================

def print_source_summary(s: dict) -> None:
    print(f"\n  {C.BOLD}Resultats {s['source']}:{C.R}")
    print(f"    Lignes lues       : {s['lines_read']:>12,}")
    print(f"    JSON valides      : {s['json_valid']:>12,}  "
          f"({100*s['json_valid']/max(s['lines_read'],1):.2f}%)")
    if s["json_invalid"] > 0:
        print(f"    {C.RED}JSON invalides    : {s['json_invalid']:>12,}{C.R}")
    if s["missing_text"] > 0:
        print(f"    {C.YELLOW}text manquant     : {s['missing_text']:>12,}{C.R}")
    if s["empty_text"] > 0:
        print(f"    {C.YELLOW}text vide         : {s['empty_text']:>12,}{C.R}")
    if s["too_short_text"] > 0:
        print(f"    {C.YELLOW}text < {MIN_TEXT_LEN} chars   : {s['too_short_text']:>12,}{C.R}")
    if s["low_latin_ratio"] > 0:
        pct = 100 * s["low_latin_ratio"] / max(s["valid_docs"], 1)
        col = C.RED if pct > 5 else C.YELLOW
        print(f"    {col}low latin (<85%) : {s['low_latin_ratio']:>12,}  ({pct:.2f}%){C.R}")
    print(f"    {C.GREEN}Docs VALIDES     : {s['valid_docs']:>12,}{C.R}")

    print(f"\n    Champs JSON vus :")
    for field, count in sorted(s["fields_seen"].items(), key=lambda x: -x[1]):
        types = s["field_types"].get(field, {})
        types_str = ", ".join(f"{t}:{n}" for t, n in types.items())
        print(f"      - {field:<20} {count:>10,}  ({types_str})")

    print(f"\n    Longueur text (chars) :")
    print(f"      min / avg / max  : {s['min_text_len']:,} / "
          f"{s['avg_text_len']:,.0f} / {s['max_text_len']:,}")
    print(f"      Buckets :")
    for name, lo, hi in LENGTH_BUCKETS:
        n = s["length_buckets"].get(name, 0)
        pct = 100 * n / max(s["valid_docs"], 1)
        bar = "#" * int(pct / 2)
        hi_str = "inf" if hi == float("inf") else f"{int(hi):,}"
        print(f"      {name:<12} [{lo:>6,}-{hi_str:>7}]  {n:>10,}  {pct:>5.1f}%  {bar}")

    print(f"\n    Top 15 trigrammes (sample {s['trigrams_doc_sample']:,} docs) :")
    for tg, count in s["top_trigrams"][:15]:
        print(f"      {count:>8,}  {tg}")

    print(f"\n    Duree : {s['elapsed_sec']}s")


# ============================================================================
# ECRITURE OUTPUTS
# ============================================================================

def write_json_report(all_stats: list[dict], output_path: Path) -> None:
    """Ecrit le rapport JSON complet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convertir Counter et types non-serializables
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(x) for x in obj]
        if isinstance(obj, tuple):
            return [clean(x) for x in obj]
        if isinstance(obj, (Counter, defaultdict)):
            return clean(dict(obj))
        return obj

    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": clean(all_stats),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{C.GREEN}[OK] Rapport JSON ecrit : {output_path}{C.R}")


def write_visual_samples(all_stats: list[dict], output_path: Path) -> None:
    """Ecrit les echantillons visuels en texte lisible."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 78 + "\n")
        f.write("RODIN - Echantillons visuels par source (5 docs aleatoires chacune)\n")
        f.write("=" * 78 + "\n\n")
        for s in all_stats:
            f.write(f"\n{'#' * 78}\n")
            f.write(f"# SOURCE : {s['source']}\n")
            f.write(f"{'#' * 78}\n\n")
            for i, sample in enumerate(s.get("visual_samples", []), 1):
                f.write(f"--- Echantillon {i} (ligne {sample['line_no']}, "
                        f"len={sample['len']}, fields={sample['fields']}) ---\n")
                f.write(sample["preview"])
                f.write("\n\n")
    print(f"{C.GREEN}[OK] Echantillons visuels : {output_path}{C.R}")


# ============================================================================
# DETECTION zstd
# ============================================================================

def check_zstd_available() -> None:
    try:
        result = subprocess.run(["zstd", "--version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError("zstd retourne un code d'erreur")
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"{C.RED}ERREUR: zstd n'est pas accessible dans le PATH.{C.R}")
        print("Installer zstd ou ajouter au PATH avant de relancer.")
        sys.exit(1)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inspection sources zstd RODIN")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help=f"Octets decompresses max par source (defaut: {DEFAULT_MAX_BYTES:,})")
    parser.add_argument("--source", type=str, default=None,
                        help="Inspecter une seule source (ex: hplt)")
    parser.add_argument("--no-color", action="store_true",
                        help="Desactiver les couleurs ANSI")
    args = parser.parse_args()

    if args.no_color or os.environ.get("NO_COLOR"):
        C.disable()

    print(f"{C.BOLD}{C.CYAN}")
    print("=" * 78)
    print("  RODIN - Inspection sources zstd (Phase 2 etape 1)")
    print("=" * 78)
    print(f"{C.R}")
    print(f"  JSON parser : {JSON_LIB}")
    print(f"  Limite par source : {args.max_bytes / 1e9:.2f} Go decompresses")
    print(f"  Output : {OUTPUT_DIR}")

    check_zstd_available()

    sources_to_run = [args.source] if args.source else SOURCES
    all_stats = []

    for source in sources_to_run:
        zst_path = CLEANED_DIR / source / f"{source}_cleaned.jsonl.zst"
        if not zst_path.exists():
            print(f"{C.RED}[SKIP] {source} : {zst_path} introuvable{C.R}")
            continue

        try:
            stats = analyze_source(source, zst_path, args.max_bytes)
            all_stats.append(stats)
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Interruption utilisateur, sauvegarde partielle...{C.R}")
            break
        except Exception as e:
            print(f"{C.RED}[ERREUR] {source} : {e}{C.R}")
            import traceback
            traceback.print_exc()
            continue

    # Outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(all_stats, OUTPUT_DIR / "inspection_report.json")
    write_visual_samples(all_stats, OUTPUT_DIR / "samples_by_source.txt")

    # Resume final
    print(f"\n{C.BOLD}{C.CYAN}")
    print("=" * 78)
    print("  RESUME GLOBAL")
    print("=" * 78)
    print(f"{C.R}")
    print(f"  {'Source':<14} {'Lignes':>12} {'Valides':>12} {'Champs':>8} {'AvgLen':>8}")
    print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*8} {'-'*8}")
    for s in all_stats:
        print(f"  {s['source']:<14} "
              f"{s['lines_read']:>12,} "
              f"{s['valid_docs']:>12,} "
              f"{len(s['fields_seen']):>8} "
              f"{s['avg_text_len']:>8,.0f}")
    print()
    print(f"{C.GREEN}Inspection terminee.{C.R}")
    print(f"  -> Rapport JSON   : {OUTPUT_DIR / 'inspection_report.json'}")
    print(f"  -> Echantillons   : {OUTPUT_DIR / 'samples_by_source.txt'}")
    print()
    print(f"{C.BOLD}Prochaine etape :{C.R}")
    print(f"  1. Examiner samples_by_source.txt visuellement")
    print(f"  2. Verifier les top trigrammes pour boilerplate suspect")
    print(f"  3. Si OK, lancer 09_sample_for_bpe.py")


if __name__ == "__main__":
    main()
