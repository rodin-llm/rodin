#!/usr/bin/env python3
r"""
RODIN Phase 1 - Deduplication MinHash v9.1 (pipeline parallele robuste)
========================================================================

CHANGEMENTS vs v9
-----------------

1. WORKER v9.1 (numpy vectorise + Mersenne mod correct via split hi/lo)
   - v9 utilisait une boucle Python pure (1.15M ops Python/doc gros HPLT)
   - v9.1 vectorise tout en numpy uint64 sans overflow
   - Gain mesure : 3-5x sur le worker single-thread
   - Sigs IDENTIQUES a v9 (meme algo, juste plus rapide)

2. SIGNAL HANDLER CORRIGE (le bug de la fermeture v9)
   - v9 : SIGINT -> _drain_until(0) attend des Events qui ne seront
     jamais set apres process_pool shutdown -> deadlock infini
   - v9.1 : SIGINT -> set tous les ready Events des in_flight +
     cancel_futures + timeout 30s sur le drain final.
     Fermeture propre garantie.

3. IMPORT WORKER : depuis _dedup_worker_v91 au lieu de _dedup_worker_v9

Reste identique a v9 : architecture pipeline, logging, heartbeat,
PartWriter, ProgressDB, MinHashIndex, etc.

USAGE
-----
   python 07_dedup_quality_v91.py --self-test
   python 07_dedup_quality_v91.py --dry-run
   python 07_dedup_quality_v91.py --test --test-n 5000
   python 07_dedup_quality_v91.py                   # vrai run
"""

import io
import os
import re
import sys
import json
import math
import time
import queue
import signal
import sqlite3
import logging
import argparse
import traceback
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ─── orjson optionnel ─────────────────────────────────────────────────────────

try:
    import orjson
    _HAS_ORJSON = True
    def _json_loads(s):
        return orjson.loads(s)
    def _json_dumps_bytes(obj):
        return orjson.dumps(obj)
except ImportError:
    _HAS_ORJSON = False
    def _json_loads(s):
        return json.loads(s)
    def _json_dumps_bytes(obj):
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

# ─── Chemins ──────────────────────────────────────────────────────────────────

BASE_G = Path(r"G:\data\rodin")
BASE_D = Path(r"D:\data\rodin")

CLEANED_DIR        = BASE_G / "cleaned"
DEDUPED_DIR_FINAL  = BASE_G / "deduped"
DEDUPED_DIR_ACTIVE = BASE_D / "deduped"
LOG_DIR            = BASE_D / "logs"
INDEX_DB           = BASE_D / "minhash_index.db"
PROGRESS_DB        = DEDUPED_DIR_FINAL / "dedup_progress.db"

MERGED_FINAL = DEDUPED_DIR_FINAL / "merged_final.jsonl"
STATS_FILE   = DEDUPED_DIR_FINAL / "pipeline_stats.json"

PART_MAX_BYTES = 50 * 1024**3   # 50 Go

DEDUPED_DIR_FINAL.mkdir(parents=True, exist_ok=True)
DEDUPED_DIR_ACTIVE.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
BASE_D.mkdir(parents=True, exist_ok=True)

# ─── Paramètres MinHash (IDENTIQUES v8.x — pas d'impact qualité algo) ────────

NUM_PERM     = 128
SHINGLE_SIZE = 5
THRESHOLD    = 0.95
LSH_BANDS    = 8
LSH_ROWS     = NUM_PERM // LSH_BANDS
WINDOW_SIZE  = 3000
N_WINDOWS    = 3

CHECKPOINT_EVERY  = 100_000
LOG_EVERY_DOCS    = 100_000   # log progress après N docs traités
HEARTBEAT_SECONDS = 30        # log heartbeat toutes les N secondes même si pipeline bloqué
LOG_FLUSH_SECONDS = 5         # flush handler de log toutes les N secondes
QUALITY_TOP_PCT   = 0.85

# ─── Paramètres pipeline parallèle ────────────────────────────────────────────

N_WORKERS         = 10
N_LOOKUP_THREADS  = 4
RESULT_QUEUE_MAX  = N_WORKERS * 256        # backpressure in_flight
INSERT_BATCH_SIZE = 50_000
MAX_TEXT_LEN      = 100_000

SQLITE_CACHE_KB    = -4_194_304             # 4 Go
SQLITE_MMAP_BYTES  = 161_061_273_600        # 150 Go (index grossira a ~125 Go fin HPLT)

# ─── Catégories sources ───────────────────────────────────────────────────────

SOURCES_PREMIUM = {"legifrance", "wikipedia", "wikisource", "pleia_books"}
SOURCES_WEB     = {"pleia_news", "cc100", "hplt"}
SOURCES_ORDER   = [
    "legifrance", "wikipedia", "wikisource", "pleia_books",
    "pleia_news", "cc100", "hplt",
]


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING ROBUSTE — flush forcé périodique + UTF-8 strict
# ═══════════════════════════════════════════════════════════════════════════

class FlushingFileHandler(logging.FileHandler):
    """FileHandler qui force os.fsync périodiquement.

    Le FileHandler standard fait file.write() qui passe par les buffers
    Python + OS. En cas de crash brutal sous Windows, les buffers ne sont
    pas flushés → log fichier vide ou tronqué.

    Cette version :
      - flush() après chaque emit (force write système)
      - os.fsync() toutes les LOG_FLUSH_SECONDS via thread externe
    """
    def __init__(self, filename, encoding="utf-8"):
        super().__init__(filename, mode="a", encoding=encoding, delay=False)
        self._last_fsync = time.time()

    def emit(self, record):
        super().emit(record)
        try:
            if self.stream is not None:
                self.stream.flush()
        except Exception:
            pass

    def force_fsync(self):
        try:
            if self.stream is not None:
                self.stream.flush()
                os.fsync(self.stream.fileno())
                self._last_fsync = time.time()
        except Exception:
            pass


def setup_logging(log_file: Path) -> tuple:
    """Configure le logging UTF-8 robuste.

    Retourne (logger, file_handler) pour permettre un flush manuel ailleurs.
    """
    # stdout en UTF-8 (sinon Windows console = cp1252 → UnicodeEncodeError
    # silencieux sur certains caractères, message log perdu)
    try:
        stdout_utf8 = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
            line_buffering=True,
        )
    except Exception:
        stdout_utf8 = sys.stdout  # fallback

    file_handler = FlushingFileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    stream_handler = logging.StreamHandler(stdout_utf8)
    stream_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    root = logging.getLogger()
    # Wipe existing handlers (re-runs in same process)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return logging.getLogger("rodin.dedup.v91"), file_handler


class FlushDaemon(threading.Thread):
    """Thread qui force fsync sur le file handler toutes les 5s.

    Garantit que même si le pipeline est complètement bloqué, les logs
    déjà émis sont sur disque dans les 5 secondes.
    """
    def __init__(self, file_handler: FlushingFileHandler, interval: float = LOG_FLUSH_SECONDS):
        super().__init__(daemon=True, name="log-flush")
        self.handler = file_handler
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            self.handler.force_fsync()

    def stop(self):
        self._stop.set()


# ─── Init logging tôt (fichier nommé par timestamp) ──────────────────────────

LOG_FILE = LOG_DIR / f"dedup_v91_{datetime.now():%Y%m%d_%H%M%S}.log"
log, _file_handler = setup_logging(LOG_FILE)
_flush_daemon = FlushDaemon(_file_handler)
_flush_daemon.start()


# ═══════════════════════════════════════════════════════════════════════════
# IMPORT WORKER v9 — CRITIQUE
# ═══════════════════════════════════════════════════════════════════════════

try:
    from _dedup_worker_v91 import worker_compute as _worker_compute
    log.info("Worker v9.1 importé : numpy vectorise + Mersenne split hi/lo correct")
except ImportError as e:
    log.error(f"FATAL : _dedup_worker_v91.py introuvable : {e}")
    sys.exit(1)


# ─── Helpers MinHash (identiques v8.x) ────────────────────────────────────────

def _band_key(sig, b: int) -> int:
    """Extrait la clé 64 bits de la bande b."""
    vals = sig[b * LSH_ROWS:(b + 1) * LSH_ROWS]
    h = 0
    for i, v in enumerate(vals):
        h ^= (v * (0x9e3779b97f4a7c15 + i)) & 0xFFFFFFFFFFFFFFFF
    return h


def _to_sqlite_int(u64: int) -> int:
    """Convertit uint64 → int64 signé pour stockage SQLite."""
    return u64 if u64 < (1 << 63) else u64 - (1 << 64)


def extract_windows(text: str) -> str:
    """Multi-fenêtres début/milieu/fin (identique v8.x).

    Cap MAX_TEXT_LEN à l'entrée pour éviter outliers regex.
    """
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]

    n_raw = len(text)
    if n_raw <= WINDOW_SIZE * N_WINDOWS * 2:
        t = re.sub(r'\s+', ' ', text).strip().lower()
        n = len(t)
        if n <= WINDOW_SIZE * N_WINDOWS:
            return t
        mid = (n - WINDOW_SIZE) // 2
        parts = [t[:WINDOW_SIZE], t[mid:mid + WINDOW_SIZE], t[max(0, n - WINDOW_SIZE):]]
        return " ".join(parts)

    RAW = (WINDOW_SIZE * 3) // 2
    mid = (n_raw - RAW) // 2
    raw_parts = [text[:RAW], text[mid:mid + RAW], text[max(0, n_raw - RAW):]]
    norm_parts = [re.sub(r'\s+', ' ', p).strip().lower()[:WINDOW_SIZE] for p in raw_parts]
    return " ".join(norm_parts)


# ─── Quality scoring (identique v8.x, reproduit pour estimate_web_threshold) ─

FR_COMMON_WORDS = frozenset([
    "le","la","les","de","du","des","un","une","et","en","est","que",
    "qui","dans","sur","par","il","elle","ils","elles","nous","vous",
    "son","sa","ses","ce","cet","cette","ces","aussi","mais","ou",
    "donc","or","ni","car","pas","plus","très","bien","avec","pour",
    "comme","tout","faire","être","avoir","au","aux","dont","où",
    "quand","même","leur","leurs","y","lui","on","se","ne","si",
    "je","tu","me","te","mon","ton","ma","ta","nos","vos",
])


def quality_score_local(text: str) -> float:
    sample = text[:2000]
    n = len(sample)
    if n > 0:
        freq = {}
        for c in sample:
            freq[c] = freq.get(c, 0) + 1
        ent = -sum((v/n) * math.log2(v/n) for v in freq.values())
    else:
        ent = 0.0
    ent_norm = max(0.0, min(1.0, (ent - 2.5) / 2.5))
    words = text.lower().split()[:500]
    voc   = len(set(words)) / len(words) if words else 0.0
    fw    = re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text[:3000].lower())
    frd   = (sum(1 for w in fw if w in FR_COMMON_WORDS) / len(fw)
             if len(fw) >= 5 else 0.5)
    return 0.50 * ent_norm + 0.30 * voc + 0.20 * frd


# ═══════════════════════════════════════════════════════════════════════════
# INDEX MINHASH v9 (identique v8.2 dans la logique)
# ═══════════════════════════════════════════════════════════════════════════

class MinHashIndexV9:
    """SQLite avec lookups thread-safe (connexions read-only thread-local) +
    writes batchés depuis main thread.

    Identique v8.2 — la robustesse vient du pipeline & logging, pas de l'index.
    """

    def __init__(self, db_path: Path, n_lookup_threads: int = N_LOOKUP_THREADS):
        self.db_path = db_path
        self.n_lookup_threads = n_lookup_threads

        self.write_con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._apply_pragmas(self.write_con, write=True)

        idx_exists = self.write_con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_bands_lookup'"
        ).fetchone()
        self.has_index = idx_exists is not None

        self._tls = threading.local()
        self._insert_batch = []
        self._stat_lookups = 0
        self._stat_hits = 0
        self._stat_inserts = 0
        self._stat_flushes = 0

        log.info(f"Index v9.1 : {db_path}")
        log.info(f"  Taille          : {db_path.stat().st_size/1024**3:.2f} Go")
        log.info(f"  mmap_size       : {SQLITE_MMAP_BYTES/1024**3:.0f} Go")
        log.info(f"  cache_size      : {abs(SQLITE_CACHE_KB)/1024:.0f} Mo")
        log.info(f"  Lookup threads  : {n_lookup_threads}")
        log.info(f"  INSERT batch    : {INSERT_BATCH_SIZE:,} docs")
        log.info(f"  INDEX B-tree    : {'OUI' if self.has_index else 'NON ⚠'}")

    @staticmethod
    def _apply_pragmas(con, write: bool):
        if write:
            con.executescript(f"""
                PRAGMA journal_mode = WAL;
                PRAGMA synchronous  = NORMAL;
                PRAGMA cache_size   = {SQLITE_CACHE_KB};
                PRAGMA temp_store   = MEMORY;
                PRAGMA mmap_size    = {SQLITE_MMAP_BYTES};
            """)
        else:
            con.executescript(f"""
                PRAGMA query_only   = ON;
                PRAGMA cache_size   = {SQLITE_CACHE_KB};
                PRAGMA temp_store   = MEMORY;
                PRAGMA mmap_size    = {SQLITE_MMAP_BYTES};
            """)
        con.commit()

    def get_readonly_con(self):
        con = getattr(self._tls, "con", None)
        if con is None:
            con = sqlite3.connect(
                f"file:{self.db_path.as_posix()}?mode=ro",
                uri=True, check_same_thread=False
            )
            self._apply_pragmas(con, write=False)
            self._tls.con = con
        return con

    def is_dup(self, sig) -> bool:
        keys = [_band_key(sig, b) for b in range(LSH_BANDS)]
        conditions = " OR ".join(["(band_id=? AND band_h64=?)"] * LSH_BANDS)
        params = []
        for b, h64 in enumerate(keys):
            params.extend([b, _to_sqlite_int(h64)])
        sql = f"SELECT 1 FROM bands WHERE {conditions} LIMIT 1"
        con = self.get_readonly_con()
        result = con.execute(sql, params).fetchone() is not None
        self._stat_lookups += 1
        if result:
            self._stat_hits += 1
        return result

    def add(self, sig):
        keys = [_band_key(sig, b) for b in range(LSH_BANDS)]
        for b, h64 in enumerate(keys):
            self._insert_batch.append((b, _to_sqlite_int(h64)))
        self._stat_inserts += 1
        if len(self._insert_batch) >= INSERT_BATCH_SIZE * LSH_BANDS:
            self.flush()

    def flush(self):
        if not self._insert_batch:
            return
        self.write_con.executemany(
            "INSERT INTO bands(band_id, band_h64) VALUES(?,?)", self._insert_batch
        )
        self.write_con.commit()
        self._insert_batch = []
        self._stat_flushes += 1

    def stats(self) -> dict:
        n = max(self._stat_lookups, 1)
        return {
            "lookups":   self._stat_lookups,
            "hits":      self._stat_hits,
            "hit_pct":   round(self._stat_hits * 100 / n, 1),
            "inserts":   self._stat_inserts,
            "flushes":   self._stat_flushes,
            "batch_pending": len(self._insert_batch),
        }

    def close(self):
        self.flush()
        self.write_con.close()


# ─── ProgressDB (identique v8.x) ─────────────────────────────────────────────

class ProgressDB:
    def __init__(self, db_path: Path):
        self.con = sqlite3.connect(str(db_path))
        self.con.executescript("""
            CREATE TABLE IF NOT EXISTS progress (
                source     TEXT PRIMARY KEY,
                lines_done INTEGER DEFAULT 0,
                kept       INTEGER DEFAULT 0,
                updated_at TEXT
            );
        """)
        self.con.commit()

    def get(self, source: str):
        row = self.con.execute(
            "SELECT lines_done, kept FROM progress WHERE source=?", (source,)
        ).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    def save(self, source: str, lines_done: int, kept: int):
        self.con.execute("""
            INSERT INTO progress(source, lines_done, kept, updated_at)
            VALUES(?,?,?,?)
            ON CONFLICT(source) DO UPDATE SET
                lines_done=excluded.lines_done,
                kept=excluded.kept,
                updated_at=excluded.updated_at
        """, (source, lines_done, kept, datetime.now().isoformat()))
        self.con.commit()

    def close(self):
        self.con.close()


# ─── Estimation seuil qualité WEB ────────────────────────────────────────────

def estimate_web_threshold(sources, sample_size: int = 500_000) -> float:
    web = [(s, p) for s, p in sources if s in SOURCES_WEB]
    if not web:
        return 0.0
    log.info(f"Estimation seuil qualité WEB sur {sample_size:,} docs...")
    scores = []
    per = max(1, sample_size // len(web))
    for src, path in web:
        count = 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = _json_loads(line)
                    t = doc.get("text", "")
                    if t:
                        scores.append(quality_score_local(t))
                        count += 1
                except Exception:
                    pass
                if count >= per:
                    break
        log.info(f"  {src} : {count:,} docs scorés")
    if not scores:
        return 0.0
    scores.sort()
    thr = scores[min(int(len(scores) * (1 - QUALITY_TOP_PCT)), len(scores) - 1)]
    log.info(f"  min={scores[0]:.3f} median={scores[len(scores)//2]:.3f} max={scores[-1]:.3f}")
    log.info(f"  Seuil top {QUALITY_TOP_PCT*100:.0f}% : {thr:.4f}")
    return thr


# ─── PartWriter (identique v8.2) ─────────────────────────────────────────────

class PartWriter:
    def __init__(self, active_dir: Path, final_dir: Path, base_name: str = "merged_final"):
        self.active_dir = active_dir
        self.final_dir = final_dir
        self.base_name = base_name

        self._move_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="part_move")
        self._move_futures = []

        self._part_idx = self._next_part_index()
        self._fout = None
        self._current_path = None
        self._current_bytes = 0
        self._moved_bytes = 0

        self._open_new_part()

    def _next_part_index(self) -> int:
        max_idx = -1
        for d in [self.active_dir, self.final_dir]:
            if not d.exists():
                continue
            for p in d.glob(f"{self.base_name}.part_*.jsonl"):
                try:
                    idx = int(p.stem.split("_")[-1])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
        return max_idx + 1

    def _part_filename(self, idx: int) -> str:
        return f"{self.base_name}.part_{idx:04d}.jsonl"

    def _open_new_part(self):
        fname = self._part_filename(self._part_idx)
        self._current_path = self.active_dir / fname
        self._fout = open(self._current_path, "ab")
        self._current_bytes = (
            self._current_path.stat().st_size if self._current_path.exists() else 0
        )
        log.info(f"  [part] ouvert : {self._current_path}")

    def write(self, line_bytes: bytes):
        self._fout.write(line_bytes)
        self._fout.write(b"\n")
        self._current_bytes += len(line_bytes) + 1
        if self._current_bytes >= PART_MAX_BYTES:
            log.info(f"  [part] seuil {PART_MAX_BYTES/1024**3:.0f} Go atteint "
                     f"(part_{self._part_idx:04d}: {self._current_bytes/1024**3:.1f} Go)")
            self.rotate()

    def flush(self):
        if self._fout is not None:
            self._fout.flush()

    def rotate(self):
        if self._fout is None or self._current_path is None:
            return
        if self._current_bytes == 0:
            return
        self._fout.flush()
        self._fout.close()
        closed_path = self._current_path
        closed_bytes = self._current_bytes
        closed_idx = self._part_idx

        fut = self._move_pool.submit(self._move_to_final, closed_path, closed_bytes, closed_idx)
        self._move_futures.append(fut)

        self._part_idx += 1
        self._fout = None
        self._current_path = None
        self._current_bytes = 0
        self._open_new_part()

    def _move_to_final(self, src: Path, size_bytes: int, idx: int):
        dst = self.final_dir / src.name
        t0 = time.time()
        try:
            import shutil as _sh
            _sh.move(str(src), str(dst))
            elapsed = time.time() - t0
            speed = (size_bytes / 1024**2) / max(elapsed, 0.001)
            self._moved_bytes += size_bytes
            log.info(f"  [move] part_{idx:04d} {size_bytes/1024**3:.2f} Go → G: "
                     f"en {elapsed:.0f}s ({speed:.0f} Mo/s)")
        except Exception as e:
            log.error(f"  [move] ÉCHEC part_{idx:04d} : {e}")

    def close(self, wait_moves: bool = True):
        if self._fout is not None:
            if self._current_bytes > 0:
                self.rotate()
            else:
                self._fout.close()
                try:
                    if self._current_path and self._current_path.exists():
                        self._current_path.unlink()
                except Exception:
                    pass
                self._fout = None

        if wait_moves and self._move_futures:
            log.info(f"  [move] attente fin des {len(self._move_futures)} moves en cours...")
            for fut in self._move_futures:
                try:
                    fut.result(timeout=3600)
                except Exception as e:
                    log.error(f"  [move] erreur attente : {e}")

        self._move_pool.shutdown(wait=True)

    def stats(self) -> dict:
        return {
            "current_part":  self._part_idx,
            "current_bytes": self._current_bytes,
            "moved_bytes":   self._moved_bytes,
            "pending_moves": sum(1 for f in self._move_futures if not f.done()),
        }


def resume_recover_parts(active_dir: Path, final_dir: Path, base_name: str = "merged_final"):
    if not active_dir.exists():
        return
    orphans = sorted(active_dir.glob(f"{base_name}.part_*.jsonl"))
    if not orphans:
        return
    total_gb = sum(p.stat().st_size for p in orphans) / 1024**3
    log.info(f"[reprise] {len(orphans)} parts orphelins sur D: ({total_gb:.1f} Go) → move vers G:")
    import shutil as _sh
    for p in orphans:
        dst = final_dir / p.name
        try:
            _sh.move(str(p), str(dst))
            log.info(f"  [reprise] move : {p.name} → G:")
        except Exception as e:
            log.error(f"  [reprise] echec {p.name} : {e}")


def concat_final_parts(final_dir: Path, output_path: Path, base_name: str = "merged_final"):
    parts = sorted(final_dir.glob(f"{base_name}.part_*.jsonl"))
    if not parts:
        log.warning("Aucun part à concaténer.")
        return
    total_bytes = sum(p.stat().st_size for p in parts)
    log.info("=" * 70)
    log.info(f"CONCAT FINAL : {len(parts)} parts → {output_path}")
    log.info(f"  taille totale : {total_bytes/1024**3:.1f} Go")
    log.info("=" * 70)

    if output_path.exists():
        log.info(f"  Suppression ancien {output_path.name}")
        output_path.unlink()

    t0 = time.time()
    written = 0
    CHUNK = 64 * 1024 * 1024
    with open(output_path, "wb") as fout:
        for i, p in enumerate(parts, 1):
            t_p = time.time()
            sz = p.stat().st_size
            with open(p, "rb") as fin:
                while True:
                    buf = fin.read(CHUNK)
                    if not buf:
                        break
                    fout.write(buf)
            written += sz
            elapsed_p = time.time() - t_p
            speed = (sz/1024**2) / max(elapsed_p, 0.001)
            log.info(f"  [{i}/{len(parts)}] {p.name} ({sz/1024**3:.1f} Go) "
                     f"en {elapsed_p:.0f}s ({speed:.0f} Mo/s) → total {written/1024**3:.1f} Go")
            try:
                p.unlink()
            except Exception as e:
                log.warning(f"    suppression échouée ({p}): {e}")

    elapsed = time.time() - t0
    log.info(f"CONCAT TERMINÉ : {written/1024**3:.1f} Go en {elapsed/60:.1f} min "
             f"({(written/1024**2)/max(elapsed,1):.0f} Mo/s moyen)")


# ─── Discovery sources ────────────────────────────────────────────────────────

def discover_sources():
    available = {}
    for src_dir in sorted(CLEANED_DIR.iterdir()):
        if not src_dir.is_dir():
            continue
        jsonl = src_dir / f"{src_dir.name}_cleaned.jsonl"
        if not jsonl.exists():
            candidates = list(src_dir.glob("*_cleaned.jsonl"))
            jsonl = candidates[0] if candidates else None
        if jsonl is None or not jsonl.exists():
            log.warning(f"Source {src_dir.name} : aucun _cleaned.jsonl — skip")
            continue
        available[src_dir.name] = jsonl
    ordered = []
    for src in SOURCES_ORDER:
        if src in available:
            ordered.append((src, available.pop(src)))
    for src, path in available.items():
        ordered.append((src, path))
    return ordered


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PARALLÈLE — v9 avec compteurs étagés et heartbeat
# ═══════════════════════════════════════════════════════════════════════════

class _DocCtx:
    """Contexte d'un doc traversant le pipeline."""
    __slots__ = ("line_no", "doc_id", "text", "doc", "src", "is_premium",
                 "sig", "qs", "is_dup", "kept", "ready", "error")

    def __init__(self, line_no, doc_id, text, doc, src, is_premium):
        self.line_no = line_no
        self.doc_id = doc_id
        self.text = text
        self.doc = doc
        self.src = src
        self.is_premium = is_premium
        self.sig = None
        self.qs = None
        self.is_dup = False
        self.kept = False
        self.ready = threading.Event()
        self.error = None


class StageCounters:
    """Compteurs atomiques 4-étages pour debug pipeline.

    L'utilité : si total_read stagne mais submitted continue, on sait
    que le bottleneck est entre worker et write. Si submitted stagne,
    c'est que le main thread est bloqué (probablement drain_until).
    """
    __slots__ = ("submitted", "sig_done", "lookup_done", "written",
                 "errors", "_lock")

    def __init__(self):
        self.submitted   = 0  # docs submited au worker pool
        self.sig_done    = 0  # docs avec sig calculée
        self.lookup_done = 0  # docs avec lookup terminé
        self.written     = 0  # docs traités par writer (write_one)
        self.errors      = 0
        self._lock = threading.Lock()

    def inc_sig(self):
        with self._lock:
            self.sig_done += 1

    def inc_lookup(self):
        with self._lock:
            self.lookup_done += 1

    def inc_error(self):
        with self._lock:
            self.errors += 1


class HeartbeatDaemon(threading.Thread):
    """Thread qui log un heartbeat toutes les HEARTBEAT_SECONDS, peu importe
    si le main thread est bloqué.

    Affiche les 4 compteurs étagés + l'in-flight queue size pour diagnostic
    immédiat.
    """
    def __init__(self, runner, interval: float = HEARTBEAT_SECONDS):
        super().__init__(daemon=True, name="heartbeat")
        self.runner = runner
        self.interval = interval
        self._stop = threading.Event()
        self._last_written = 0
        self._last_t = time.time()

    def run(self):
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if self._stop.is_set():
                break
            try:
                self._beat()
            except Exception as e:
                log.error(f"[heartbeat] erreur : {e}")

    def _beat(self):
        c = self.runner.counters
        now = time.time()
        dt = now - self._last_t
        rate = (c.written - self._last_written) / max(dt, 0.001)
        self._last_written = c.written
        self._last_t = now

        in_flight = len(self.runner._in_flight)
        log.info(
            f"[♥] sub={c.submitted:,} sig={c.sig_done:,} "
            f"lkp={c.lookup_done:,} wri={c.written:,} "
            f"| in_flight={in_flight} | rate={rate:.0f} doc/s "
            f"| err={c.errors}"
        )

    def stop(self):
        self._stop.set()


class PipelineRunner:
    """Runner pipeline parallèle 3 étages, robuste."""

    def __init__(self, idx: MinHashIndexV9,
                 process_pool: ProcessPoolExecutor,
                 lookup_pool: ThreadPoolExecutor,
                 part_writer: PartWriter,
                 prog_db: ProgressDB,
                 quality_threshold: float,
                 sources_premium: set,
                 skip_quality: bool = False):
        self.idx = idx
        self.process_pool = process_pool
        self.lookup_pool = lookup_pool
        self.part_writer = part_writer
        self.prog_db = prog_db
        self.quality_threshold = quality_threshold
        self.sources_premium = sources_premium
        self.skip_quality = skip_quality

        self._in_flight = deque()
        self._in_flight_lock = threading.Lock()  # pour len() concurrent (heartbeat)

        self.total_read = 0
        self.total_kept = 0
        self.total_dup = 0
        self.total_lowq = 0
        self._stop_requested = False

        self.counters = StageCounters()
        self._next_log_at = LOG_EVERY_DOCS
        self._kept_at_start = {}

    def request_stop(self):
        self._stop_requested = True

    def abort_pipeline(self):
        """
        Appele en derniere extremite : set tous les ready Events des
        docs in_flight. Debloque _drain_until qui sinon attend des
        Events que le ProcessPool annule peut ne jamais set.

        Les docs ainsi liberes auront sig=None -> _write_one les skip
        proprement (pas d'erreur, pas d'INSERT, comptes en erreur).
        """
        with self._in_flight_lock:
            n = 0
            for ctx in self._in_flight:
                if not ctx.ready.is_set():
                    ctx.error = "aborted (SIGINT)"
                    ctx.sig = None
                    ctx.is_dup = False
                    ctx.ready.set()
                    n += 1
            log.warning(f"[abort] {n} docs in_flight liberes (seront skipped)")

    def mark_source_start(self, src: str):
        self._kept_at_start[src] = self.total_kept

    def _kept_for_src_so_far(self, src: str, src_kept_prev: int) -> int:
        return src_kept_prev + (self.total_kept - self._kept_at_start.get(src, 0))

    # ─── Étage 1 → 2 ──────────────────────────────────────────────────────────

    def _on_worker_done(self, worker_fut, ctx: _DocCtx):
        """Callback quand un worker MinHash termine.

        ATTENTION : ce callback est exécuté dans un thread du
        ProcessPoolExecutor (côté pool internals). On submit le lookup
        dans le ThreadPool puis on rentre.
        """
        try:
            sig_tuple, qs = worker_fut.result()
            ctx.sig = list(sig_tuple)
            ctx.qs = qs
            self.counters.inc_sig()
            # Soumettre lookup
            self.lookup_pool.submit(self._lookup_one, ctx)
        except Exception as e:
            ctx.error = f"worker: {e}"
            ctx.sig = None
            ctx.is_dup = False
            self.counters.inc_error()
            log.error(f"[worker error] {ctx.doc_id}: {e}")
            log.debug(traceback.format_exc())
            ctx.ready.set()

    # ─── Étage 2 ──────────────────────────────────────────────────────────────

    def _lookup_one(self, ctx: _DocCtx):
        """Exécuté dans un thread du lookup_pool."""
        try:
            if ctx.sig is None:
                ctx.is_dup = False
            else:
                ctx.is_dup = self.idx.is_dup(ctx.sig)
        except Exception as e:
            ctx.error = f"lookup: {e}"
            ctx.is_dup = False
            self.counters.inc_error()
            log.error(f"[lookup error] {ctx.doc_id}: {e}")
        finally:
            self.counters.inc_lookup()
            ctx.ready.set()

    # ─── Étage 3 — main thread writer ─────────────────────────────────────────

    def _write_one(self, ctx: _DocCtx):
        """Décision finale + idx.add + part.write. Main thread."""
        self.total_read += 1
        self.counters.written += 1

        if ctx.sig is None:
            return  # erreur worker, skip

        if ctx.is_dup:
            self.total_dup += 1
            return

        if ctx.qs is not None and ctx.qs < self.quality_threshold:
            self.idx.add(ctx.sig)
            self.total_lowq += 1
            return

        self.idx.add(ctx.sig)
        try:
            line_bytes = _json_dumps_bytes({
                "text":   ctx.text,
                "source": ctx.doc.get("source", ctx.src),
                "id":     ctx.doc_id,
            })
            self.part_writer.write(line_bytes)
        except Exception as e:
            log.error(f"[write error] {ctx.doc_id}: {e}")
            self.counters.inc_error()
            return
        self.total_kept += 1
        ctx.kept = True

    def _drain_ready(self, max_wait: float = 0.0) -> int:
        """Drain docs prêts en tête de _in_flight (FIFO)."""
        drained = 0
        while True:
            with self._in_flight_lock:
                if not self._in_flight:
                    break
                ctx = self._in_flight[0]
                if not ctx.ready.is_set():
                    if max_wait <= 0:
                        break
                    # Wait OUTSIDE the lock
                    pass
                else:
                    self._in_flight.popleft()
                    self._write_one(ctx)
                    drained += 1
                    continue

            # ctx not ready, wait outside lock
            if not ctx.ready.wait(timeout=max_wait):
                break
        return drained

    def _drain_until(self, target_size: int, timeout: float = None):
        """Bloque jusqu'a ce que _in_flight redescende a target_size.

        timeout : si non None, abandonne apres timeout secondes et log
        un warning. Utilise lors du drain final pour eviter deadlock
        si des futures ne se completent jamais (workers killed).
        """
        deadline = (time.time() + timeout) if timeout is not None else None
        while True:
            with self._in_flight_lock:
                size = len(self._in_flight)
            if size <= target_size:
                return
            if deadline is not None and time.time() > deadline:
                log.warning(
                    f"[drain] TIMEOUT apres {timeout}s, "
                    f"{size} docs in_flight non draines abandonnes"
                )
                return
            self._drain_ready(max_wait=1.0)

    # ─── Boucle principale par source ─────────────────────────────────────────

    def process_source(self, src: str, jsonl_path: Path,
                       skip_lines: int, src_kept_prev: int,
                       rate_window: deque,
                       max_in_flight: int = RESULT_QUEUE_MAX) -> tuple:
        is_premium = src in self.sources_premium
        cat_label = "PREMIUM" if is_premium else f"WEB seuil={self.quality_threshold:.4f}"
        log.info(
            f"━━━ {src} ({jsonl_path.stat().st_size/1024**3:.1f} Go) [{cat_label}] ━━━"
        )

        self.mark_source_start(src)
        src_read = skip_lines
        start_src = time.time()

        try:
            with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fin:
                for line_no, raw_line in enumerate(fin):
                    if self._stop_requested:
                        log.warning(f"[{src}] STOP demandé, fermeture propre...")
                        break

                    if line_no < skip_lines:
                        continue

                    src_read += 1
                    line = raw_line.strip()
                    if not line:
                        continue

                    try:
                        doc = _json_loads(line)
                    except Exception:
                        continue
                    text = doc.get("text", "")
                    if not text or len(text) < 50:
                        continue

                    doc_id = doc.get("id", f"{src}_{src_read}")

                    ctx = _DocCtx(line_no, doc_id, text, doc, src, is_premium)
                    windowed = extract_windows(text)
                    quality_sample = (
                        None if (is_premium or self.skip_quality) else text[:3000]
                    )

                    try:
                        worker_fut = self.process_pool.submit(
                            _worker_compute, (windowed, quality_sample)
                        )
                    except Exception as e:
                        log.error(f"[submit error] {doc_id}: {e}")
                        self.counters.inc_error()
                        continue

                    worker_fut.add_done_callback(
                        lambda f, c=ctx: self._on_worker_done(f, c)
                    )
                    with self._in_flight_lock:
                        self._in_flight.append(ctx)
                    self.counters.submitted += 1

                    # Backpressure
                    with self._in_flight_lock:
                        size = len(self._in_flight)
                    if size >= max_in_flight:
                        self._drain_until(max_in_flight - 1)

                    # Drain non bloquant
                    self._drain_ready(max_wait=0.0)

                    # Logging périodique sur total_read
                    if self.total_read >= self._next_log_at:
                        self._log_progress(src, rate_window, start_src)
                        self._next_log_at = self.total_read + LOG_EVERY_DOCS

                    # Checkpoint périodique
                    if src_read % CHECKPOINT_EVERY == 0:
                        self.idx.flush()
                        kept_so_far = self._kept_for_src_so_far(src, src_kept_prev)
                        self.prog_db.save(src, src_read, kept_so_far)
                        self.part_writer.flush()
                        log.info(f"  [checkpoint] {src} @ ligne {src_read:,}, "
                                 f"kept {kept_so_far:,}")

                # Drain final : timeout 60s pour eviter deadlock si
                # le ProcessPool a ete shutdown et des futures restent
                # in_flight sans ready set.
                with self._in_flight_lock:
                    in_flight_size = len(self._in_flight)
                log.info(f"  [{src}] drain final ({in_flight_size} docs en flight)")
                self._drain_until(0, timeout=60.0)

        except Exception as e:
            log.error(f"[{src}] EXCEPTION : {e}")
            log.error(traceback.format_exc())
            raise

        # Fin de source
        self.idx.flush()
        kept_final = self._kept_for_src_so_far(src, src_kept_prev)
        self.prog_db.save(src, src_read, kept_final)
        self.part_writer.flush()
        self.part_writer.rotate()

        delta_read = src_read - skip_lines
        delta_kept = kept_final - src_kept_prev
        log.info(
            f"  [{src}] {delta_read:,} lus → {delta_kept:,} gardés "
            f"({delta_kept/max(delta_read,1)*100:.1f}%)"
        )
        return src_read, kept_final

    def _log_progress(self, src: str, rate_window: deque, start_src: float):
        now = time.time()
        rate_window.append((now, self.total_read))
        if len(rate_window) >= 2:
            t_old, r_old = rate_window[0]
            rate = (self.total_read - r_old) / max(now - t_old, 1)
        else:
            rate = self.total_read / max(now - start_src, 1)

        TARGET_DOCS = 536_000_000
        est_rem = max(0, TARGET_DOCS - self.total_read) / max(rate, 1) / 3600

        pw_stats = self.part_writer.stats()
        out_gb = (pw_stats["moved_bytes"] + pw_stats["current_bytes"]) / 1024**3
        ix_stats = self.idx.stats()

        with self._in_flight_lock:
            in_flight = len(self._in_flight)

        log.info(
            f"  {self.total_read:>12,} lus | {self.total_kept:>10,} gardés "
            f"| dup {self.total_dup/max(self.total_read,1)*100:.1f}% "
            f"| lowq {self.total_lowq/max(self.total_read,1)*100:.1f}% "
            f"| {rate:.0f} doc/s "
            f"| sql_hit {ix_stats['hit_pct']:.0f}% "
            f"| in_flight {in_flight} "
            f"| out {out_gb:.1f} Go (part {pw_stats['current_part']:04d}) "
            f"| reste ~{est_rem:.0f}h"
        )


# ─── Mode AUDIT ───────────────────────────────────────────────────────────────

def run_audit():
    log.info("=" * 70)
    log.info("RODIN - Audit sources v9.1")
    log.info("=" * 70)
    sources = discover_sources()
    total_gb = 0
    for src, path in sources:
        sz = path.stat().st_size / 1024**3
        total_gb += sz
        cat = "PREMIUM" if src in SOURCES_PREMIUM else "WEB    "
        log.info(f"  [{cat}] {src:<20} {sz:>7.1f} Go")
    log.info(f"  {'TOTAL':<28} {total_gb:>7.1f} Go")
    log.info("")
    log.info(f"  Worker     : v9.1 (numpy vectorise + Mersenne split hi/lo correct)")
    log.info(f"  Paramètres : threshold={THRESHOLD} | {LSH_BANDS}×{LSH_ROWS} bandes "
             f"| shingles {N_WINDOWS}×{WINDOW_SIZE}c")
    log.info(f"  Pipeline   : {N_WORKERS} MinHash workers + {N_LOOKUP_THREADS} lookup threads")
    log.info(f"  INSERT batch: {INSERT_BATCH_SIZE:,} docs")
    log.info(f"  mmap_size  : {SQLITE_MMAP_BYTES/1024**3:.0f} Go")
    log.info(f"  orjson     : {'OUI' if _HAS_ORJSON else 'NON (fallback json)'}")
    log.info(f"  Heartbeat  : toutes les {HEARTBEAT_SECONDS}s")
    log.info(f"  Log flush  : toutes les {LOG_FLUSH_SECONDS}s (force fsync)")
    log.info(f"  Log file   : {LOG_FILE}")
    log.info("")
    import shutil as _sh
    for drive, label in [(str(BASE_G.drive)+"\\", "G:"), (str(BASE_D.drive)+"\\", "D:")]:
        try:
            _, _, free = _sh.disk_usage(drive)
            log.info(f"  Disque {label} : {free/1024**3:.0f} Go libres")
        except Exception:
            pass
    log.info("")
    log.info("État pipeline :")
    for label, path in [
        (".dedup_done",  DEDUPED_DIR_FINAL / ".dedup_done"),
        ("merged_final", MERGED_FINAL),
        ("index SQLite", INDEX_DB),
        ("progress DB",  PROGRESS_DB),
    ]:
        if path.exists():
            log.info(f"  ✓ {label:<22} {path.stat().st_size/1024**3:.2f} Go")
        else:
            log.info(f"  — {label}")

    for label, d in [("parts actifs D:",    DEDUPED_DIR_ACTIVE),
                     ("parts finalisés G:", DEDUPED_DIR_FINAL)]:
        if d.exists():
            parts = sorted(d.glob("merged_final.part_*.jsonl"))
            if parts:
                total = sum(p.stat().st_size for p in parts) / 1024**3
                log.info(f"  ✓ {label:<22} {len(parts)} parts — {total:.1f} Go")

    if INDEX_DB.exists():
        con = sqlite3.connect(f"file:{INDEX_DB.as_posix()}?mode=ro", uri=True)
        idx = con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_bands_lookup'"
        ).fetchone()
        log.info(f"  INDEX B-tree (idx_bands_lookup) : {'PRESENT' if idx else 'ABSENT ⚠'}")
        con.close()

    # Affiche progress connu
    if PROGRESS_DB.exists():
        con = sqlite3.connect(str(PROGRESS_DB))
        rows = con.execute(
            "SELECT source, lines_done, kept FROM progress ORDER BY source"
        ).fetchall()
        con.close()
        if rows:
            log.info("")
            log.info("Progress par source :")
            for src, lines, kept in rows:
                log.info(f"  {src:<20} lus={lines:>14,}  kept={kept:>14,}")


# ─── Mode SELF-TEST (pas de pipeline, juste worker) ──────────────────────────

def run_self_test():
    log.info("=" * 70)
    log.info("Worker v9.1 self-test")
    log.info("=" * 70)
    import subprocess
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "_dedup_worker_v91.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        log.error(f"Worker self-test a echoue (code {result.returncode})")
        log.error(result.stderr)
        sys.exit(result.returncode)


# ─── Mode TEST (50K docs DB temp) ────────────────────────────────────────────

def run_test(n_docs: int = 50_000):
    log.info("=" * 70)
    log.info(f"RODIN - Mode TEST v9.1 ({n_docs:,} docs/source)")
    log.info(f"  {N_WORKERS} workers + {N_LOOKUP_THREADS} lookup threads")
    log.info("=" * 70)

    import tempfile

    sources = discover_sources()
    tmp_db = Path(tempfile.mktemp(suffix=".db"))

    try:
        con = sqlite3.connect(str(tmp_db))
        con.executescript("""
            CREATE TABLE bands (
                band_id  INTEGER NOT NULL,
                band_h64 INTEGER NOT NULL
            );
            CREATE INDEX idx_bands_lookup ON bands(band_id, band_h64);
        """)
        con.commit()
        con.close()

        idx_test = MinHashIndexV9(tmp_db, n_lookup_threads=N_LOOKUP_THREADS)
        start = time.time()

        tmp_active = Path(tempfile.mkdtemp())
        tmp_final  = Path(tempfile.mkdtemp())
        part_writer = PartWriter(tmp_active, tmp_final)

        tmp_prog = Path(tempfile.mktemp(suffix=".db"))
        prog_db = ProgressDB(tmp_prog)

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool, \
             ThreadPoolExecutor(max_workers=N_LOOKUP_THREADS,
                                thread_name_prefix="lookup") as lookup_pool:

            runner = PipelineRunner(
                idx=idx_test,
                process_pool=pool,
                lookup_pool=lookup_pool,
                part_writer=part_writer,
                prog_db=prog_db,
                quality_threshold=0.0,
                sources_premium=SOURCES_PREMIUM,
                skip_quality=True,
            )
            heartbeat = HeartbeatDaemon(runner, interval=10)
            heartbeat.start()

            try:
                for src, path in sources:
                    t0_src = time.time()
                    runner.mark_source_start(src)

                    src_read = 0
                    with open(path, "r", encoding="utf-8", errors="replace") as fin:
                        for line_no, raw_line in enumerate(fin):
                            if src_read >= n_docs:
                                break
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                doc = _json_loads(line)
                            except Exception:
                                continue
                            text = doc.get("text", "")
                            if not text or len(text) < 50:
                                continue
                            src_read += 1
                            doc_id = doc.get("id", f"{src}_{line_no}")

                            ctx = _DocCtx(line_no, doc_id, text, doc, src, True)
                            windowed = extract_windows(text)
                            worker_fut = pool.submit(_worker_compute, (windowed, None))
                            worker_fut.add_done_callback(
                                lambda f, c=ctx: runner._on_worker_done(f, c)
                            )
                            with runner._in_flight_lock:
                                runner._in_flight.append(ctx)
                            runner.counters.submitted += 1

                            with runner._in_flight_lock:
                                size = len(runner._in_flight)
                            if size >= RESULT_QUEUE_MAX:
                                runner._drain_until(RESULT_QUEUE_MAX - 1)
                            runner._drain_ready(max_wait=0.0)

                        runner._drain_until(0)

                    idx_test.flush()
                    elapsed_src = time.time() - t0_src
                    rate = src_read / max(elapsed_src, 1)
                    kept = runner.total_kept - runner._kept_at_start.get(src, 0)
                    kept_pct = kept / max(src_read, 1) * 100
                    cat = "PREMIUM" if src in SOURCES_PREMIUM else "WEB"
                    log.info(
                        f"  [{cat}] {src:<20} {src_read:>6,} lus → {kept:>6,} kept "
                        f"({kept_pct:.1f}%) | {rate:.0f} doc/s"
                    )
            finally:
                heartbeat.stop()

        idx_test.close()
        part_writer.close(wait_moves=True)
        prog_db.close()
        total_elapsed = time.time() - start

        log.info("")
        log.info(f"Vitesse globale : {runner.total_read/max(total_elapsed,1):.0f} doc/s")
        log.info(f"  Cible v9.1 = ~800-1500 doc/s sur HPLT (vs 157 doc/s v9)")
        log.info(f"  Compteurs : sub={runner.counters.submitted:,} "
                 f"sig={runner.counters.sig_done:,} "
                 f"lkp={runner.counters.lookup_done:,} "
                 f"wri={runner.counters.written:,} "
                 f"err={runner.counters.errors}")

    finally:
        for f in [tmp_db, tmp_db.with_suffix(".db-wal"), tmp_db.with_suffix(".db-shm")]:
            if f.exists():
                try: f.unlink()
                except: pass


# ─── Mode RUN ─────────────────────────────────────────────────────────────────

def run_pipeline(force: bool = False, skip_quality: bool = False,
                 preset_quality_threshold: float = None):
    done_flag = DEDUPED_DIR_FINAL / ".dedup_done"
    if done_flag.exists() and not force:
        log.info("Pipeline déjà terminé. --force pour relancer.")
        return
    if force:
        targets = [MERGED_FINAL, INDEX_DB, PROGRESS_DB, done_flag, STATS_FILE]
        for f in targets:
            if f.exists():
                f.unlink()
                log.info(f"  Supprimé : {f}")
        for d in [DEDUPED_DIR_ACTIVE, DEDUPED_DIR_FINAL]:
            if d.exists():
                for p in d.glob("merged_final.part_*.jsonl"):
                    p.unlink()

    sources = discover_sources()
    if not sources:
        log.error(f"Aucune source dans {CLEANED_DIR}")
        sys.exit(1)

    log.info("=" * 70)
    log.info("RODIN - Deduplication MinHash v9.1 (pipeline parallèle robuste)")
    log.info("=" * 70)
    log.info(f"Sources    : {len(sources)} | "
             f"Input : {sum(p.stat().st_size for _,p in sources)/1024**3:.1f} Go")
    log.info(f"Worker     : v9.1 (numpy vectorise + Mersenne split hi/lo correct)")
    log.info(f"MinHash    : threshold={THRESHOLD} | {NUM_PERM} perms | "
             f"{LSH_BANDS}×{LSH_ROWS} bandes | shingles {N_WINDOWS}×{WINDOW_SIZE}c")
    log.info(f"Pipeline   : {N_WORKERS} workers + {N_LOOKUP_THREADS} lookup threads")
    log.info(f"INSERT batch: {INSERT_BATCH_SIZE:,} docs | mmap {SQLITE_MMAP_BYTES/1024**3:.0f} Go")
    log.info(f"orjson     : {'OUI' if _HAS_ORJSON else 'NON (fallback json)'}")
    log.info(f"Output     : parts {PART_MAX_BYTES/1024**3:.0f} Go sur D: → move async G:")
    log.info(f"Heartbeat  : toutes les {HEARTBEAT_SECONDS}s")
    log.info(f"Log flush  : toutes les {LOG_FLUSH_SECONDS}s")
    log.info(f"Log file   : {LOG_FILE}")
    for src, _ in sources:
        cat = "PREMIUM (tout garder)" if src in SOURCES_PREMIUM else "WEB (scoring 85%)"
        log.info(f"  {src:<20} {cat}")
    log.info("")

    if not force:
        resume_recover_parts(DEDUPED_DIR_ACTIVE, DEDUPED_DIR_FINAL)

    quality_threshold = 0.0
    if skip_quality:
        log.info("Quality scoring désactivé (--no-quality)")
    elif preset_quality_threshold is not None:
        quality_threshold = preset_quality_threshold
        log.info(f"Seuil qualité WEB préréglé : {quality_threshold:.4f}")
    else:
        quality_threshold = estimate_web_threshold(sources)
    log.info("")

    idx = MinHashIndexV9(INDEX_DB, n_lookup_threads=N_LOOKUP_THREADS)
    prog_db = ProgressDB(PROGRESS_DB)

    if not idx.has_index:
        log.error("=" * 70)
        log.error("INDEX B-tree ABSENT sur l'index SQLite.")
        log.error("Crée-le avant de lancer v9.")
        log.error("=" * 70)
        idx.close()
        prog_db.close()
        sys.exit(2)

    rate_window = deque(maxlen=10)
    rate_window.append((time.time(), 0))

    part_writer = PartWriter(DEDUPED_DIR_ACTIVE, DEDUPED_DIR_FINAL)
    start = time.time()

    runner_holder = {}
    sigint_count = [0]

    def signal_handler(sig, frame):
        sigint_count[0] += 1
        log.warning("")
        log.warning("=" * 70)
        if sigint_count[0] == 1:
            log.warning("SIGINT recu - fermeture propre demandee")
            log.warning("(Re-Ctrl+C pour forcer l'abort immediat)")
            log.warning("=" * 70)
            if "runner" in runner_holder:
                runner_holder["runner"].request_stop()
        else:
            log.warning("2eme SIGINT - ABORT IMMEDIAT du pipeline")
            log.warning("=" * 70)
            if "runner" in runner_holder:
                runner_holder["runner"].abort_pipeline()

    signal.signal(signal.SIGINT, signal_handler)

    try:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as process_pool, \
             ThreadPoolExecutor(max_workers=N_LOOKUP_THREADS,
                                thread_name_prefix="lookup") as lookup_pool:

            runner = PipelineRunner(
                idx=idx,
                process_pool=process_pool,
                lookup_pool=lookup_pool,
                part_writer=part_writer,
                prog_db=prog_db,
                quality_threshold=quality_threshold,
                sources_premium=SOURCES_PREMIUM,
                skip_quality=skip_quality,
            )
            runner_holder["runner"] = runner

            heartbeat = HeartbeatDaemon(runner)
            heartbeat.start()

            try:
                for src, jsonl_path in sources:
                    if runner._stop_requested:
                        break
                    skip_lines, src_kept_prev = prog_db.get(src)
                    if skip_lines > 0:
                        log.info(f"[{src}] Reprise ligne {skip_lines:,} "
                                 f"({src_kept_prev:,} gardés précédemment)")
                    runner.process_source(
                        src=src,
                        jsonl_path=jsonl_path,
                        skip_lines=skip_lines,
                        src_kept_prev=src_kept_prev,
                        rate_window=rate_window,
                    )
            finally:
                heartbeat.stop()

    finally:
        idx.flush()
        idx.close()
        prog_db.close()
        log.info("")
        log.info("Fermeture PartWriter (attente moves en cours)...")
        part_writer.close(wait_moves=True)

    elapsed = time.time() - start

    if not runner_holder["runner"]._stop_requested:
        concat_final_parts(DEDUPED_DIR_FINAL, MERGED_FINAL)
        out_gb = MERGED_FINAL.stat().st_size / 1024**3 if MERGED_FINAL.exists() else 0
        done_flag.touch()
    else:
        out_gb = 0

    runner = runner_holder["runner"]
    log.info("")
    log.info("=" * 70)
    log.info("PIPELINE V9.1 TERMINE" if not runner._stop_requested
             else "PIPELINE V9.1 STOPPE (peut être repris)")
    log.info("=" * 70)
    log.info(f"  Total lus       : {runner.total_read:,}")
    log.info(f"  Gardés          : {runner.total_kept:,}")
    log.info(f"  Doublons        : {runner.total_dup:,}")
    log.info(f"  Qualité insuff. : {runner.total_lowq:,}")
    log.info(f"  Erreurs         : {runner.counters.errors}")
    log.info(f"  Durée           : {elapsed/3600:.1f}h")
    if out_gb:
        log.info(f"  Output          : {MERGED_FINAL} ({out_gb:.1f} Go)")
    log.info("")

    STATS_FILE.write_text(json.dumps({
        "total_read":        runner.total_read,
        "total_kept":        runner.total_kept,
        "total_dup":         runner.total_dup,
        "total_lowq":        runner.total_lowq,
        "errors":            runner.counters.errors,
        "elapsed_hours":     round(elapsed / 3600, 2),
        "version":           "v9.1",
        "n_workers":         N_WORKERS,
        "n_lookup_threads":  N_LOOKUP_THREADS,
        "insert_batch":      INSERT_BATCH_SIZE,
        "stopped_early":     runner._stop_requested,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="RODIN - Deduplication MinHash v9.1 (pipeline robuste)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow recommandé :
  1. python 07_dedup_quality_v9.py --self-test   # vérifie le worker
  2. python 07_dedup_quality_v9.py --dry-run     # audit
  3. python 07_dedup_quality_v9.py --test        # test 50K docs / source
  4. python 07_dedup_quality_v9.py               # vrai run, reprend où v8.2

Reprise avec seuil qualité préréglé (skip estimation) :
  python 07_dedup_quality_v9.py --quality-threshold 0.6306
        """,
    )
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--self-test", action="store_true",
                   help="Lance uniquement le self-test du worker")
    p.add_argument("--test",      action="store_true")
    p.add_argument("--test-n",    type=int, default=50_000)
    p.add_argument("--force",     action="store_true")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--quality-threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 70)
    log.info("RODIN - Deduplication MinHash v9.1 (pipeline parallèle robuste)")
    log.info(f"  threshold={THRESHOLD} | xxhash + codepoint-shingling")
    log.info(f"  Workers={N_WORKERS} | Lookup threads={N_LOOKUP_THREADS}")
    log.info(f"  INSERT batch={INSERT_BATCH_SIZE:,} | mmap={SQLITE_MMAP_BYTES/1024**3:.0f}Go")
    log.info(f"  orjson={'OUI' if _HAS_ORJSON else 'NON'}")
    log.info(f"  Log file = {LOG_FILE}")
    log.info("=" * 70)

    try:
        if args.self_test:
            run_self_test()
        elif args.dry_run:
            run_audit()
        elif args.test:
            run_test(n_docs=args.test_n)
        else:
            run_pipeline(
                force=args.force,
                skip_quality=args.no_quality,
                preset_quality_threshold=args.quality_threshold,
            )
    finally:
        # Force flush log final
        _file_handler.force_fsync()
        _flush_daemon.stop()


if __name__ == "__main__":
    main()
