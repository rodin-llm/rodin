# 17_locate_ghost.py
# Localise la ligne fantome par dichotomie sur le predicat
# "decode(tokens du doc i) == text de la ligne i".
# Etape 1 : index d'offsets de lignes (scan ~18 min, une fois)
# Etape 2 : dichotomie (~28 sondes, quelques secondes)
# Etape 3 : autopsie de la ligne trouvee (orjson vs json stdlib)
#
# Usage :
#   python -u .\scripts\17_locate_ghost.py            (tout)
#   python -u .\scripts\17_locate_ghost.py --skip-offsets  (index deja la)

import argparse
import json
import os
import time

import numpy as np
import orjson
import sentencepiece as spm

IN_PATH = r"G:\data\rodin\deduped\merged_final.jsonl"
SP_MODEL = r"G:\data\rodin\bpe\rodin.model"
SHARD_DIR = r"G:\data\rodin\tokenized"
MANIFEST = os.path.join(SHARD_DIR, "manifest.json")
IDX_DIR = r"D:\rodin_index"
EOS_OFFSETS = os.path.join(IDX_DIR, "doc_eos_offsets.u64.tmp")  # du 15
LINE_OFFSETS = os.path.join(IDX_DIR, "line_offsets.u64")
OUT_REPORT = os.path.join(IDX_DIR, "ghost_located.json")

TEXT_FIELDS = ("text", "content", "raw_content", "body", "page_content")
MAX_TEXT_LEN = 200_000
N_DOCS = 307_379_231     # EOS reellement presents dans les shards
N_LINES = 307_379_232    # lignes du JSONL


# ---------------------------------------------------------------- etape 1
def build_line_offsets():
    print("[ETAPE 1] construction de l'index d'offsets de lignes...")
    t0 = time.time()
    next_log = t0
    in_size = os.path.getsize(IN_PATH)
    tmp = LINE_OFFSETS + ".tmp"
    buf = np.empty(4_000_000, dtype=np.uint64)
    n_buf = 0
    pos = 0
    n_lines = 0
    with open(IN_PATH, "rb") as fh, open(tmp, "wb") as out:
        for line in fh:
            buf[n_buf] = pos
            n_buf += 1
            if n_buf == buf.shape[0]:
                buf.tofile(out)
                n_buf = 0
            pos += len(line)
            n_lines += 1
            now = time.time()
            if now - next_log >= 60:
                next_log = now
                rate = pos / (now - t0)
                print(f"[PROGRESS] {pos / 1024**3:7.0f} Go | "
                      f"{n_lines:>11,} lignes | {rate / 1024**2:5.0f} Mo/s | "
                      f"ETA {(in_size - pos) / rate / 3600:.2f} h")
        buf[n_buf] = pos          # sentinelle finale = taille fichier
        n_buf += 1
        buf[:n_buf].tofile(out)
        out.flush()
        os.fsync(out.fileno())
    assert n_lines == N_LINES, f"lignes lues {n_lines:,} != {N_LINES:,}"
    os.replace(tmp, LINE_OFFSETS)
    print(f"[OK] {n_lines:,} lignes indexees en "
          f"{(time.time() - t0) / 60:.1f} min -> {LINE_OFFSETS}")


# ---------------------------------------------------------------- acces
class Corpus:
    def __init__(self):
        self.line_off = np.fromfile(LINE_OFFSETS, dtype=np.uint64)
        assert self.line_off.shape[0] == N_LINES + 1
        self.eos = np.fromfile(EOS_OFFSETS, dtype=np.uint64)
        assert self.eos.shape[0] == N_DOCS
        with open(MANIFEST, "r", encoding="utf-8") as f:
            man = json.load(f)
        self.shard_files = [e["file"] for e in man["shards"]]
        counts = np.array([e["tokens"] for e in man["shards"]],
                          dtype=np.uint64)
        self.cum = np.concatenate(([0], np.cumsum(counts)))  # len 65
        self.maps = {}
        self.fh = open(IN_PATH, "rb")
        self.sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
        self.probes = 0

    def _shard(self, si):
        if si not in self.maps:
            path = os.path.join(SHARD_DIR, self.shard_files[si])
            self.maps[si] = np.memmap(path, dtype=np.uint16, mode="r")
        return self.maps[si]

    def doc_tokens(self, i):
        """Tokens du doc i (EOS exclu), gere un eventuel chevauchement."""
        start = 0 if i == 0 else int(self.eos[i - 1]) + 1
        end = int(self.eos[i])  # position de l'EOS, exclue
        parts = []
        pos = start
        while pos < end:
            si = int(np.searchsorted(self.cum, pos, side="right")) - 1
            lo = pos - int(self.cum[si])
            hi = min(end - int(self.cum[si]),
                     int(self.cum[si + 1] - self.cum[si]))
            parts.append(np.asarray(self._shard(si)[lo:hi]))
            pos = int(self.cum[si]) + hi
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def line_raw(self, i):
        off = int(self.line_off[i])
        length = int(self.line_off[i + 1]) - off
        self.fh.seek(off)
        return self.fh.read(length)

    def line_text(self, i):
        """Texte que le script 12 aurait extrait, ou None s'il a skippe."""
        raw = self.line_raw(i).strip()
        if not raw:
            return None
        try:
            obj = orjson.loads(raw)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        for f in TEXT_FIELDS:
            v = obj.get(f)
            if isinstance(v, str) and v:
                return v[:MAX_TEXT_LEN]
        return None

    def match(self, i):
        """True si doc i des shards == ligne i du JSONL."""
        self.probes += 1
        txt = self.line_text(i)
        if txt is None:
            return False  # ligne skippee par le 12 -> ne peut pas matcher
        ids = self.doc_tokens(i)
        decoded = self.sp.decode(ids.tolist())
        ok = decoded == txt
        print(f"  [SONDE {self.probes:02d}] doc {i:>11,} -> "
              f"{'MATCH' if ok else 'MISMATCH'}")
        return ok


# ---------------------------------------------------------------- etape 2+3
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-offsets", action="store_true")
    args = ap.parse_args()

    if not args.skip_offsets or not os.path.exists(LINE_OFFSETS):
        build_line_offsets()
    else:
        print("[ETAPE 1] index d'offsets existant, saute.")

    c = Corpus()

    print("\n[ETAPE 2] dichotomie...")
    # bornes : match(0) doit etre True, match(N_DOCS-1) False
    # (sauf si le ghost est la toute derniere ligne du JSONL)
    if not c.match(0):
        print("[STOP] doc 0 ne matche pas la ligne 0 -> probleme plus "
              "profond qu'un simple skip, ne pas reparer. Autopsie ligne 0 :")
        autopsy(c, 0)
        return
    if c.match(N_DOCS - 1):
        ghost = N_LINES - 1
        print(f"[RESULTAT] tout matche jusqu'au dernier doc -> le ghost "
              f"est la DERNIERE ligne du JSONL ({ghost:,}).")
    else:
        lo, hi = 0, N_DOCS - 1   # match(lo)=True, match(hi)=False
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if c.match(mid):
                lo = mid
            else:
                hi = mid
        ghost = hi
        # raffinement local (cas pathologique de textes voisins identiques)
        while ghost > 0 and not c.match(ghost - 1):
            ghost -= 1
        print(f"\n[RESULTAT] premiere divergence au doc {ghost:,} "
              f"-> ligne fantome candidate : {ghost:,}")

    print("\n[ETAPE 3] verification + autopsie...")
    # preuve du decalage : le doc 'ghost' doit matcher la ligne ghost+1
    if ghost < N_LINES - 1:
        txt_next = c.line_text(ghost + 1)
        ids = c.doc_tokens(ghost)
        decoded = c.sp.decode(ids.tolist())
        shift_ok = (txt_next is not None and decoded == txt_next)
        print(f"[PREUVE DECALAGE] doc {ghost:,} == ligne {ghost + 1:,} : "
              f"{shift_ok}")
        if not shift_ok:
            print("[ALERTE] le decalage +1 n'est pas confirme. Inspecter "
                  "manuellement la fenetre avant toute reparation.")
    autopsy(c, ghost)


def autopsy(c, line_no):
    raw = c.line_raw(line_no)
    print(f"\n--- AUTOPSIE LIGNE {line_no:,} ---")
    print(f"longueur brute : {len(raw):,} octets")
    print(f"tete brute     : "
          f"{raw[:200].decode('utf-8', errors='replace')!r}")
    try:
        obj = orjson.loads(raw.strip())
        print(f"orjson         : OK, cles={list(obj.keys())}")
        for f in TEXT_FIELDS:
            v = obj.get(f)
            print(f"  champ {f!r:14}: type={type(v).__name__}, "
                  f"repr={repr(v)[:80]}")
    except Exception as e:
        print(f"orjson         : ECHEC -> {type(e).__name__}: {e}")
        try:
            obj = json.loads(raw.decode("utf-8", errors="strict").strip())
            print(f"json stdlib    : OK, cles={list(obj.keys())} "
                  f"(=> divergence orjson/json = cause racine)")
            t = obj.get("text")
            print(f"  text repr    : {repr(t)[:120]}")
            print(f"  source/id    : {obj.get('source')} / {obj.get('id')}")
        except Exception as e2:
            print(f"json stdlib    : ECHEC aussi -> "
                  f"{type(e2).__name__}: {e2}")

    report = {"ghost_line": line_no, "raw_len": len(raw),
              "raw_head": raw[:300].decode("utf-8", errors="replace")}
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] rapport : {OUT_REPORT}")


if __name__ == "__main__":
    main()