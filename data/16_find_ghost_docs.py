# 16_find_ghost_docs.py  (v3)
# Localise les lignes que 12_tokenize_corpus.py a SAUTEES (0 token ecrit),
# en repliquant exactement ses predicats de skip.
# Capture aussi les offsets octets de lignes-reperes (frontieres de sources)
# pour la validation par decodage croise (script 17).
#
# Usage : python -u .\scripts\16_find_ghost_docs.py
#
# Verdict attendu : exactement 1 ghost (ecart 307,379,231 EOS vs
# 307,379,232 lignes constate par le script 15).

import json
import os
import time

import orjson
import sentencepiece as spm

IN_PATH = r"G:\data\rodin\deduped\merged_final.jsonl"
SP_MODEL = r"G:\data\rodin\bpe\rodin.model"
OUT_REPORT = r"D:\rodin_index\ghost_docs.json"

# --- Alignes sur 12_tokenize_corpus.py (verifies par Select-String) ---
TEXT_FIELDS = ("text", "content", "raw_content", "body", "page_content")
MAX_TEXT_LEN = 200_000
# -----------------------------------------------------------------------

SAFE_MIN_CONTENT = 16   # >=16 octets bruts de payload texte => ligne saine
LOG_EVERY_SECS = 60

# lignes-reperes : frontieres de sources (passe A)
LANDMARK_LINES = {0, 472_066, 2_548_323, 2_555_485, 2_838_052}

KEY_A = b'"text":'    # forme compacte (orjson / separators=(',',':'))
KEY_B = b'"text": '   # forme json.dump par defaut


def fast_filter_is_safe(line: bytes) -> bool:
    """True si la ligne produira A COUP SUR >=1 token + EOS dans le 12.

    Critere : champ "text" present, valeur string, >=16 octets de payload
    brut avant la fin de ligne. Les echappements JSON (\\n, \\", \\uXXXX)
    ne font que RALLONGER la forme brute par rapport au texte parse, donc
    payload brut long => texte parse non vide => encode non vide
    (byte_fallback + add_dummy_prefix). Aucun parse JSON necessaire.

    Conservateur : au moindre doute -> False (verification complete)."""
    i = line.find(KEY_A)
    if i == -1:
        return False
    j = i + len(KEY_A)
    # tolerer espaces/tabs apres le ':' (format json.dump)
    n = len(line)
    while j < n and line[j] in b" \t":
        j += 1
    if j >= n or line[j] != ord('"'):
        return False  # valeur non-string (null, nombre...) -> verifier
    j += 1  # debut du payload
    # payload brut >= SAFE_MIN_CONTENT avant '"' fermant + fin de ligne
    return n - j >= SAFE_MIN_CONTENT + 2


def replicate_12_predicate(raw: bytes, sp) -> tuple:
    """Replique exacte des skips du script 12 (lignes 98-121).
    Retourne (skipped: bool, reason: str|None, info: dict|None)."""
    line = raw.strip()
    if not line:
        return True, "ligne_vide", None
    try:
        obj = orjson.loads(line)
    except Exception:
        return True, "json_invalide", {
            "raw_head": raw[:120].decode("utf-8", errors="replace")}
    if not isinstance(obj, dict):
        return True, "pas_un_dict", {"type": type(obj).__name__}
    txt = None
    for f in TEXT_FIELDS:
        v = obj.get(f)
        if isinstance(v, str) and v:
            txt = v
            break
    if not txt:
        return True, "texte_vide_ou_absent", {
            "source": obj.get("source"), "id": obj.get("id"),
            "cles": list(obj.keys()),
            "text_repr": repr(obj.get("text"))[:80]}
    if len(txt) > MAX_TEXT_LEN:
        txt = txt[:MAX_TEXT_LEN]
    # predicat 'if not ids' du 12 : filet mort (prouve par les asserts
    # de main) -> on n'encode que les textes ultra-courts, par rigueur
    if len(txt) <= 4:
        ids = sp.encode(txt, out_type=int)
        if not ids:
            return True, "encode_vide", {
                "source": obj.get("source"), "id": obj.get("id"),
                "text_repr": repr(txt)}
    return False, None, None


def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)

    # preuve du filet mort : textes pathologiques -> tous >=1 token
    assert sp.encode("", out_type=int) == [], "sanity: '' doit etre vide"
    for probe in (" ", "\n", "\t", "\x00", "a", "é", "\\", '"'):
        assert len(sp.encode(probe, out_type=int)) >= 1, repr(probe)

    in_size = os.path.getsize(IN_PATH)
    print(f"[INFO] entree  : {IN_PATH} ({in_size / 1024**4:.2f} To)")
    print(f"[INFO] rapport : {OUT_REPORT}")

    ghosts = []          # [{line, reason, info, byte_offset}]
    checked_full = 0
    landmarks = {}       # line_no -> byte_offset
    line_no = 0
    bytes_done = 0
    t0 = time.time()
    next_log = t0

    with open(IN_PATH, "rb") as fh:
        for line in fh:
            if line_no in LANDMARK_LINES:
                landmarks[line_no] = bytes_done

            if not fast_filter_is_safe(line):
                checked_full += 1
                skipped, reason, info = replicate_12_predicate(line, sp)
                if skipped:
                    ghosts.append({"line": line_no, "reason": reason,
                                   "info": info, "byte_offset": bytes_done})
                    print(f"[GHOST] ligne {line_no:,} | {reason} | {info}")

            bytes_done += len(line)
            line_no += 1

            now = time.time()
            if now - next_log >= LOG_EVERY_SECS:
                next_log = now
                elapsed = now - t0
                rate = bytes_done / elapsed
                eta = (in_size - bytes_done) / rate if rate else 0
                print(f"[PROGRESS] {bytes_done / 1024**3:7.0f} Go | "
                      f"{line_no:>11,} lignes | "
                      f"verif completes {checked_full:,} | "
                      f"ghosts {len(ghosts)} | "
                      f"{rate / 1024**2:5.0f} Mo/s | "
                      f"ETA {eta / 3600:.2f} h")

    elapsed = time.time() - t0
    print(f"\n[FIN] {line_no:,} lignes en {elapsed / 60:.1f} min | "
          f"{checked_full:,} verifications completes | "
          f"{len(ghosts)} ghost(s)")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump({"input": IN_PATH,
                   "lines_total": line_no,
                   "full_checks": checked_full,
                   "ghosts": ghosts,
                   "landmarks_byte_offsets": {
                       str(k): v for k, v in sorted(landmarks.items())}},
                  f, ensure_ascii=False, indent=2)
    print(f"[OK] rapport : {OUT_REPORT}")

    if len(ghosts) == 1:
        print("[VERDICT] exactement 1 ghost -> coherent avec l'ecart "
              "307,379,231 vs 307,379,232. Reparation possible "
              "(verifier d'abord la ligne via son byte_offset).")
    elif len(ghosts) == 0:
        print("[VERDICT] 0 ghost trouve alors qu'il en manque 1 -> "
              "la replique ne matche pas le script 12. STOP, comparer "
              "les deux codes avant toute reparation.")
    else:
        print(f"[VERDICT] {len(ghosts)} ghosts pour 1 EOS manquant -> "
              "INCOHERENT. STOP, ne pas reparer en l'etat.")


if __name__ == "__main__":
    main()