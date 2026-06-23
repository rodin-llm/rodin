# verif_ligne_fantome.py — confirme le nombre d'objets dans la ligne fusionnee
import orjson

IN_PATH = r"G:\data\rodin\deduped\merged_final.jsonl"
LINE_OFFSETS = r"D:\rodin_index\line_offsets.u64"
GHOST = 219_077_862

import numpy as np
off = np.fromfile(LINE_OFFSETS, dtype=np.uint64)
with open(IN_PATH, "rb") as fh:
    fh.seek(int(off[GHOST]))
    raw = fh.read(int(off[GHOST + 1]) - int(off[GHOST]))

raw = raw.strip()
print(f"longueur : {len(raw):,} octets")

# decoupe gloutonne par raw_decode (json stdlib) pour compter les objets
import json
dec = json.JSONDecoder()
idx, n, objs = 0, 0, []
s = raw.decode("utf-8", errors="strict")
while idx < len(s):
    while idx < len(s) and s[idx] in " \t\r\n":
        idx += 1
    if idx >= len(s):
        break
    obj, end = dec.raw_decode(s, idx)
    objs.append(obj)
    n += 1
    idx = end
print(f"objets JSON concatenes dans cette ligne : {n}")
for k, o in enumerate(objs):
    t = o.get("text", "")
    print(f"  obj {k}: source={o.get('source')} id={o.get('id')} "
          f"text[:60]={t[:60]!r} len_text={len(t)}")