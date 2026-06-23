#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
format_chatml.py — Convertit le JSONL brut (instruction/reponse) en ChatML pour le SFT.

Entree  : sft_raw.jsonl  ->  {"instruction":..., "reponse":..., "registre":...}
Sortie  : sft_chatml.jsonl -> {"text": "<|im_start|>...", "registre":...}

Format ChatML produit (sans system par defaut, ajoutable via --system) :
  <|im_start|>user
  {instruction}<|im_end|>
  <|im_start|>assistant
  {reponse}<|im_end|>

Le script fait aussi un dernier nettoyage/dedup et un split train/val.

Usage :
  python format_chatml.py --in sft_raw.jsonl --out-dir . --val-ratio 0.02
"""

import argparse
import hashlib
import json
import os
import random
import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

def norm_hash(s):
    return hashlib.sha256(re.sub(r"\s+", " ", s.lower()).strip().encode("utf-8")).hexdigest()

def to_chatml(instruction, reponse, system=None):
    parts = []
    if system:
        parts.append(f"{IM_START}system\n{system}{IM_END}\n")
    parts.append(f"{IM_START}user\n{instruction}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n{reponse}{IM_END}")
    return "".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--system", default=None, help="prompt systeme optionnel a injecter")
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    rows = []
    skipped = 0

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue
            instr = (obj.get("instruction") or "").strip()
            rep = (obj.get("reponse") or obj.get("réponse") or "").strip()
            if not instr or not rep:
                skipped += 1
                continue
            h = norm_hash(instr)
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
            rows.append({
                "text": to_chatml(instr, rep, args.system),
                "registre": obj.get("registre", "?"),
            })

    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val = rows[:n_val]
    train = rows[n_val:]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "sft_chatml_train.jsonl")
    val_path = os.path.join(args.out_dir, "sft_chatml_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # repartition par registre
    from collections import Counter
    rep_counts = Counter(r["registre"] for r in rows)

    print(f"[format] total uniques={len(rows)} | train={len(train)} | val={len(val)} | ignores={skipped}")
    print("[format] repartition par registre :")
    for reg, c in rep_counts.most_common():
        print(f"   {reg:<18} {c}")
    print(f"[format] ecrit : {train_path}")
    print(f"[format] ecrit : {val_path}")

if __name__ == "__main__":
    main()
