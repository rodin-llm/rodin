#!/usr/bin/env python3
import json, sys, re, random
from collections import Counter

path = sys.argv[1]
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
print(f"total exemples : {len(rows)}\n")

# repartition par registre
c = Counter(r["registre"] for r in rows)
print("=== repartition par registre ===")
for reg, n in c.most_common():
    print(f"  {reg:<16} {n:>5}  ({100*n/len(rows):.1f}%)")

# stats longueur par registre
print("\n=== longueur moyenne REPONSE (caracteres) par registre ===")
by_reg = {}
for r in rows:
    by_reg.setdefault(r["registre"], []).append(len(r["reponse"]))
for reg in sorted(by_reg, key=lambda k: -sum(by_reg[k])/len(by_reg[k])):
    L = by_reg[reg]
    print(f"  {reg:<16} moy={sum(L)//len(L):>4}  min={min(L):>4}  max={max(L):>5}")

# detection de gras potentiel : reponses factuelles longues
print("\n=== alertes potentielles ===")
fact_long = [r for r in rows if r["registre"]=="qa_factuelle" and len(r["reponse"])>300]
print(f"  qa_factuelle avec reponse >300 car (gras possible) : {len(fact_long)}")

# anglais residuel grossier
en_markers = (" the ", " and ", " is ", " you ", " your ", "here is", "for example")
eng = [r for r in rows if any(m in r["reponse"].lower() for m in en_markers)]
print(f"  reponses avec anglais residuel possible : {len(eng)}")

# reponses suspectes courtes
short = [r for r in rows if len(r["reponse"])<25]
print(f"  reponses tres courtes (<25 car) : {len(short)}")

# doublons d'instruction (devrait etre 0)
instrs = [re.sub(r'\s+',' ',r["instruction"].lower()).strip() for r in rows]
dups = len(instrs) - len(set(instrs))
print(f"  doublons d'instruction restants : {dups}")
