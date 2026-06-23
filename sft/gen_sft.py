#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_sft.py — Generation de dataset SFT francais pour RODIN-1B-Instruct.

Principe : mix graines + variantes.
  - On lit des graines (registre / theme / tache / poids) depuis seeds.jsonl.
  - Pour chaque graine, on demande a un modele instruct local (via Ollama)
    de produire un BATCH de variantes : couples (instruction, reponse) en FR.
  - Sortie : JSONL une ligne par exemple : {"instruction": ..., "reponse": ..., "registre": ...}

Robustesse :
  - Reprise : relit la sortie existante, deduplique par hash d'instruction,
    reprend la generation la ou elle en etait (par graine).
  - Ecriture atomique ligne par ligne (flush + fsync) -> coupure sans perte.
  - Parsing JSON defensif (tolere fences ```json, texte autour).
  - Garde-fous qualite (longueurs, vide, balises <think>, doublons).

Usage minimal :
  python gen_sft.py --seeds seeds.jsonl --out sft_raw.jsonl --target 5000

Le script tourne en boucle jusqu'a atteindre --target exemples valides,
en repartissant la generation selon les poids des graines.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error

# --------------------------------------------------------------------------- #
# Configuration par defaut
# --------------------------------------------------------------------------- #

OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434/api/chat"
MODEL_DEFAULT = "qwen3.5:9b-q8_0"

# Bornes qualite (en caracteres). Ajustables via CLI si besoin.
MIN_INSTR_CHARS = 8
MAX_INSTR_CHARS = 800
MIN_REP_CHARS = 20
MAX_REP_CHARS = 4000

# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = (
    "Tu es un generateur de donnees d'entrainement en francais. "
    "Tu produis des paires (instruction, reponse) de haute qualite, "
    "entierement en francais correct et naturel. "
    "Tu reponds UNIQUEMENT par du JSON valide, sans aucun texte autour, "
    "sans balises de reflexion, sans commentaire."
)

def build_user_prompt(registre, theme, tache, n_variants):
    """Construit la consigne pour generer n_variants couples instruction/reponse."""
    return f"""Genere {n_variants} exemples d'entrainement DIFFERENTS et VARIES en francais.

Registre vise : {registre}
Theme : {theme}
Type de tache : {tache}

Contraintes IMPORTANTES :
- Chaque exemple est un couple : une "instruction" (ce qu'un utilisateur demanderait reellement) et une "reponse" (une reponse de qualite, utile, correcte et naturelle en francais).
- Les instructions doivent etre REALISTES, comme ecrites par un vrai utilisateur francophone. Varie les formulations, les longueurs, les angles.
- Quand la tache implique un texte a traiter (resumer, reecrire, corriger), INCLUS le texte directement dans l'instruction.
- Les reponses sont entierement en francais, sans anglais, sans balises, sans meta-commentaire.
- INTERDIT ABSOLUMENT : ne montre aucun raisonnement, aucune hesitation, aucune correction dans la reponse. Pas de "en realite", "verifions", "non,", "reponse corrigee", "attendez", "?". La reponse est DIRECTE, ASSUREE et FINALE.
- Si tu n'es pas certain d'un fait precis, NE GENERE PAS cet exemple : choisis un sujet plus simple et indiscutable. Mieux vaut un fait trivial sur et net qu'un fait pointu approximatif.
- Ne numerote pas, ne mets pas de titre. Pas de "Exemple 1".
- Varie fortement les sujets concrets a l'interieur du theme.

Format de sortie STRICT : un objet JSON unique avec une cle "exemples" contenant une liste d'objets, chacun avec exactement les cles "instruction" et "reponse".

Exemple de structure (ne recopie pas son contenu) :
{{"exemples": [{{"instruction": "...", "reponse": "..."}}, {{"instruction": "...", "reponse": "..."}}]}}

Reponds maintenant avec UNIQUEMENT ce JSON."""

# --------------------------------------------------------------------------- #
# Appel Ollama
# --------------------------------------------------------------------------- #

def call_ollama(url, model, system_prompt, user_prompt, temperature, timeout):
    """Appel non-stream a l'API /api/chat d'Ollama. Retourne le texte du message."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # desactive le mode reflexion (Qwen3.x)
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    # /api/chat non-stream : {"message": {"content": "..."}, ...}
    return obj.get("message", {}).get("content", "")

# --------------------------------------------------------------------------- #
# Parsing defensif du JSON renvoye par le modele
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

def extract_json_block(text):
    """Tente d'extraire un objet JSON depuis une sortie potentiellement bruitee."""
    if not text:
        return None
    # vire d'eventuelles balises de reflexion
    text = _THINK_RE.sub("", text).strip()
    # 1) tente un fence ```json ... ```
    m = _FENCE_RE.search(text)
    if m:
        candidate = m.group(1).strip()
        parsed = _try_load(candidate)
        if parsed is not None:
            return parsed
    # 2) tente le texte entier
    parsed = _try_load(text)
    if parsed is not None:
        return parsed
    # 3) tente d'isoler du premier { au dernier }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        parsed = _try_load(candidate)
        if parsed is not None:
            return parsed
    return None

def _try_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None

def extract_examples(parsed):
    """Normalise differentes formes possibles vers une liste de dicts {instruction, reponse}."""
    if parsed is None:
        return []
    # forme attendue : {"exemples": [...]}
    if isinstance(parsed, dict):
        for key in ("exemples", "examples", "data", "items"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # parfois le modele renvoie directement {"instruction":..., "reponse":...}
        if "instruction" in parsed and ("reponse" in parsed or "réponse" in parsed):
            return [parsed]
        return []
    # forme : liste directe
    if isinstance(parsed, list):
        return parsed
    return []

# --------------------------------------------------------------------------- #
# Validation qualite
# --------------------------------------------------------------------------- #

def get_field(ex, *names):
    for n in names:
        if n in ex and isinstance(ex[n], str):
            return ex[n].strip()
    return ""

def valid_example(instr, rep):
    if not instr or not rep:
        return False
    if not (MIN_INSTR_CHARS <= len(instr) <= MAX_INSTR_CHARS):
        return False
    if not (MIN_REP_CHARS <= len(rep) <= MAX_REP_CHARS):
        return False
    # rejette les restes de balises / meta
    low = (instr + " " + rep).lower()
    if "<think>" in low or "</think>" in low:
        return False
    if low.startswith("```") or rep.startswith("{") and rep.endswith("}"):
        return False
    # rejette si la reponse est juste un echo de l'instruction
    if instr == rep:
        return False
    # rejette le raisonnement-qui-fuit / hesitation dans la REPONSE
    rep_low = rep.lower()
    delib_markers = (
        "en réalité", "en realite", "vérifions", "verifions", "réponse corrigée",
        "reponse corrigee", "attendez", "attends,", "non, ", "? non", "hmm",
        "laissez-moi", "laisse-moi vérifier", "je vais vérifier", "corrigeons",
        "réfléchissons", "reflechissons", "voyons voir", "il me semble que non",
    )
    if any(m in rep_low for m in delib_markers):
        return False
    # rejette les aveux d'incertitude / invention de terme (le modele se trahit)
    uncertainty_markers = (
        "le terme exact", "semble être une confusion", "semble etre une confusion",
        "n'est pas un terme juridique", "n'est pas un terme officiel",
        "est impropre", "abus de langage", "souvent confondu", "mal placé",
        "mal place", "je ne suis pas sûr", "je ne suis pas sur",
    )
    if any(m in rep_low for m in uncertainty_markers):
        return False
    # une reponse factuelle qui contient un "?" est suspecte (auto-questionnement)
    if rep.count("?") >= 2:
        return False
    return True

def norm_hash(instr):
    key = re.sub(r"\s+", " ", instr.lower()).strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# --------------------------------------------------------------------------- #
# Gestion graines / poids
# --------------------------------------------------------------------------- #

def load_seeds(path):
    seeds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("poids", 1)
            seeds.append(obj)
    if not seeds:
        sys.exit("Aucune graine chargee.")
    return seeds

def weighted_pick(seeds, rng):
    total = sum(s["poids"] for s in seeds)
    r = rng.uniform(0, total)
    acc = 0
    for s in seeds:
        acc += s["poids"]
        if r <= acc:
            return s
    return seeds[-1]

# --------------------------------------------------------------------------- #
# Reprise : relecture de la sortie existante
# --------------------------------------------------------------------------- #

def load_existing(path):
    seen = set()
    count = 0
    if not os.path.exists(path):
        return seen, count
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            instr = obj.get("instruction", "")
            if instr:
                seen.add(norm_hash(instr))
                count += 1
    return seen, count

# --------------------------------------------------------------------------- #
# Boucle principale
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Generation dataset SFT FR via Ollama.")
    ap.add_argument("--seeds", required=True, help="fichier seeds.jsonl")
    ap.add_argument("--out", required=True, help="sortie JSONL (cumulative, reprise auto)")
    ap.add_argument("--target", type=int, default=5000, help="nb d'exemples valides vises")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--url", default=OLLAMA_URL_DEFAULT)
    ap.add_argument("--variants", type=int, default=6, help="variantes demandees par appel")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--timeout", type=int, default=300, help="timeout HTTP par appel (s)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-fails", type=int, default=50, help="echecs consecutifs avant abandon")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = load_seeds(args.seeds)
    seen, existing = load_existing(args.out)
    print(f"[init] graines={len(seeds)} | deja presents={existing} | cible={args.target}", flush=True)

    if existing >= args.target:
        print("[init] cible deja atteinte. Rien a faire.", flush=True)
        return

    out_f = open(args.out, "a", encoding="utf-8")
    kept = existing
    consec_fails = 0
    t0 = time.time()
    calls = 0

    try:
        while kept < args.target:
            seed = weighted_pick(seeds, rng)
            user_prompt = build_user_prompt(
                seed["registre"], seed["theme"], seed["tache"], args.variants
            )
            calls += 1
            try:
                raw = call_ollama(
                    args.url, args.model, SYSTEM_PROMPT, user_prompt,
                    args.temperature, args.timeout
                )
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                consec_fails += 1
                print(f"[warn] appel echoue ({e}). echecs consecutifs={consec_fails}", flush=True)
                if consec_fails >= args.max_fails:
                    print("[stop] trop d'echecs consecutifs. Ollama est-il lance ?", flush=True)
                    break
                time.sleep(2)
                continue

            parsed = extract_json_block(raw)
            examples = extract_examples(parsed)

            added_this_call = 0
            for ex in examples:
                if not isinstance(ex, dict):
                    continue
                instr = get_field(ex, "instruction", "instr", "prompt")
                rep = get_field(ex, "reponse", "réponse", "response", "answer")
                if not valid_example(instr, rep):
                    continue
                h = norm_hash(instr)
                if h in seen:
                    continue
                seen.add(h)
                rec = {"instruction": instr, "reponse": rep, "registre": seed["registre"]}
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                os.fsync(out_f.fileno())
                kept += 1
                added_this_call += 1
                if kept >= args.target:
                    break

            if added_this_call == 0:
                consec_fails += 1
                if consec_fails >= args.max_fails:
                    print("[stop] trop d'appels sans exemple valide.", flush=True)
                    break
            else:
                consec_fails = 0

            if calls % 5 == 0 or added_this_call:
                rate = kept / max(1e-9, (time.time() - t0)) * 60.0
                print(
                    f"[gen] valides={kept}/{args.target} "
                    f"(+{added_this_call}) | appels={calls} | "
                    f"~{rate:.1f}/min | dernier registre={seed['registre']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[interrupt] arret demande. Sortie sauvegardee, reprise possible.", flush=True)
    finally:
        out_f.close()

    dt = time.time() - t0
    print(f"[done] total valides={kept} | appels={calls} | duree={dt/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
