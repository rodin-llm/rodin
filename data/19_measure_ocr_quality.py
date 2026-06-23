# 19_measure_ocr_quality.py
# Mesure CHIFFREE du bruit OCR par source, par echantillonnage via l'index
# (doc_eos_offsets.u64 + doc_sources.u8 + doc_lengths.u32). LECTURE SEULE :
# ne touche ni les shards (SHA-256 intacts) ni le JSONL. Aucun filtrage ecrit.
# Transforme "OCR incertain" en distributions + % + echantillons a l'oeil.
#
# Lecture : D:\rodin_index\ (index) + G:\data\rodin\tokenized\ (shards)
#           + G:\data\rodin\bpe\rodin.model (tokenizer)
# Ecriture: D:\rodin_index\ocr_quality_pleias.json
#           D:\rodin_index\ocr_samples_<source>.txt
#
# Usage :
#   python -u .\scripts\19_measure_ocr_quality.py --dry-run   (200 docs/source)
#   python -u .\scripts\19_measure_ocr_quality.py             (run complet)
#   python -u .\scripts\19_measure_ocr_quality.py --sources pleia_news,wikipedia
#
# Methode (voir handoff section 9) :
#   - echantillon aleatoire seede de docs par source (defaut 4000)
#   - sur un PREFIXE de --cap tokens (defaut 4096), representatif du doc
#   - panel de metriques par doc :
#       frac_suspect_words : run consonnes>=4, mot sans voyelle, alnum intra-mot,
#                            anomalie de casse, caractere hors-FR  (charabia STRUCTURE)
#       fertility          : tokens / mot vs baseline 1.54        (charabia LEXICAL,
#                            les non-mots fragmentent -> proxy OOV sans lexique)
#       frac_non_fr_chars  : caracteres hors charset FR attendu   (symboles)
#       frac_byte_fallback : tokens byte-fallback / total         (corroboration)
#   - wikipedia inclus par defaut comme CONTROLE propre (baseline relative)
#   - table keep-fraction (docs% et tokens%) a plusieurs seuils sur suspect
#   - dump des docs aux deciles pour calibration a l'oeil

import argparse
import json
import os
import string
import sys
import time

import numpy as np
import sentencepiece as spm

# ---------------------------------------------------------------- chemins
IDX_DIR = r"D:\rodin_index"
DOC_EOS = os.path.join(IDX_DIR, "doc_eos_offsets.u64")
DOC_EOS_TMP = os.path.join(IDX_DIR, "doc_eos_offsets.u64.tmp")
DOC_SOURCES = os.path.join(IDX_DIR, "doc_sources.u8")
DOC_LENGTHS = os.path.join(IDX_DIR, "doc_lengths.u32")
SOURCES_MAP = os.path.join(IDX_DIR, "sources_map.json")

SHARD_DIR = r"G:\data\rodin\tokenized"
MANIFEST = os.path.join(SHARD_DIR, "manifest.json")
SP_MODEL = r"G:\data\rodin\bpe\rodin.model"

OUT_JSON = os.path.join(IDX_DIR, "ocr_quality_pleias.json")

# ---------------------------------------------------------------- constantes
N_DOCS = 307_379_231
EXPECTED_TOKENS = 339_519_697_529
BASELINE_FERTILITY = 1.54        # global mesure en Phase 2

DEFAULT_SOURCES = ["pleia_news", "pleia_books", "wikipedia"]
CONTROL_SOURCES = {"wikipedia", "wikisource", "legifrance"}   # references propres
SUSPECT_THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
DECILES = list(range(0, 101, 10))

# charset FR attendu (lettres FR + chiffres + ponctuation usuelle + espaces)
FR_LETTERS = set("abcdefghijklmnopqrstuvwxyz"
                 "àâäçéèêëîïôöùûüÿœæ"
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "ÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸŒÆ")
VOWELS = set("aeiouyàâäéèêëîïôöùûüœæ")
FR_CONS = {c for c in FR_LETTERS if c.lower() not in VOWELS and c.islower()}
DIGITS = set("0123456789")
PUNCT = set(" \t\n\r\f\v.,;:!?…'\"«»“”‘’()[]{}<>/\\|-–—_·•@#&%‰°²³+=*~^$€£¥§©®™’")
ALLOWED = FR_LETTERS | DIGITS | PUNCT
STRIP_CHARS = string.punctuation + "«»“”‘’—–…·•‚„‹›"


# ---------------------------------------------------------------- tokenizer
def build_byte_lookup(sp):
    """Tableau bool is_byte[id] : True si la piece est un byte-fallback <0xNN>.
    Detection par motif de piece (portable, independant de la version de l'API)."""
    n = sp.get_piece_size()
    is_byte = np.zeros(n, dtype=bool)
    for tid in range(n):
        p = sp.id_to_piece(tid)
        if len(p) == 6 and p.startswith("<0x") and p.endswith(">"):
            is_byte[tid] = True
    return is_byte


# ---------------------------------------------------------------- acces shards
class Shards:
    """Lecture aleatoire des tokens d'un doc via offsets EOS (lecture seule)."""

    def __init__(self):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            man = json.load(f)
        self.files = [e["file"] for e in man["shards"]]
        counts = np.array([e["tokens"] for e in man["shards"]], dtype=np.uint64)
        self.cum = np.concatenate(([0], np.cumsum(counts)))   # len = n_shards+1
        self.maps = {}

    def _map(self, si):
        if si not in self.maps:
            path = os.path.join(SHARD_DIR, self.files[si])
            self.maps[si] = np.memmap(path, dtype=np.uint16, mode="r")
        return self.maps[si]

    def doc_prefix(self, start, end):
        """Tokens [start, end) (EOS deja exclu par l'appelant), robuste au
        chevauchement de shard (en pratique inexistant : shards auto-contenus)."""
        parts = []
        pos = start
        while pos < end:
            si = int(np.searchsorted(self.cum, pos, side="right")) - 1
            base = int(self.cum[si])
            lo = pos - base
            hi = min(end - base, int(self.cum[si + 1]) - base)
            parts.append(np.asarray(self._map(si)[lo:hi]))
            pos = base + hi
        return parts[0] if len(parts) == 1 else np.concatenate(parts)


# ---------------------------------------------------------------- analyse
def is_suspect_word(core: str) -> bool:
    """Heuristiques sans lexique. core = mot deja debarrasse de sa ponctuation
    de bord, contenant >=1 caractere alphanumerique."""
    has_alpha = any(c.isalpha() for c in core)
    if not has_alpha:
        return False  # nombre pur, etc. -> non suspect

    # caractere alphabetique hors-FR (cyrillique, grec, symboles lettres...)
    if any(c.isalpha() and c not in FR_LETTERS for c in core):
        return True

    # melange lettre/chiffre INTRA-mot (lettre adjacente a un chiffre) : "n8oob"
    for a, b in zip(core, core[1:]):
        if (a.isalpha() and b.isdigit()) or (a.isdigit() and b.isalpha()):
            return True

    low = core.lower()
    core_alpha = core.isalpha()

    # mot alphabetique >=3 sans aucune voyelle : "xnzr"
    if core_alpha and len(core) >= 3 and not any(c in VOWELS for c in low):
        return True

    # run de consonnes >=4 : "xnnr"
    run = mx = 0
    for c in low:
        if c in FR_CONS:
            run += 1
            mx = max(mx, run)
        else:
            run = 0
    if mx >= 4:
        return True

    # anomalie de casse : >=2 transitions casse au sein d'un mot alpha >=4
    if core_alpha and len(core) >= 4:
        trans = sum(1 for a, b in zip(core, core[1:]) if a.isupper() != b.isupper())
        if trans >= 2:
            return True

    return False


def analyze(prefix_ids: np.ndarray, sp, is_byte, text_cache):
    """Retourne (frac_suspect, fertility, frac_non_fr, frac_bf, text)."""
    n_tok = int(prefix_ids.shape[0])
    if n_tok == 0:
        return 0.0, 0.0, 0.0, 0.0, ""

    frac_bf = float(is_byte[prefix_ids].mean())
    text = sp.decode(prefix_ids.tolist())

    n_chars = len(text)
    non_fr = sum(1 for c in text if (c not in ALLOWED) and (not c.isspace()))
    frac_non_fr = non_fr / max(n_chars, 1)

    n_words = n_suspect = 0
    for w in text.split():
        core = w.strip(STRIP_CHARS)
        if not any(c.isalnum() for c in core):
            continue
        n_words += 1
        if is_suspect_word(core):
            n_suspect += 1

    fertility = n_tok / max(n_words, 1)
    frac_suspect = n_suspect / max(n_words, 1)
    return frac_suspect, fertility, frac_non_fr, frac_bf, text


def noise_rank(suspect, fertility, non_fr, bf):
    """Cle de TRI uniquement (pour ordonner les echantillons a l'oeil).
    PAS la metrique de decision (ce sont les distributions par metrique qui
    decident). Combinateur OR : un doc est bruite si UN signal est haut."""
    fert_excess = min(max((fertility - BASELINE_FERTILITY) / 1.5, 0.0), 1.0)
    return max(suspect, fert_excess, non_fr, bf)


# ---------------------------------------------------------------- main
def pct_block(arr):
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": round(float(a.mean()), 4),
        "p50": round(float(np.percentile(a, 50)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p90": round(float(np.percentile(a, 90)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "p99": round(float(np.percentile(a, 99)), 4),
        "max": round(float(a.max()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                    help="liste separee par virgules (defaut pleia_news,pleia_books,wikipedia)")
    ap.add_argument("--sample", type=int, default=4000,
                    help="docs echantillonnes par source (defaut 4000)")
    ap.add_argument("--cap", type=int, default=4096,
                    help="prefixe analyse en tokens par doc (defaut 4096)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true",
                    help="200 docs/source, sorties suffixees .dryrun")
    ap.add_argument("--clean", action="store_true",
                    help="autorise l'ecrasement des sorties existantes")
    args = ap.parse_args()

    sample_n = 200 if args.dry_run else args.sample
    suffix = ".dryrun" if args.dry_run else ""
    out_json = OUT_JSON + suffix

    if os.path.exists(out_json) and not args.dry_run and not args.clean:
        print(f"[ERREUR] {out_json} existe deja. Relancer avec --clean.")
        sys.exit(1)

    # --- tokenizer ---
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    is_byte = build_byte_lookup(sp)
    print(f"[INFO] tokenizer : {sp.get_piece_size():,} pieces | "
          f"{int(is_byte.sum())} pieces byte-fallback")

    # --- index (memmap pour les gros, fromfile pour doc_sources) ---
    eos_path = DOC_EOS if os.path.exists(DOC_EOS) else DOC_EOS_TMP
    if not os.path.exists(eos_path):
        print(f"[ERREUR] offsets EOS introuvables ({DOC_EOS}).")
        sys.exit(1)
    if eos_path == DOC_EOS_TMP:
        print("[AVERTISSEMENT] index EOS non finalise (.tmp) : run script 18 "
              "d'abord pour graver l'index. Lecture du .tmp en attendant.")

    eos = np.memmap(eos_path, dtype=np.uint64, mode="r")
    lengths = np.memmap(DOC_LENGTHS, dtype=np.uint32, mode="r")
    doc_src = np.fromfile(DOC_SOURCES, dtype=np.uint8)
    for name, arr, exp in (("doc_eos", eos, N_DOCS),
                           ("doc_lengths", lengths, N_DOCS),
                           ("doc_sources", doc_src, N_DOCS)):
        if arr.shape[0] != exp:
            print(f"[ERREUR] {name} : {arr.shape[0]:,} != {exp:,} attendus")
            sys.exit(1)
    if int(eos[-1]) + 1 != EXPECTED_TOKENS:
        print(f"[ERREUR] coherence tokens : dernier EOS+1 ({int(eos[-1]) + 1:,}) "
              f"!= {EXPECTED_TOKENS:,}")
        sys.exit(1)
    print(f"[OK] index charge : {N_DOCS:,} docs, {EXPECTED_TOKENS:,} tokens")

    with open(SOURCES_MAP, "r", encoding="utf-8") as f:
        smap = json.load(f)["sources"]
    name_to_code = {k: v["code"] for k, v in smap.items()}

    shards = Shards()
    rng = np.random.default_rng(args.seed)
    wanted = [s.strip() for s in args.sources.split(",") if s.strip()]

    report = {
        "input_index": IDX_DIR,
        "tokenizer": SP_MODEL,
        "seed": args.seed,
        "sample_per_source": sample_n,
        "analyze_token_cap": args.cap,
        "baseline_fertility": BASELINE_FERTILITY,
        "dry_run": args.dry_run,
        "sources": {},
    }

    t0 = time.time()
    for name in wanted:
        if name not in name_to_code:
            print(f"[SKIP] source inconnue : '{name}'")
            continue
        code = name_to_code[name]
        idx_all = np.flatnonzero(doc_src == code)
        if idx_all.size == 0:
            print(f"[SKIP] aucune ligne pour '{name}' (code {code})")
            continue

        n = min(sample_n, idx_all.size)
        chosen = rng.choice(idx_all, size=n, replace=False)
        chosen.sort()
        role = "controle" if name in CONTROL_SOURCES else "cible"
        print(f"\n[SOURCE] {name} (code {code}, {role}) : "
              f"{idx_all.size:,} docs, echantillon {n:,}")

        m_suspect = np.empty(n, np.float64)
        m_fert = np.empty(n, np.float64)
        m_nonfr = np.empty(n, np.float64)
        m_bf = np.empty(n, np.float64)
        m_len = np.empty(n, np.float64)     # longueur FULL doc (ponderation tokens)
        m_rank = np.empty(n, np.float64)
        samples = []                        # (rank, dict) pour dump deciles

        tlog = time.time()
        for k, i in enumerate(chosen):
            i = int(i)
            start = 0 if i == 0 else int(eos[i - 1]) + 1
            end = int(eos[i])                          # EOS exclu
            end = min(end, start + args.cap)
            prefix = shards.doc_prefix(start, end)
            su, fe, nf, bf, text = analyze(prefix, sp, is_byte, None)
            r = noise_rank(su, fe, nf, bf)
            m_suspect[k] = su
            m_fert[k] = fe
            m_nonfr[k] = nf
            m_bf[k] = bf
            m_len[k] = float(lengths[i])
            m_rank[k] = r
            samples.append((r, {
                "doc": i, "full_tokens": int(lengths[i]),
                "frac_suspect": round(su, 4), "fertility": round(fe, 3),
                "frac_non_fr": round(nf, 4), "frac_byte_fallback": round(bf, 4),
                "excerpt": text[:600].replace("\n", " "),
            }))
            if time.time() - tlog >= 30:
                tlog = time.time()
                print(f"  [{k + 1:>6,}/{n:,}] {(k + 1) / (time.time() - t0):.0f} docs/s")

        # --- table keep-fraction sur frac_suspect_words ---
        total_w = m_len.sum()
        keep = {}
        for thr in SUSPECT_THRESHOLDS:
            mask = m_suspect < thr
            docs_pct = 100.0 * mask.mean()
            tok_pct = 100.0 * (m_len[mask].sum() / total_w) if total_w else 0.0
            keep[f"{thr:.2f}"] = {"docs_pct": round(docs_pct, 2),
                                  "tokens_pct": round(tok_pct, 2)}

        # --- forme de la distribution (indices de bimodalite) ---
        frac_clean = 100.0 * (m_suspect < 0.05).mean()
        frac_dirty = 100.0 * (m_suspect > 0.20).mean()

        report["sources"][name] = {
            "code": code, "role": role,
            "total_docs": int(idx_all.size), "sampled": int(n),
            "metrics": {
                "frac_suspect_words": pct_block(m_suspect),
                "fertility": pct_block(m_fert),
                "frac_non_fr_chars": pct_block(m_nonfr),
                "frac_byte_fallback": pct_block(m_bf),
            },
            "shape": {"pct_docs_suspect_lt_0.05": round(frac_clean, 2),
                      "pct_docs_suspect_gt_0.20": round(frac_dirty, 2)},
            "keep_fraction_vs_suspect_threshold": keep,
        }

        # --- console : resume compact ---
        ps = report["sources"][name]["metrics"]
        print(f"  suspect_words  p50={ps['frac_suspect_words']['p50']:.3f} "
              f"p90={ps['frac_suspect_words']['p90']:.3f} "
              f"p99={ps['frac_suspect_words']['p99']:.3f}")
        print(f"  fertility      p50={ps['fertility']['p50']:.2f} "
              f"p90={ps['fertility']['p90']:.2f} "
              f"p99={ps['fertility']['p99']:.2f}  (baseline {BASELINE_FERTILITY})")
        print(f"  non_fr_chars   p50={ps['frac_non_fr_chars']['p50']:.4f} "
              f"p99={ps['frac_non_fr_chars']['p99']:.4f}")
        print(f"  byte_fallback  p50={ps['frac_byte_fallback']['p50']:.4f} "
              f"p99={ps['frac_byte_fallback']['p99']:.4f}")
        print(f"  forme : {frac_clean:.1f}% docs propres (<0.05) | "
              f"{frac_dirty:.1f}% docs sales (>0.20)")
        print(f"  keep@suspect<0.10 : {keep['0.10']['docs_pct']:.1f}% docs / "
              f"{keep['0.10']['tokens_pct']:.1f}% tokens")

        # --- dump des deciles pour calibration a l'oeil ---
        samples.sort(key=lambda x: x[0])
        out_txt = os.path.join(IDX_DIR, f"ocr_samples_{name}.txt") + suffix
        with open(out_txt, "w", encoding="utf-8") as fh:
            fh.write(f"# Echantillons OCR '{name}' ordonnes par bruit croissant\n")
            fh.write(f"# {n} docs echantillonnes, seed {args.seed}, cap {args.cap} tokens\n")
            fh.write(f"# decile 0 = plus propre, decile 100 = plus bruite\n\n")
            for d in DECILES:
                pos = min(int(d / 100.0 * (n - 1)), n - 1)
                r, s = samples[pos]
                fh.write(f"{'=' * 78}\n")
                fh.write(f"DECILE {d:3d}  rank={r:.3f}  doc={s['doc']}  "
                         f"full_tokens={s['full_tokens']}\n")
                fh.write(f"  suspect={s['frac_suspect']}  fertility={s['fertility']}  "
                         f"non_fr={s['frac_non_fr']}  byte_fb={s['frac_byte_fallback']}\n")
                fh.write(f"{'-' * 78}\n")
                fh.write(s["excerpt"] + "\n\n")
        print(f"  [OK] echantillons a l'oeil -> {out_txt}")

    # --- ecriture rapport ---
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] rapport -> {out_json}")
    print(f"[FIN] {time.time() - t0:.0f} s")

    # --- guidance de lecture ---
    print("\n[LECTURE] comparer chaque source CIBLE au CONTROLE (wikipedia) :")
    print("  - forme bimodale (bcp propres + bcp sales) -> filtrer par doc")
    print("    (masque keep/drop au niveau dataloader, JAMAIS toucher les shards)")
    print("  - propre quasi partout              -> garder, ponderer normalement")
    print("  - sale quasi partout                -> sous-ponderer lourd / drop")
    print("  Tu as les tokens en rab (run 30-35B << sources) : seuil AGRESSIF OK.")


if __name__ == "__main__":
    main()
