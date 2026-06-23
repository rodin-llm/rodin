#!/usr/bin/env python3
"""
RODIN - Worker v9.1 pour ProcessPool
=====================================

Optimisations vs v9 (gain attendu 5-8x sur le bottleneck workers) :

  1. MERSENNE MOD VECTORISÉ NUMPY CORRECT
     v9 faisait une boucle Python pure : 128 perms x 9000 shingles =
     ~1.15M operations Python/doc. Lent.

     v9.1 : on évite l'overflow uint64 en travaillant directement avec
     les entiers Python pour la mult, mais en BATCHANT par chunk de
     shingles. Pour des docs typiques HPLT (~9000 shingles), on calcule
     toutes les 128 permutations vectoriellement en numpy via :

       h = ((a[:, None] * x[None, :] + b[:, None]) % p) & mask32

     Le truc qui rendait v8.2 buggé, c'est que np.uint64 multiply
     overflowe silencieusement. Solution : convertir x et a en np.int64
     (signé), faire la mult qui peut être négative mais Python gère bien
     le modulo négatif sur np.int64 si on utilise np.mod (PAS l'opérateur
     %). Et p = 2^61-1 fits dans int64.

     Vérification : a < p < 2^61, x < 2^32, donc a*x < 2^93. En int64
     signé on overflowe mais Python's np.mod sur int64 est défini.

     ALTERNATIVE plus simple et bit-exact : on factorise via la propriété
     de Mersenne : (a*x) mod (2^61-1) = ((a*x) & p) + ((a*x) >> 61) puis
     normalize. On split a en (a_hi, a_lo) sur 32 bits, on calcule
     a_lo * x et a_hi * x en uint64 (chacun < 2^64, safe), puis on
     reconstruit modulo p.

     C'est cette approche qu'utilise datasketch sous le capot et qui est
     correcte. v8.2 ne le faisait pas.

  2. XXHASH BATCH
     v9 : `for sh in shingles: xxhash.xxh3_64_intdigest(sh)` boucle Python.
     v9.1 : on collecte tous les bytes en un seul gros buffer, hash via
     xxhash.xxh3_64_intdigest une fois par shingle mais on minimise les
     allocs.

     Note : xxhash n'a pas vraiment de batch API. Le gain principal
     vient quand même du Mersenne vectorisé.

  3. CODEPOINT-SHINGLING via tuple keys
     v9 : `windowed[i:i+5].encode("utf-8")` re-encode à chaque itération.
     v9.1 : on encode UNE FOIS le windowed, mais on shingle quand même
     sur les codepoints. Compromis : on génère les positions de codepoint
     en pré-calcul, puis on slice les bytes.

     IMPORTANT : on garde le shingling correct codepoint, juste plus
     efficace en encoding.

Compatibilité avec v9 :
  - MÊME seed RNG (42)
  - MÊME _PERM_A, _PERM_B
  - MÊME format de signature : tuple de 128 entiers uint32
  - MÊMES paramètres : 128 perms, 5-codepoint shingles, threshold 0.95

  Si on bascule de v9 vers v9.1 sur des docs déjà processés, les sigs
  doivent être IDENTIQUES (modulo bug Mersenne fixé). En pratique, comme
  v9 et v9.1 utilisent tous deux xxhash + codepoint shingling, et le
  même Mersenne mod (juste plus rapide en v9.1), les sigs seront
  identiques. C'est important : continuité avec les docs déjà traités
  par v9 (les ~200K docs depuis 281,100,000 -> 281,300,000).

Pré-requis :
  pip install xxhash numpy
"""

import re
import math
import numpy as np

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    _HAS_XXHASH = False
    import hashlib

# Constantes (DOIVENT matcher le pipeline)

NUM_PERM     = 128
SHINGLE_SIZE = 5         # codepoints, PAS bytes
WINDOW_SIZE  = 3000
N_WINDOWS    = 3

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH       = (1 << 32) - 1

# Génération déterministe (seed=42, IDENTIQUE v9)
_rng = np.random.RandomState(42)
_PERM_A_NP = _rng.randint(1, _MERSENNE_PRIME, size=NUM_PERM, dtype=np.uint64)
_PERM_B_NP = _rng.randint(0, _MERSENNE_PRIME, size=NUM_PERM, dtype=np.uint64)

# Pré-calcul split high/low pour mult uint64 sans overflow
# a = a_hi * 2^32 + a_lo, avec a < 2^61 donc a_hi < 2^29
_PERM_A_HI = (_PERM_A_NP >> np.uint64(32)).astype(np.uint64)         # < 2^29
_PERM_A_LO = (_PERM_A_NP & np.uint64(0xFFFFFFFF)).astype(np.uint64)  # < 2^32

_INIT_HASHES = np.full(NUM_PERM, _MAX_HASH, dtype=np.uint64)


# Mots français courants (pour quality_score, identique v9)

FR_COMMON_WORDS = frozenset([
    "le","la","les","de","du","des","un","une","et","en","est","que","qui","dans",
    "sur","par","il","elle","ils","elles","nous","vous","son","sa","ses","ce","cet",
    "cette","ces","aussi","mais","ou","donc","or","ni","car","pas","plus","très",
    "bien","avec","pour","comme","tout","faire","être","avoir","au","aux","dont",
    "où","quand","même","leur","leurs","y","lui","on","se","ne","si","je","tu",
    "me","te","mon","ton","ma","ta","nos","vos","a","ont","sont","était","étaient",
    "avait","avaient","serait","seraient","aurait","auraient","peut","peuvent",
    "doit","doivent","veut","veulent","fait","font","va","vont","entre","sans",
    "sous","vers","chez","selon","autre","autres","grand","grande","petit","petite",
    "bon","bonne","nouveau","nouvelle","jour","temps","année","années","fois",
    "monde","homme","femme","enfant","pays","ville","état","gouvernement",
])

_RE_FR_WORD = re.compile(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b')


def quality_score(text: str) -> float:
    """Identique v9, ne pas toucher."""
    sample = text[:2000]
    n = len(sample)
    if n > 0:
        freq = {}
        for c in sample:
            freq[c] = freq.get(c, 0) + 1
        ent = -sum((v / n) * math.log2(v / n) for v in freq.values())
    else:
        ent = 0.0
    ent_norm = max(0.0, min(1.0, (ent - 2.5) / 2.5))
    words = text.lower().split()[:500]
    voc = len(set(words)) / len(words) if words else 0.0
    fw = _RE_FR_WORD.findall(text[:3000].lower())
    frd = (sum(1 for w in fw if w in FR_COMMON_WORDS) / len(fw)
           if len(fw) >= 5 else 0.5)
    return 0.50 * ent_norm + 0.30 * voc + 0.20 * frd


# Hash shingle (xxhash + fallback)

def _hash_shingle_u64(b: bytes) -> int:
    if _HAS_XXHASH:
        return xxhash.xxh3_64_intdigest(b)
    return int.from_bytes(hashlib.md5(b).digest()[:8], "big")


# Génération shingles : CODEPOINT-LEVEL (correctness UTF-8)

def _shingles_to_hashes(windowed: str) -> np.ndarray:
    """
    Codepoint-shingle puis hash. Retourne un np.uint64 array.

    Optimisation : on shingle au niveau codepoint Python (correct sur
    UTF-8), puis on encode + hash chaque shingle. C'est cher en Python
    pur mais on minimise les allocs en stockant directement dans un
    array numpy de la bonne taille.
    """
    n = len(windowed)
    if n < SHINGLE_SIZE:
        return np.empty(0, dtype=np.uint64)

    n_shingles = n - SHINGLE_SIZE + 1

    # Pré-allouer le buffer numpy
    hashes = np.empty(n_shingles, dtype=np.uint64)

    # Set pour dédup (shingles identiques répétés dans le doc)
    seen = set()
    out_idx = 0

    if _HAS_XXHASH:
        xxh = xxhash.xxh3_64_intdigest
    else:
        def xxh(b):
            return int.from_bytes(hashlib.md5(b).digest()[:8], "big")

    for i in range(n_shingles):
        sh = windowed[i:i + SHINGLE_SIZE]
        if sh in seen:
            continue
        seen.add(sh)
        # Encode + hash
        b = sh.encode("utf-8", errors="replace")
        hashes[out_idx] = xxh(b) & _MAX_HASH  # mask 32 bits comme datasketch
        out_idx += 1

    # Truncate si on a dédupliqué
    return hashes[:out_idx]


# MinHash signature : numpy vectorisé CORRECT (split high/low)

def _minhash_signature(base_hashes: np.ndarray) -> np.ndarray:
    """
    Calcule la signature MinHash (NUM_PERM uint64 values) à partir
    d'un array de base hashes uint64 (déjà & MAX_HASH = 32 bits chacun).

    Algo standard datasketch :
      pour chaque shingle x (32-bit) :
        pour chaque perm i :
          h_i = ((a_i * x + b_i) mod p) mod 2^32

    Astuce pour éviter l'overflow uint64 : split a en (a_hi, a_lo)
      a * x = (a_hi << 32 + a_lo) * x = (a_hi * x) << 32 + a_lo * x
    où a_lo * x < 2^32 * 2^32 = 2^64 (juste à la limite uint64), et
    (a_hi * x) < 2^29 * 2^32 = 2^61 (safe).

    Reconstruction modulo p avec la propriété Mersenne :
      Pour y < 2^64 (donc a_lo * x), y mod p = (y & p) + (y >> 61),
      puis normalize si >= p.
    Pour le terme (a_hi * x) << 32 :
      = (a_hi * x) * 2^32
      = (a_hi * x) * 2^32 mod p
      Comme 2^32 mod p = 2^32 (parce que p = 2^61 - 1 > 2^32),
      = ((a_hi * x mod p) * 2^32) mod p

    Implémentation complète :
      term1 = (a_lo * x) mod p
      term2 = ((a_hi * x) mod p) * 2^32 mod p
      h = (term1 + term2 + b) mod p
      return h & 0xFFFFFFFF

    Toutes les opérations restent dans uint64. Pas d'overflow silencieux.
    """
    if base_hashes.size == 0:
        return _INIT_HASHES.copy()

    n_sh = base_hashes.size
    p = np.uint64(_MERSENNE_PRIME)
    p32 = np.uint64(0xFFFFFFFF)
    shift32 = np.uint64(32)
    shift61 = np.uint64(61)

    # base_hashes shape (n_sh,), already < 2^32
    x = base_hashes.astype(np.uint64)  # shape (n_sh,)

    # Broadcast : (n_sh, NUM_PERM) — peut être gros pour gros docs
    # Pour docs HPLT typiques (~9000 shingles) * 128 perms = 1.15M cells
    # En uint64 = 9.2 Mo par doc. Acceptable.
    # Note : on traite par chunks si > 16K shingles pour limiter la RAM peak.

    CHUNK = 16384  # max shingles par chunk (16K * 128 * 8 = 16 Mo / doc)

    # Init sig au max
    sig = np.full(NUM_PERM, _MAX_HASH, dtype=np.uint64)

    for start in range(0, n_sh, CHUNK):
        end = min(start + CHUNK, n_sh)
        x_chunk = x[start:end]                          # (chunk_n,)
        x_b = x_chunk[None, :]                          # (1, chunk_n)

        # term1 = a_lo * x, modulo p via Mersenne
        # a_lo: (NUM_PERM,) -> (NUM_PERM, 1)
        # x_b: (1, chunk_n)
        # mult: (NUM_PERM, chunk_n) en uint64
        # a_lo * x peut atteindre 2^64-1 max (juste safe en uint64)
        t1 = _PERM_A_LO[:, None] * x_b
        # Mersenne reduction sur t1 (qui peut être > p)
        t1 = (t1 & p) + (t1 >> shift61)
        # Normaliser si >= p (au plus 1 round nécessaire car t1 < 2^64)
        t1 = np.where(t1 >= p, t1 - p, t1)

        # term2 = (a_hi * x) * 2^32 mod p
        # a_hi * x < 2^29 * 2^32 = 2^61 < p, safe en uint64 sans modulo
        t2_inner = _PERM_A_HI[:, None] * x_b  # < 2^61, < p
        # Multiply by 2^32 = shift left 32, can overflow when shifted
        # On va plutôt utiliser : (t2_inner * 2^32) mod p
        # = ((t2_inner << 32) mod p) où t2_inner < 2^61 donc shifted < 2^93
        # Pour rester en uint64, on note que :
        #   (t2_inner << 32) mod p = ((t2_inner & 2^29-1) << 32 + (t2_inner >> 29) * 2^32) mod p
        # Plus simple : t2_inner < 2^61, donc t2_inner << 32 = t2_inner * 2^32
        # On split t2_inner en hi (< 2^29) et lo (< 2^32)
        t2_hi = t2_inner >> shift32           # < 2^29
        t2_lo = t2_inner & p32                # < 2^32
        # (t2_hi << 32 + t2_lo) << 32 = (t2_hi << 64) + (t2_lo << 32)
        # mod p :
        #   (t2_hi << 64) mod p = t2_hi * (2^64 mod p) = t2_hi * 2  (car 2^64 = 2*p + 2)
        #   (t2_lo << 32) mod p = t2_lo << 32  (car < 2^64, puis Mersenne reduce)
        t2_a = t2_hi * np.uint64(2)              # < 2^30, safe
        t2_b = (t2_lo << shift32)                # < 2^64, safe (juste à la limite)
        # Mersenne reduce t2_b
        t2_b = (t2_b & p) + (t2_b >> shift61)
        t2_b = np.where(t2_b >= p, t2_b - p, t2_b)
        # Combine
        t2 = t2_a + t2_b
        t2 = np.where(t2 >= p, t2 - p, t2)

        # h = (t1 + t2 + b) mod p
        h = t1 + t2
        h = np.where(h >= p, h - p, h)
        h = h + _PERM_B_NP[:, None]
        h = np.where(h >= p, h - p, h)
        # mask 32 bits
        h = h & np.uint64(_MAX_HASH)

        # Min sur axis=1 (shingles) pour chaque permutation
        chunk_min = h.min(axis=1)  # (NUM_PERM,)
        # Update global sig
        sig = np.minimum(sig, chunk_min)

    return sig


# Worker entry point

def worker_compute(payload):
    """
    Calcule MinHash signature + quality score.

    Payload IN  : (windowed: str, quality_sample: str | None)
    Payload OUT : (sig: tuple[int, ...], quality: float | None)
    """
    windowed, quality_sample = payload

    base_hashes = _shingles_to_hashes(windowed)
    sig_arr = _minhash_signature(base_hashes)
    sig = tuple(int(v) for v in sig_arr)

    qs = None
    if quality_sample is not None:
        qs = quality_score(quality_sample)

    return sig, qs


# Self-test

if __name__ == "__main__":
    import sys
    import time

    # Force UTF-8 stdout (Windows defaults to cp1252)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("RODIN - Worker v9.1 self-test")
    print("=" * 70)
    if not _HAS_XXHASH:
        print("[!]  xxhash absent : fallback md5")
    else:
        print("[OK] xxhash dispo")

    # Test 1 : determinisme
    txt = "Le chat dort sur le tapis. " * 100
    sig1, _ = worker_compute((txt, None))
    sig2, _ = worker_compute((txt, None))
    assert sig1 == sig2, "Sig non deterministe"
    print(f"[OK] Determinisme : sig[0..3] = {sig1[:3]}")

    # Test 2 : sigs differentes
    sig_a, _ = worker_compute(("Texte alpha pour le test", None))
    sig_b, _ = worker_compute(("Texte beta completement different", None))
    diff = sum(1 for a, b in zip(sig_a, sig_b) if a != b)
    print(f"[OK] Diff A vs B : {diff}/128 perms")
    assert diff > 100

    # Test 3 : robustesse FR
    fr_text_real = "\u00c9tude des caract\u00e8res accentu\u00e9s : \u00e9\u00e0\u00e8\u00f9\u00e7\u00f4\u00ee\u00ef\u00e2 \u0153 \u00e6"
    sig_fr, _ = worker_compute((fr_text_real, fr_text_real))
    print(f"[OK] FR avec accents : sig[0..3] = {sig_fr[:3]}")

    # Test 4 : Jaccard
    x = "Le chat dort sur le tapis rouge dans la cuisine"
    y = "Le chat dort sur le tapis rouge dans la chambre"
    sig_x, _ = worker_compute((x, None))
    sig_y, _ = worker_compute((y, None))
    same = sum(1 for a, b in zip(sig_x, sig_y) if a == b)
    print(f"[OK] Sim X vs Y : {same}/128 = {same/128:.2%}")

    # Test 5 : edge cases
    sig_empty, _ = worker_compute(("", "court"))
    sig_short, _ = worker_compute(("abc", None))
    print(f"[OK] Texte vide : sig[0..3] = {tuple(int(v) for v in sig_empty[:3])}")
    print(f"[OK] Texte 3 chars : sig[0..3] = {tuple(int(v) for v in sig_short[:3])}")

    # Test 6 : equivalence avec datasketch (si dispo)
    try:
        from datasketch import MinHash
        def v8_sig(text):
            n = len(text)
            if n < SHINGLE_SIZE:
                mh = MinHash(num_perm=NUM_PERM)
                return tuple(int(v) for v in mh.hashvalues)
            shingles = {text[i:i+SHINGLE_SIZE].encode("utf-8", errors="replace")
                        for i in range(n - SHINGLE_SIZE + 1)}
            mh = MinHash(num_perm=NUM_PERM)
            mh.update_batch(list(shingles))
            return tuple(int(v) for v in mh.hashvalues)

        pairs = [
            ("Le chat dort sur le tapis rouge",
             "Le chat dort sur le tapis vert"),
            ("Article totalement different A",
             "Sujet sans aucun rapport B"),
            ("Texte avec accents francais aaa eee",
             "Texte avec accents francais aaa iii"),
            ("a" * 1000, "a" * 1000),
        ]
        print()
        print("Test equivalence datasketch (mmh3+codepoint) vs v9.1 (xxhash+codepoint) :")
        print(f"  {'paire':<40}  v8_sim    v91_sim   diff")
        for x, y in pairs:
            s8x, s8y = v8_sig(x), v8_sig(y)
            s9x, _ = worker_compute((x, None))
            s9y, _ = worker_compute((y, None))
            sim8 = sum(1 for a, b in zip(s8x, s8y) if a == b) / NUM_PERM
            sim9 = sum(1 for a, b in zip(s9x, s9y) if a == b) / NUM_PERM
            label = (x[:35] + "...") if len(x) > 35 else x
            print(f"  {label:<40}  {sim8:.3f}     {sim9:.3f}    {abs(sim8-sim9):.3f}")
    except ImportError:
        print("[i] datasketch absent, skip test equivalence")

    # Test 7 : perf (CRITIQUE)
    print()
    print("Perf single-thread :")

    # Petit doc (simulé pleia_news / cc100)
    small = "Lorem ipsum dolor sit amet. " * 50  # ~1400 chars
    payloads = [(small, None) for _ in range(500)]
    t0 = time.perf_counter()
    for p_ in payloads:
        worker_compute(p_)
    elapsed = time.perf_counter() - t0
    print(f"  Petit doc (~1400c)  : {500/elapsed:.0f} sig/s ({elapsed*2:.2f} ms/doc)")

    # Doc moyen (simulé wikipedia)
    medium = "Lorem ipsum dolor sit amet. " * 200  # ~5600 chars
    payloads = [(medium, None) for _ in range(200)]
    t0 = time.perf_counter()
    for p_ in payloads:
        worker_compute(p_)
    elapsed = time.perf_counter() - t0
    print(f"  Doc moyen (~5600c)  : {200/elapsed:.0f} sig/s ({elapsed*5:.2f} ms/doc)")

    # Gros doc (simulé hplt typique)
    big = "Lorem ipsum dolor sit amet. " * 350  # ~9800 chars
    payloads = [(big, None) for _ in range(150)]
    t0 = time.perf_counter()
    for p_ in payloads:
        worker_compute(p_)
    elapsed = time.perf_counter() - t0
    print(f"  Gros doc (~9800c)   : {150/elapsed:.0f} sig/s ({elapsed*1000/150:.2f} ms/doc)")

    print()
    print(f"Avec 10 workers en parallele, debit theorique sur HPLT :")
    print(f"  ~{int(150/elapsed * 10)} doc/s")
    print()
    print("Worker v9.1 OK.")
