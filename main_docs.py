import itertools
import random
import time
from pathlib import Path
from typing import Dict, Set, Tuple, List

DOC_DIR = Path("minhash")  
DOCS = ["D1.txt", "D2.txt", "D3.txt", "D4.txt"]

M = 20011  



# Q1) k-grams

def char_kgrams(text: str, k: int) -> Set[str]:
    text = text.rstrip("\n")  
    grams = set()
    for i in range(len(text) - k + 1):
        grams.add(text[i:i + k])
    return grams


def word_kgrams(text: str, k: int) -> Set[Tuple[str, ...]]:
    words = text.strip().split()
    grams = set()
    for i in range(len(words) - k + 1):
        grams.add(tuple(words[i:i + k]))
    return grams


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)



# Q2) MinHash

def stable_token_hash(token) -> int:
    
    s = repr(token).encode("utf-8")
    
    h = 2166136261
    for byte in s:
        h ^= byte
        h = (h * 16777619) % (2 ** 32)
    return h


def generate_hash_functions(t: int, m: int, rng: random.Random) -> List[Tuple[int, int]]:
    funcs = []
    for _ in range(t):
        a = rng.randint(1, m - 1)
        b = rng.randint(0, m - 1)
        funcs.append((a, b))
    return funcs


def minhash_signature(grams: Set, hash_funcs: List[Tuple[int, int]], m: int) -> List[int]:
    sig = []
    for a, b in hash_funcs:
        min_val = None
        for g in grams:
            x = stable_token_hash(g) % m
            hv = (a * x + b) % m
            if min_val is None or hv < min_val:
                min_val = hv
        sig.append(min_val if min_val is not None else 0)
    return sig


def minhash_similarity(sig1: List[int], sig2: List[int]) -> float:
    assert len(sig1) == len(sig2)
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)



# Q3) LSH probability

def lsh_probability(s: float, r: int, b: int) -> float:
    # f(s) = 1 - (1 - s^r)^b
    return 1.0 - (1.0 - (s ** r)) ** b


def factor_pairs(n: int) -> List[Tuple[int, int]]:
    pairs = []
    for r in range(1, n + 1):
        if n % r == 0:
            b = n // r
            pairs.append((r, b))
    return pairs


def choose_rb_for_threshold(t: int, tau: float) -> Tuple[int, int]:

    best = None
    best_gap = float("inf")
    for r, b in factor_pairs(t):
        implied = (1.0 / b) ** (1.0 / r)
        gap = abs(implied - tau)
        if gap < best_gap:
            best_gap = gap
            best = (r, b)
    return best





def load_documents() -> Dict[str, str]:
    docs = {}
    for fname in DOCS:
        p = DOC_DIR / fname
        docs[fname] = p.read_text(encoding="utf-8")
    return docs


def print_pairwise_table(title: str, values: Dict[Tuple[str, str], float]):

    print(title)
    for (a, b), v in values.items():
        print(f"{a} vs {b}: {v:.6f}")


def main():
    docs = load_documents()


    pairs = list(itertools.combinations(DOCS, 2))


    grams_c2 = {d: char_kgrams(docs[d], 2) for d in DOCS}
    exact_c2 = {(a, b): jaccard(grams_c2[a], grams_c2[b]) for a, b in pairs}
    print_pairwise_table("Exact Jaccard — Character 2-grams (6 numbers)", exact_c2)


    grams_c3 = {d: char_kgrams(docs[d], 3) for d in DOCS}
    exact_c3 = {(a, b): jaccard(grams_c3[a], grams_c3[b]) for a, b in pairs}
    print_pairwise_table("Exact Jaccard — Character 3-grams (6 numbers)", exact_c3)


    grams_w2 = {d: word_kgrams(docs[d], 2) for d in DOCS}
    exact_w2 = {(a, b): jaccard(grams_w2[a], grams_w2[b]) for a, b in pairs}
    print_pairwise_table("Exact Jaccard — Word 2-grams (6 numbers)", exact_w2)


    d1, d2 = "D1.txt", "D2.txt"
    true_j = exact_c3[(d1, d2)]

    print(f"MinHash Approx — using Character 3-grams for {d1} vs {d2}")
    print(f"True Jaccard (for reference): {true_j:.6f}")

    t_values = [20, 60, 150, 300, 600]
    for t in t_values:
        rng = random.Random(12345)
        hfs = generate_hash_functions(t, M, rng)
        start = time.time()
        sig1 = minhash_signature(grams_c3[d1], hfs, M)
        sig2 = minhash_signature(grams_c3[d2], hfs, M)
        approx = minhash_similarity(sig1, sig2)
        elapsed = time.time() - start
        print(f"t={t:<3}  approx={approx:.6f}  time={elapsed:.4f}s")


    t = 160
    tau = 0.7
    r, bands = choose_rb_for_threshold(t, tau)
    print("LSH — choose r, b for t=160 and threshold τ=0.7")
    print(f"Chosen (r,b) = ({r},{bands}) since r*b = {r*bands}")
    print(f"At s=τ=0.7, f(τ) = {lsh_probability(tau, r, bands):.6f}")


    probs = {(dA, dB): lsh_probability(exact_c3[(dA, dB)], r, bands) for dA, dB in pairs}
    print_pairwise_table("LSH probability per pair (6 numbers) — using 3-grams Jaccard", probs)


if __name__ == "__main__":
    main()
