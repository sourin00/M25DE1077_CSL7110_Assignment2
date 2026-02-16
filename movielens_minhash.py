
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

DATA_PATH = Path("data/ml-100k/u.data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


P = 20011

USERS_LIMIT = None


def load_movielens_u_data(path: Path) -> Dict[int, Set[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Put MovieLens 100k at data/ml-100k/u.data")

    user_movies: Dict[int, Set[int]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        for line in f:

            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u = int(parts[0])
            m = int(parts[1])
            user_movies[u].add(m)

    if USERS_LIMIT is not None:

        kept = sorted(user_movies.keys())[:USERS_LIMIT]
        user_movies = {u: user_movies[u] for u in kept}

    return dict(user_movies)


def jaccard_sets(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def compute_exact_pairs(
    user_movies: Dict[int, Set[int]],
    threshold: float
) -> Set[Tuple[int, int]]:
    users = sorted(user_movies.keys())
    exact_pairs = set()
    total_pairs = len(users) * (len(users) - 1) // 2

    start = time.time()
    checked = 0
    for i in range(len(users)):
        ui = users[i]
        si = user_movies[ui]
        for j in range(i + 1, len(users)):
            uj = users[j]
            sj = user_movies[uj]
            sim = jaccard_sets(si, sj)
            if sim >= threshold:
                exact_pairs.add((ui, uj))
            checked += 1


        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = checked / max(elapsed, 1e-9)
            print(f"[Exact] processed {checked}/{total_pairs} pairs "
                  f"({100.0*checked/total_pairs:.1f}%) at {rate:.0f} pairs/s")

    print(f"[Exact] done. Found {len(exact_pairs)} pairs with J >= {threshold}. "
          f"Time: {time.time() - start:.2f}s")
    return exact_pairs


def save_pairs(pairs: Set[Tuple[int, int]], out_path: Path, header: str):
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header.strip() + "\n")
        for u1, u2 in sorted(pairs):
            f.write(f"{u1}\t{u2}\n")


def generate_hash_params(t: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # a in [1, P-1], b in [0, P-1]
    a = rng.integers(1, P, size=t, dtype=np.int64)
    b = rng.integers(0, P, size=t, dtype=np.int64)
    return a, b


def compute_signatures(
    user_movies_arrs: List[np.ndarray],
    t: int,
    seed: int
) -> np.ndarray:

    a, b = generate_hash_params(t, seed)
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    n_users = len(user_movies_arrs)
    sig = np.empty((n_users, t), dtype=np.int64)

    for idx, movies in enumerate(user_movies_arrs):

        vals = (a[:, None] * movies[None, :] + b[:, None]) % P
        sig[idx, :] = vals.min(axis=1)

    return sig


def minhash_pairs_above_threshold(
    signatures: np.ndarray,
    threshold: float
) -> Set[Tuple[int, int]]:

    n_users, t = signatures.shape
    need = int(np.ceil(threshold * t))

    pairs = set()
    for i in range(n_users):
        si = signatures[i]
        for j in range(i + 1, n_users):

            matches = int(np.count_nonzero(si == signatures[j]))
            if matches >= need:
                pairs.add((i, j))
    return pairs


def main():
    print(f"Loading MovieLens from: {DATA_PATH}")
    user_movies = load_movielens_u_data(DATA_PATH)
    users = sorted(user_movies.keys())
    print(f"Users loaded: {len(users)}")


    user_movie_arrs = [np.array(sorted(user_movies[u]), dtype=np.int64) for u in users]


    exact_threshold = 0.5
    exact_pairs = compute_exact_pairs(user_movies, exact_threshold)
    save_pairs(
        exact_pairs,
        RESULTS_DIR / "movielens_exact_pairs_ge_0.5.txt",
        header=f"# Exact Jaccard pairs with similarity >= {exact_threshold}"
    )


    t_list = [50, 100, 200]
    runs = 5
    approx_threshold = 0.5

    for t in t_list:
        fp_list = []
        fn_list = []


        print(f"MinHash on MovieLens — t={t}, threshold={approx_threshold}, runs={runs}")


        for run in range(runs):
            seed = 1000 + 37 * run + t
            start = time.time()

            sig = compute_signatures(user_movie_arrs, t=t, seed=seed)


            idx_pairs = minhash_pairs_above_threshold(sig, threshold=approx_threshold)
            approx_pairs = {(users[i], users[j]) for (i, j) in idx_pairs}

            fp = len(approx_pairs - exact_pairs)
            fn = len(exact_pairs - approx_pairs)

            fp_list.append(fp)
            fn_list.append(fn)

            elapsed = time.time() - start
            print(f"Run {run+1}/{runs} seed={seed} -> "
                  f"estimated_pairs={len(approx_pairs)}  FP={fp}  FN={fn}  time={elapsed:.2f}s")

            out_file = RESULTS_DIR / f"movielens_minhash_pairs_ge_0.5_t{t}_run{run+1}.txt"
            save_pairs(
                approx_pairs,
                out_file,
                header=f"# MinHash estimated pairs with similarity >= {approx_threshold}, t={t}, run={run+1}, seed={seed}"
            )

        print(f"\n[t={t}] Average FP over {runs} runs: {sum(fp_list)/runs:.2f}")
        print(f"[t={t}] Average FN over {runs} runs: {sum(fn_list)/runs:.2f}")


        with (RESULTS_DIR / "movielens_minhash_summary.txt").open("a", encoding="utf-8") as f:
            f.write(f"t={t}\tavg_FP={sum(fp_list)/runs:.2f}\tavg_FN={sum(fn_list)/runs:.2f}\n")

    print("\nCheck the 'results/' folder for output files.")


if __name__ == "__main__":
    main()
