
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

    user_movies = defaultdict(set)
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
    else:
        user_movies = dict(user_movies)

    return user_movies


def jaccard_sets(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


def compute_exact_pairs_for_thresholds(
    user_movies: Dict[int, Set[int]],
    thresholds: List[float]
) -> Dict[float, Set[Tuple[int, int]]]:
    users = sorted(user_movies.keys())
    out = {thr: set() for thr in thresholds}

    total_pairs = len(users) * (len(users) - 1) // 2
    start = time.time()
    checked = 0


    thr_sorted = sorted(thresholds)

    for i in range(len(users)):
        ui = users[i]
        si = user_movies[ui]
        for j in range(i + 1, len(users)):
            uj = users[j]
            sj = user_movies[uj]
            sim = jaccard_sets(si, sj)

            for thr in thr_sorted:
                if sim >= thr:
                    out[thr].add((ui, uj))
                else:

                    break

            checked += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = checked / max(elapsed, 1e-9)
            print(f"[Exact] processed {checked}/{total_pairs} pairs "
                  f"({100.0*checked/total_pairs:.1f}%) at {rate:.0f} pairs/s")

    print(f"[Exact] done in {time.time() - start:.2f}s")
    for thr in thr_sorted:
        print(f"[Exact] pairs with J >= {thr}: {len(out[thr])}")
    return out


def save_pairs(pairs: Set[Tuple[int, int]], out_path: Path, header: str):
    with out_path.open("w", encoding="utf-8") as f:
        f.write(header.strip() + "\n")
        for u1, u2 in sorted(pairs):
            f.write(f"{u1}\t{u2}\n")


def generate_hash_params(t: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, P, size=t, dtype=np.int64)
    b = rng.integers(0, P, size=t, dtype=np.int64)
    return a, b


def compute_signatures(user_movie_arrs: List[np.ndarray], t: int, seed: int) -> np.ndarray:
    a, b = generate_hash_params(t, seed)
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    n_users = len(user_movie_arrs)
    sig = np.empty((n_users, t), dtype=np.int64)

    for idx, movies in enumerate(user_movie_arrs):
        vals = (a[:, None] * movies[None, :] + b[:, None]) % P
        sig[idx, :] = vals.min(axis=1)
    return sig


def est_similarity_from_signatures(sig: np.ndarray, i: int, j: int) -> float:
    return float(np.count_nonzero(sig[i] == sig[j]) / sig.shape[1])


def lsh_candidate_pairs(sig: np.ndarray, r: int, b: int) -> Set[Tuple[int, int]]:

    n_users, t = sig.shape
    assert r * b == t, f"Expected r*b=t but got r={r}, b={b}, t={t}"

    candidates: Set[Tuple[int, int]] = set()

    for band_idx in range(b):
        start = band_idx * r
        end = start + r

        buckets = defaultdict(list)

        for u in range(n_users):
            key = sig[u, start:end].tobytes()
            buckets[key].append(u)


        for users_in_bucket in buckets.values():
            if len(users_in_bucket) < 2:
                continue
            users_in_bucket.sort()
            for i in range(len(users_in_bucket)):
                ui = users_in_bucket[i]
                for j in range(i + 1, len(users_in_bucket)):
                    uj = users_in_bucket[j]
                    candidates.add((ui, uj))

    return candidates


def lsh_find_pairs_above_threshold(
    sig: np.ndarray,
    users: List[int],
    r: int,
    b: int,
    threshold: float
) -> Set[Tuple[int, int]]:
    cand_idx_pairs = lsh_candidate_pairs(sig, r, b)
    out_pairs = set()
    for i, j in cand_idx_pairs:
        sim = est_similarity_from_signatures(sig, i, j)
        if sim >= threshold:
            out_pairs.add((users[i], users[j]))
    return out_pairs


def main():
    print(f"Loading MovieLens from: {DATA_PATH}")
    user_movies = load_movielens_u_data(DATA_PATH)
    users = sorted(user_movies.keys())
    print(f"Users loaded: {len(users)}")

    user_movie_arrs = [np.array(sorted(user_movies[u]), dtype=np.int64) for u in users]


    gt_thresholds = [0.6, 0.8]
    gt_sets = compute_exact_pairs_for_thresholds(user_movies, gt_thresholds)


    configs = [
        (50, 5, 10),
        (100, 5, 20),
        (200, 5, 40),
        (200, 10, 20),
    ]

    runs = 5

    for target_thr in [0.6, 0.8]:
        gt = gt_sets[target_thr]

        print(f"LSH on MovieLens — target similarity >= {target_thr} (average over {runs} runs)")

        for (t, r, b) in configs:
            fp_list = []
            fn_list = []


            print(f"Config: t={t}, r={r}, b={b} (r*b={r*b})")


            for run in range(runs):
                seed = 5000 + 41 * run + t + 7 * r + 13 * b
                start = time.time()

                sig = compute_signatures(user_movie_arrs, t=t, seed=seed)
                est_pairs = lsh_find_pairs_above_threshold(sig, users, r=r, b=b, threshold=target_thr)

                fp = len(est_pairs - gt)
                fn = len(gt - est_pairs)

                fp_list.append(fp)
                fn_list.append(fn)

                elapsed = time.time() - start
                print(f"Run {run+1}/{runs} seed={seed} -> "
                      f"candidates+filtered_pairs={len(est_pairs)}  FP={fp}  FN={fn}  time={elapsed:.2f}s")

                out_file = RESULTS_DIR / f"movielens_lsh_pairs_ge_{target_thr}_t{t}_r{r}_b{b}_run{run+1}.txt"
                save_pairs(
                    est_pairs,
                    out_file,
                    header=f"# LSH estimated pairs with similarity >= {target_thr}, t={t}, r={r}, b={b}, run={run+1}, seed={seed}"
                )

            avg_fp = sum(fp_list) / runs
            avg_fn = sum(fn_list) / runs
            print(f"AVG over {runs} runs -> FP={avg_fp:.2f}, FN={avg_fn:.2f}")

            with (RESULTS_DIR / "movielens_lsh_summary.txt").open("a", encoding="utf-8") as f:
                f.write(f"thr={target_thr}\tt={t}\tr={r}\tb={b}\tavg_FP={avg_fp:.2f}\tavg_FN={avg_fn:.2f}\n")

    print("\nDone. Check 'results/' for pair lists and summaries.")


if __name__ == "__main__":
    main()
