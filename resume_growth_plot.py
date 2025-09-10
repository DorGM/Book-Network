import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def cosine_topk_average_from_matrix(M, topk=10):
    M = np.asarray(M, dtype=np.float64)
    N = M.shape[0]
    topk = min(topk, max(1, N - 1))  # clamp
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Z = M / norms
    S = Z @ Z.T
    avg_sim = np.zeros(N, dtype=np.float64)
    for i in range(N):
        row = S[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -topk)[-topk:]
        vals = np.sort(row[idx])[::-1]
        avg_sim[i] = float(np.mean(vals))
    return avg_sim

def load_or_fix_avg(art_dir, topk=10):
    avg_path = os.path.join(art_dir, "avg_sim_top10.npy")
    if os.path.exists(avg_path):
        arr = np.load(avg_path)
        if np.all(np.isfinite(arr)):
            return arr

    # Fall back to recomputing from embeddings
    emb_path = os.path.join(art_dir, "emb.npy")
    if not os.path.exists(emb_path):
        raise SystemExit(f"Missing emb.npy in {art_dir}; cannot recompute.")
    emb = np.load(emb_path)
    arr = cosine_topk_average_from_matrix(emb, topk=topk)
    np.save(avg_path, arr)
    return arr

def summarize(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(np.percentile(x, 25)),
        "median": float(np.median(x)),
        "p75": float(np.percentile(x, 75)),
        "max": float(np.max(x)),
    }

def plot(series_map, out_path, bins=30, title="Average Similarity Distributions"):
    # Filter out non-finite and empty series
    series_map = {k: np.asarray(v, float)[np.isfinite(v)] for k, v in series_map.items()}
    series_map = {k: v for k, v in series_map.items() if v.size > 0}
    all_vals = np.concatenate(list(series_map.values()))
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = hi - lo if hi > lo else 1.0
    lo -= 0.02 * span; hi += 0.02 * span

    plt.figure(figsize=(8, 5))
    for label, vals in series_map.items():
        plt.hist(vals, bins=bins, range=(lo, hi), alpha=0.5, density=True, label=label)
    plt.xlabel("Average cosine similarity to top-10 neighbors")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default="artifacts/exp_growth")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10,20,50,75,100])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out-name", default="compare_avg_sim_hist.png")
    args = ap.parse_args()

    dist_map = {}
    rows = []
    for n in args.sizes:
        art_dir = os.path.join(args.artifacts_root, f"n{n}")
        avg = load_or_fix_avg(art_dir, topk=args.topk)
        dist_map[f"N={n}"] = avg
        rows.append({"size": n, **summarize(avg)})

    # Save summary and plot
    df = pd.DataFrame(rows).sort_values("size")
    df.to_csv(os.path.join(args.artifacts_root, "summary_resume.csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = "```\n" + df.to_string(index=False) + "\n```"
    with open(os.path.join(args.artifacts_root, "summary_resume.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n[SUMMARY]\n" + df.to_string(index=False))

    plot(dist_map, os.path.join(args.artifacts_root, args.out_name), bins=30,
         title="Average Similarity Distributions by Corpus Size")

if __name__ == "__main__":
    main()
