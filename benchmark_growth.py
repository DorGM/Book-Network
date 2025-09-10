import os
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import your pipeline entrypoints
import BigData_Project as BP

SIZES_DEFAULT = [10, 20, 50, 75, 100]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def prepare_base100(base_books_dir, base_meta_csv, limit=100, force=False):
    """Download StorySet once (deterministic shuffle inside) and keep exactly 'limit' rows/files."""
    if (not force) and os.path.exists(base_meta_csv):
        try:
            meta = pd.read_csv(base_meta_csv)
            # Check files exist
            ok_files = meta['path'].apply(lambda rel: os.path.exists(os.path.join(base_books_dir, rel))).all()
            if len(meta) >= limit and ok_files:
                print(f"[BASE] Reusing existing base set: {len(meta)} rows at {base_books_dir}")
                # Trim if larger than needed
                if len(meta) > limit:
                    meta = meta.iloc[:limit].copy()
                    meta.to_csv(base_meta_csv, index=False)
                return
        except Exception:
            pass

    print(f"[BASE] Downloading StorySet ({limit}) → {base_books_dir}")
    ensure_dir(base_books_dir)
    ensure_dir(os.path.dirname(base_meta_csv) or ".")
    # Use the helper from your project (deterministic shuffle inside the function)
    BP.download_storyset_with_genres(out_books=base_books_dir, out_meta=base_meta_csv, limit=limit)

    # Re-read and ensure exactly 'limit' (in case fewer produced)
    meta = pd.read_csv(base_meta_csv)
    if len(meta) < limit:
        print(f"[WARN] Only {len(meta)} books were written (requested {limit}). Continuing with available.")
    else:
        meta = meta.iloc[:limit].copy()
        meta.to_csv(base_meta_csv, index=False)

def make_subset_from_base(base_books_dir, base_meta_csv, n, out_books_dir, out_meta_csv):
    """Create nested subset n by copying first n files from base metadata order."""
    ensure_dir(out_books_dir)
    meta_full = pd.read_csv(base_meta_csv)
    if len(meta_full) < n:
        raise SystemExit(f"Base set has only {len(meta_full)} books; cannot build subset of size {n}.")

    meta_n = meta_full.iloc[:n].copy()

    # Copy files to subset dir; set relative path to just file name
    rows = []
    for _, row in meta_n.iterrows():
        src = os.path.join(base_books_dir, row['path'])
        fname = os.path.basename(row['path'])
        dst = os.path.join(out_books_dir, fname)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        rows.append({
            "book_id": row["book_id"],
            "path": fname,  # relative to this subset dir
            "title": row["title"],
            "author": row.get("author", "unknown"),
            "genre": row.get("genre", "unknown"),
        })
    pd.DataFrame(rows).to_csv(out_meta_csv, index=False)
    print(f"[SUBSET] Built subset N={n}: {out_books_dir} | {out_meta_csv}")

def run_pipeline_for_subset(books_dir, meta_csv, artifacts_dir, knn=10, force=False):
    """Run graphs → embeddings → kNN for this subset (skip if cached unless force=True)."""
    ensure_dir(artifacts_dir)
    emb_path = os.path.join(artifacts_dir, "emb.npy")
    edge_idx_path = os.path.join(artifacts_dir, "edge_index.npy")
    edge_w_path = os.path.join(artifacts_dir, "edge_weight.npy")

    need_run = force or not (os.path.exists(emb_path) and os.path.exists(edge_idx_path) and os.path.exists(edge_w_path))
    if not need_run:
        print(f"[PIPELINE] Using cached artifacts in {artifacts_dir}")
        return

    args = [
        "--books-dir", books_dir,
        "--metadata-csv", meta_csv,
        "--artifacts-dir", artifacts_dir,
        "--knn", str(knn),
        "--build-graphs", "--embed", "--build-global"
    ]
    print(f"[PIPELINE] Running: N={len(pd.read_csv(meta_csv))} | artifacts={artifacts_dir}")
    BP.main(args)

def cosine_topk_average_from_matrix(M, topk=10):
    import numpy as np
    M = np.asarray(M, dtype=np.float64)
    N = M.shape[0]
    topk = min(topk, max(1, N - 1))   # <-- clamp to N-1, but at least 1

    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    Z = M / norms
    S = Z @ Z.T  # cosine similarity

    avg_sim = np.zeros(N, dtype=np.float64)
    for i in range(N):
        row = S[i].copy()
        row[i] = -np.inf                 # exclude self
        idx = np.argpartition(row, -topk)[-topk:]
        vals = np.sort(row[idx])[::-1]
        avg_sim[i] = float(np.mean(vals))
    return avg_sim


def summarize_distribution(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
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

def plot_distributions(series_map, out_path, bins=30):
    import numpy as np
    import matplotlib.pyplot as plt

    # Filter out non-finite values per series
    series_map = {
        label: np.asarray(vals, dtype=float)[np.isfinite(vals)]
        for label, vals in series_map.items()
    }
    # Drop any empty series (just in case)
    series_map = {k: v for k, v in series_map.items() if v.size > 0}

    # Common range across all series
    all_vals = np.concatenate(list(series_map.values()))
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0  # safe fallback
    span = hi - lo if hi > lo else 1.0
    lo -= 0.02 * span
    hi += 0.02 * span

    plt.figure(figsize=(8, 5))
    for label, vals in series_map.items():
        plt.hist(vals, bins=bins, range=(lo, hi), alpha=0.5, label=label, density=True)
    plt.xlabel("Average cosine similarity to top-10 neighbors")
    plt.ylabel("Density")
    plt.title("Average Similarity Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Benchmark growth: nested subsets and similarity distributions")
    ap.add_argument("--limit", type=int, default=100, help="Base dataset size to download once")
    ap.add_argument("--sizes", type=int, nargs="+", default=SIZES_DEFAULT,
                    help="Nested subset sizes (each adds more books)")
    ap.add_argument("--topk", type=int, default=10, help="Top-K neighbors to average per book")
    ap.add_argument("--knn", type=int, default=10, help="k in kNN graph during pipeline")
    ap.add_argument("--root", type=str, default=".", help="Project root")
    ap.add_argument("--force", action="store_true", help="Rebuild even if artifacts exist")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    data_dir = os.path.join(root, "data")
    artifacts_root = os.path.join(root, "artifacts", "exp_growth")
    ensure_dir(artifacts_root)

    # 1) Build base set (100)
    base_books_dir = os.path.join(data_dir, f"books_base{args.limit}")
    base_meta_csv  = os.path.join(data_dir, f"metadata_base{args.limit}.csv")
    prepare_base100(base_books_dir, base_meta_csv, limit=args.limit, force=args.force)

    # 2) Build nested subsets
    dist_map = {}
    summary_rows = []

    for n in args.sizes:
        subset_books = os.path.join(data_dir, f"books_n{n}")
        subset_meta  = os.path.join(data_dir, f"metadata_n{n}.csv")
        subset_art   = os.path.join(artifacts_root, f"n{n}")
        ensure_dir(subset_art)

        make_subset_from_base(base_books_dir, base_meta_csv, n, subset_books, subset_meta)
        run_pipeline_for_subset(subset_books, subset_meta, subset_art, knn=args.knn, force=args.force)

        # 3) Compute average similarity to top-10 neighbors
        avg_sim = compute_avg_topk_similarity(subset_art, topk=args.topk, save=True)
        dist_map[f"N={n}"] = avg_sim

        # 4) Collect summary stats
        s = summarize_distribution(avg_sim)
        s.update({"size": n})
        summary_rows.append(s)

    # 5) Save summary table and combined plot
    df = pd.DataFrame(summary_rows).sort_values("size")
    df.to_csv(os.path.join(artifacts_root, "avg_sim_summary.csv"), index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = "```\n" + df.to_string(index=False) + "\n```"
    with open(os.path.join(artifacts_root, "avg_sim_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[SUMMARY]\n{df.to_string(index=False)}")

    plot_distributions(dist_map, os.path.join(artifacts_root, "compare_avg_sim_hist.png"), bins=30)
    print("[DONE] Growth benchmarking complete.")

if __name__ == "__main__":
    main()
