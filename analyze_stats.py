#!/usr/bin/env python3
"""
Analyze & visualize BookNet pipeline statistics (run after BigData_Project.py).

Inputs:
- artifacts/stats.json
- artifacts/metrics.json (optional)
- data/metadata.csv (optional)

Outputs:
- artifacts/plots/*.png
- artifacts/tables/*.csv
- artifacts/report_viz.md
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ART = "artifacts"
PLOTS = os.path.join(ART, "plots")
TABLES = os.path.join(ART, "tables")
REPORT = os.path.join(ART, "report_viz.md")

def ensure_dirs():
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

def load_json(path):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def save_fig(fig, name):
    path = os.path.join(PLOTS, name)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return os.path.relpath(path)

def write_table(df, name):
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=False)
    return os.path.relpath(path)

def ptable(d: dict):
    if not d: return pd.DataFrame(columns=["metric","value"])
    return pd.DataFrame({"metric": list(d.keys()), "value": list(d.values())})

def percentiles_to_series(p):
    order = ["count","min","p10","p25","median","p75","p90","max","mean","std"]
    return pd.Series({k: p.get(k, None) for k in order})

def main():
    ensure_dirs()
    stats = load_json(os.path.join(ART, "stats.json"))
    metrics = load_json(os.path.join(ART, "metrics.json"))
    meta_path = os.path.join("data", "metadata.csv")

    md = ["# Visualization Report\n"]

    # Dataset summary
    ds = stats.get("dataset", {})
    md += ["## Dataset\n"]
    if ds:
        write_table(ptable(ds), "dataset_summary.csv")
        md += [f"- **Books**: {ds.get('num_books','NA')}"]
        md += [f"- **Genres (unique)**: {ds.get('num_genres','NA')}"]
        fs = ds.get("file_size_bytes", {})
        if fs:
            md += [f"- **File size (bytes)**: median {int(fs.get('median',0))}, p90 {int(fs.get('p90',0))}, max {int(fs.get('max',0))}"]
    else:
        md += ["_No dataset stats found._"]

    # Genre distribution
    md += ["\n### Genre distribution\n"]
    genre_counts = None
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        if "genre" in meta.columns:
            genre_counts = meta["genre"].astype(str).value_counts()
    else:
        labels = stats.get("labels", {})
        if "class_counts_top10" in labels:
            genre_counts = pd.Series(labels["class_counts_top10"])

    if genre_counts is not None and not genre_counts.empty:
        df_genre = genre_counts.rename("count").reset_index(name=["genre"])
        write_table(df_genre, "genre_counts.csv")
        top = genre_counts.head(20)
        fig, ax = plt.subplots(figsize=(10, 5))
        top.plot(kind="bar", ax=ax)
        ax.set_xlabel("Genre"); ax.set_ylabel("Count"); ax.set_title("Top 20 genres by count")
        ax.tick_params(axis="x", labelrotation=45)
        for lbl in ax.get_xticklabels(): lbl.set_horizontalalignment("right")
        img = save_fig(fig, "genre_top20.png")
        md += [f"![Top genres]({img})\n", "_Full table: `artifacts/tables/genre_counts.csv`._"]
    else:
        md += ["_No genre info available._"]

    # Per-book graphs
    md += ["\n## Per-book graph statistics\n"]
    gstats = stats.get("per_book_graphs", {})
    if gstats:
        for key, fname in [
            ("nodes_per_book", "nodes_per_book_percentiles.csv"),
            ("edges_per_book", "edges_per_book_percentiles.csv"),
            ("degree_over_all_books", "degree_over_all_books_percentiles.csv"),
            ("edge_weight_distribution", "edge_weight_percentiles.csv"),
            ("avg_clustering_lcc_per_book", "avg_clustering_LCC_percentiles.csv"),
        ]:
            if key in gstats: write_table(percentiles_to_series(gstats[key]).to_frame().T, fname)
        md += [f"- **Books with graphs**: {gstats.get('books_with_graphs','NA')}"]
        if "nodes_per_book" in gstats:
            p = gstats["nodes_per_book"]; md += [f"- **Nodes/book** median {int(p.get('median',0))}, p90 {int(p.get('p90',0))}, max {int(p.get('max',0))}"]
        if "edges_per_book" in gstats:
            p = gstats["edges_per_book"]; md += [f"- **Edges/book** median {int(p.get('median',0))}, p90 {int(p.get('p90',0))}, max {int(p.get('max',0))}"]
        md += ["_Percentile tables saved under `artifacts/tables/`._"]
    else:
        md += ["_No per-book graph stats found._"]

    # Embeddings
    md += ["\n## Embeddings (Graph2Vec-lite)\n"]
    emb = stats.get("embeddings", {})
    if emb:
        md += [f"- **Shape**: {emb.get('shape','NA')}"]
        md += [f"- **SVD cumulative explained variance**: {emb.get('svd_cumulative_explained','NA')}"]
        top10 = emb.get("svd_top10", [])
        if top10:
            vals = np.array(top10, dtype=float)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(np.arange(1, len(vals)+1), vals)
            ax.set_xlabel("Component"); ax.set_ylabel("Explained variance ratio")
            ax.set_title("Top components — explained variance (SVD)")
            img = save_fig(fig, "svd_explained_variance_top.png")
            md += [f"![SVD explained variance]({img})"]
    else:
        md += ["_No embedding stats found._"]

    # Global graph
    md += ["\n## Global similarity graph (k-NN on embeddings)\n"]
    gg = stats.get("global_similarity_graph", {})
    if gg:
        write_table(ptable(gg), "global_graph_summary.csv")
        md += [
            f"- **Nodes**: {gg.get('num_nodes','NA')}  •  **Edges**: {gg.get('num_edges','NA')}  •  **Avg degree**: {gg.get('avg_degree','NA')}",
            f"- **Density**: {gg.get('density','NA')}  •  **Connected components**: {gg.get('num_connected_components','NA')}  •  **Largest comp.**: {gg.get('largest_component_size','NA')}",
            f"- **Avg clustering (LCC)**: {gg.get('avg_clustering_on_LCC','NA')}",
            "_Full table: `artifacts/tables/global_graph_summary.csv`._"
        ]
    else:
        md += ["_No global graph stats found._"]

    # Classifier (if trained)
    md += ["\n## Classifier (GCN)\n"]
    clf = stats.get("classifier", {})
    if clf:
        rows = []
        for split in ("train","val","test"):
            rows.append({
                "split": split,
                "support": clf.get(f"{split}_support", 0),
                "acc": clf.get(f"{split}_acc", None),
                "balanced_acc": clf.get(f"{split}_balanced_acc", None),
                "f1_macro": clf.get(f"{split}_f1_macro", None),
            })
        df_clf = pd.DataFrame(rows)
        write_table(df_clf, "classifier_summaries.csv")
        md += [df_clf.to_markdown(index=False)]
        for split in ("train","val","test"):
            cm = clf.get(f"{split}_confusion_matrix", None)
            if cm:
                cm = np.array(cm)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                im = ax.imshow(cm, cmap="Blues")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"{split.capitalize()} confusion matrix")
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                if cm.shape[0] <= 20:
                    ax.set_xticks(range(cm.shape[1])); ax.set_yticks(range(cm.shape[0]))
                img = save_fig(fig, f"cm_{split}.png")
                md += [f"![{split} CM]({img})"]
        rep = clf.get("test_classification_report", None)
        if rep:
            df_rep = pd.DataFrame(rep).T.reset_index().rename(columns={"index":"label"})
            write_table(df_rep, "classification_report_test.csv")
            md += ["_Detailed per-class test report saved._"]
    else:
        md += ["_No classifier stats found (either skipped or labels too sparse)._"]

    # Timings
    times = stats.get("timings_sec", {})
    if times:
        md += ["\n## Timings (seconds)\n"]
        df_t = ptable(times); write_table(df_t, "stage_timings.csv")
        md += [df_t.to_markdown(index=False)]

    # metrics.json (flat)
    if metrics:
        md += ["\n## metrics.json (flat)\n"]
        df_m = ptable(metrics); write_table(df_m, "metrics_json_flat.csv")
        md += [df_m.to_markdown(index=False)]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[OK] Wrote:\n- {REPORT}\n- {PLOTS}/*.png\n- {TABLES}/*.csv")

if __name__ == "__main__":
    main()
