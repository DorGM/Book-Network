# Graph2Books (BookNet)

End-to-end pipeline for **book-level similarity search** and **(optional) genre classification** using **text-as-graph** representations.  
Each book becomes a weighted co-occurrence graph → embedded with a lightweight Graph2Vec-style method (WL + TF-IDF + SVD) → linked in a global k-NN graph for retrieval.  
When labels exist, a small GCN performs node (book/genre) classification.

- Python 3.13 compatible
- Works on CPU (CUDA optional)
- Reproducible artifacts under `artifacts/`
- Optional Streamlit UI

---

## Contents

- Features
- Repository structure
- Installation
- Quick start
- Using your own data
- Reports and stats
- Interactive app
- Growth benchmark
- Key CLI flags
- Troubleshooting
- Citation & license

---

## Features

- **Per-book graphs**: token/character co-occurrence (windowed), frequency filter, node cap, isolate pruning  
- **Graph embeddings**: WL subtree features → TF-IDF → Truncated SVD (Graph2Vec-lite)  
- **Global k-NN**: cosine/Euclidean similarity for fast retrieval and analysis  
- **Optional GCN**: 2-layer Graph Convolutional Network for genre classification  
- **Diagnostics**: dataset/graph stats, embedding variance, global-graph structure  
- **Similarity analysis**: histogram of each book’s mean similarity to its top-10 neighbors  
- **Caching**: graphs, embeddings, k-NN arrays, metrics, tables, plots in `artifacts/`  
- **Growth study**: nested subsets (e.g., N=10,20,50,75,100) to show scale effects  

---

## Repository structure

.
├── BigData_Project.py          # Main pipeline (graphs → embeds → kNN → (optional) GCN)
├── analyze_stats.py            # Turns artifacts/stats.json into a readable report
├── app_similar_books.py        # Streamlit app for interactive top-k recommendations
├── benchmark_growth.py         # Nested-subset experiment (N=10,20,50,75,100)
├── prepare_pg19.py             # (Optional) helper for other corpora
├── data/                       # Books (.txt) + metadata.csv (created by scripts)
└── artifacts/                  # Cached graphs, embeddings, kNN, metrics, plots, tables

---

## Installation

Works on Windows/macOS/Linux. Example below uses a virtual environment.

Create and activate a venv:
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate

Install dependencies:
    pip install -U pip wheel
    pip install numpy pandas scikit-learn networkx tqdm matplotlib streamlit datasets tabulate

Install PyTorch (CPU example; pick CUDA wheels if you have a GPU):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Install PyTorch Geometric:
    pip install torch-geometric

If PyG fails, see: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

---

## Quick start

Run the full pipeline on a small labeled corpus (StorySet):
    python -u BigData_Project.py ^
      --download-storyset --limit 50 ^
      --build-graphs --embed --build-global --train-gcn

Outputs (under `artifacts/`):
- `graphs.pkl`, `id_map.csv` — per-book graphs (cached)
- `emb.npy` — embeddings (N×d)
- `edge_index.npy`, `edge_weight.npy` — global k-NN graph
- `metrics.json`, `gcn_best.pt`, `training_log.csv`, `gcn_results.(csv|md)` — if `--train-gcn`
- `stats.json`, `report.md`
- `similarity_avg_hist.png`, `similarity_example_best.csv` — similarity diagnostics

Similarity only (no GCN):
    python -u BigData_Project.py --download-storyset --limit 50 --build-graphs --embed --build-global

---

## Using your own data

Place `.txt` files under `data/books/` and create `data/metadata.csv` with columns:
    book_id,path,title,author,genre

`path` is relative to `books_dir` (e.g., just the filename if files live directly under `data/books/`).

Run:
    python -u BigData_Project.py --books-dir data/books --metadata-csv data/metadata.csv ^
      --build-graphs --embed --build-global --train-gcn

---

## Reports and stats

Build a human-readable summary from `artifacts/stats.json`:
    python -u analyze_stats.py

This writes/updates `artifacts/report.md`.

---

## Interactive app

After embeddings and k-NN are created:
    streamlit run app_similar_books.py

- Choose a book by type-ahead or from list (mutually exclusive).
- Set top-k.
- The app validates inputs and returns the most similar titles (cosine).

---

## Growth benchmark

Show how similarity changes as the corpus grows (nested subsets keep comparability):
    python -u benchmark_growth.py --limit 100 --sizes 10 20 50 75 100

Artifacts under `artifacts/exp_growth/`:
- `n{N}/` — per-N embeddings and k-NN arrays
- `avg_sim_summary.(csv|md)` — summary table
- `compare_avg_sim_hist.png` — overlaid histograms (per-book mean top-10 similarity)

If a previous run produced artifacts, re-run without `--force` to reuse caches and resume.

---

## Key CLI flags (from BigData_Project.py)

- I/O: `--books-dir`, `--metadata-csv`, `--artifacts-dir`
- Graph: `--graph-type {token,char}`, `--window-size`, `--min-freq`, `--max-nodes`, `--no-lowercase`, `--no-strip-punct`
- Embeddings: `--emb-dim` (default 128), `--wl-iterations` (default 2)
- Global graph: `--knn` (default 10), `--similarity {cosine,euclidean}`
- GCN: `--hidden-dim`, `--dropout`, `--lr`, `--weight-decay`, `--gcn-epochs`, `--val-size`, `--test-size`, `--seed`
- Steps: `--build-graphs`, `--embed`, `--build-global`, `--train-gcn`
- Download: `--download-storyset`, `--limit`
- Caching: `--save-graphs` / `--no-save-graphs`, `--load-graphs`

---

## Troubleshooting

- PyG import error  
  Reinstall matching PyTorch/PyG wheels (CPU vs CUDA) per PyG docs.

- `to_markdown` requires `tabulate`  
  pip install tabulate

- JSON “not serializable (int64/float32)”  
  The code uses a NumPy-safe encoder; ensure you’re on the latest version.

- Very small N causes `-inf` in averages  
  Tools clamp top-k to `N-1` and filter non-finite values in plots. If needed, start from larger N.
