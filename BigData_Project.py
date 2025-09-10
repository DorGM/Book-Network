import os
import re
import gc
import json
import math
import time
import random
import pickle
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

# Embedding & ML
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# PyTorch + PyTorch Geometric
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, average_precision_score, top_k_accuracy_score,
    cohen_kappa_score, matthews_corrcoef, log_loss
)

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
except Exception as e:
    raise SystemExit(
        "PyTorch Geometric is required. See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html\n"
        "Typical: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && "
        "pip install torch-geometric"
    )

warnings.filterwarnings("ignore", category=UserWarning)


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        import numpy as np
        if isinstance(o, (np.integer,)):   return int(o)
        if isinstance(o, (np.floating,)):  return float(o)
        if isinstance(o, (np.ndarray,)):   return o.tolist()
        return super().default(o)

# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    books_dir: str = "data/books"
    metadata_csv: str = "data/metadata.csv"
    artifacts_dir: str = "artifacts"

    # Graph construction
    graph_type: str = "token"  # "token" or "char"
    window_size: int = 5
    min_freq: int = 3
    max_nodes: int = 5000
    lowercase: bool = True
    strip_punct: bool = True

    # Parallelism
    workers: int = min(8, os.cpu_count() or 2)

    # Graph2Vec‑lite
    emb_dim: int = 128
    wl_iterations: int = 2

    # Global similarity graph
    knn: int = 10
    similarity: str = "cosine"  # "cosine" or "euclidean"

    # GCN training
    hidden_dim: int = 128
    dropout: float = 0.5
    lr: float = 1e-2
    weight_decay: float = 5e-4
    gcn_epochs: int = 200
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    # Control steps
    save_graphs: bool = True
    load_graphs: bool = False


# -----------------------------
# Utilities
# -----------------------------
STOPWORDS = set(
    """
    a an and are as at be but by for if in into is it no not of on or s such t that the their then there these they this to was will with from we you your our can could should would i he she her his theirs ours me my mine yours who whom whose which while do does did done have has had were been being over under above below between across also than out up down off only very same too more most less least just
    """.split()
)

TOKEN_RE = re.compile(r"[A-Za-z']+")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Data Loading
# -----------------------------

def discover_books(books_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(books_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                paths.append(os.path.join(root, f))
    paths.sort()
    return paths


def load_metadata(cfg: Config, book_paths: List[str]) -> pd.DataFrame:
    if os.path.exists(cfg.metadata_csv):
        meta = pd.read_csv(cfg.metadata_csv)
        required = {"book_id", "path", "title", "author", "genre"}
        missing = required.difference(set(meta.columns))
        if missing:
            raise ValueError(f"metadata.csv missing columns: {missing}")
        # Normalize paths to absolute
        meta["abs_path"] = meta["path"].apply(lambda p: os.path.join(cfg.books_dir, p))
        # Validate existence
        missing_files = meta[~meta["abs_path"].apply(os.path.exists)]
        if len(missing_files) > 0:
            print("[WARN] Some metadata paths do not exist on disk (they will be skipped):")
            print(missing_files[["book_id", "path"]].head())
        meta = meta[meta["abs_path"].apply(os.path.exists)].copy()
        meta = meta.reset_index(drop=True)
    else:
        # Auto-index all .txt files
        rel_paths = [os.path.relpath(p, cfg.books_dir) for p in book_paths]
        meta = pd.DataFrame({
            "book_id": [f"book_{i:06d}" for i in range(len(rel_paths))],
            "path": rel_paths,
            "title": [os.path.basename(p) for p in rel_paths],
            "author": ["unknown"] * len(rel_paths),
            "genre": ["unknown"] * len(rel_paths),
        })
        meta["abs_path"] = [os.path.join(cfg.books_dir, p) for p in rel_paths]
    return meta


# -----------------------------
# Text → Graph per book
# -----------------------------

def simple_tokenize(text: str, lowercase=True, strip_punct=True) -> List[str]:
    if lowercase:
        text = text.lower()
    if strip_punct:
        tokens = TOKEN_RE.findall(text)
    else:
        tokens = text.split()
    # filter stopwords & tiny tokens
    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return tokens


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def build_token_graph(tokens: List[str], window_size: int, min_freq: int, max_nodes: int) -> nx.Graph:
    # Keep top tokens by frequency
    from collections import Counter, deque

    freqs = Counter(tokens)
    vocab = [w for w, c in freqs.items() if c >= min_freq]
    # Limit max nodes by frequency
    vocab.sort(key=lambda w: (-freqs[w], w))
    vocab = set(vocab[:max_nodes])

    G = nx.Graph()
    for w in vocab:
        G.add_node(w, label=w)

    window = deque(maxlen=window_size)
    for tok in tokens:
        if tok not in vocab:
            continue
        # add edges with items already in the window
        for other in window:
            if other == tok:
                continue
            if G.has_edge(tok, other):
                G[tok][other]['weight'] += 1
            else:
                G.add_edge(tok, other, weight=1)
        window.append(tok)
    # Prune isolated nodes (no edges)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def build_char_graph(text: str, window_size: int, max_nodes: int) -> nx.Graph:
    # Build based on character co-occurrence (letters only)
    from collections import deque, Counter
    chars = [c.lower() for c in text if c.isalpha()]
    freqs = Counter(chars)
    # Keep most frequent chars (practically all 26 letters, but robust for multilingual)
    top_chars = [ch for ch, _ in freqs.most_common(max_nodes)]
    vocab = set(top_chars)

    G = nx.Graph()
    for ch in vocab:
        G.add_node(ch, label=ch)

    window = deque(maxlen=window_size)
    for ch in chars:
        if ch not in vocab:
            continue
        for other in window:
            if other == ch:
                continue
            if G.has_edge(ch, other):
                G[ch][other]['weight'] += 1
            else:
                G.add_edge(ch, other, weight=1)
        window.append(ch)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    return G


def build_graph_for_book(path: str, cfg: Config) -> nx.Graph:
    text = read_text(path)
    if cfg.graph_type == "token":
        tokens = simple_tokenize(text, lowercase=cfg.lowercase, strip_punct=cfg.strip_punct)
        return build_token_graph(tokens, cfg.window_size, cfg.min_freq, cfg.max_nodes)
    elif cfg.graph_type == "char":
        return build_char_graph(text, cfg.window_size, cfg.max_nodes)
    else:
        raise ValueError(f"Unknown graph_type: {cfg.graph_type}")


# -----------------------------
# Embedding with Graph2Vec‑lite (WL + TF‑IDF + SVD)
# -----------------------------
from collections import defaultdict

def _wl_feature_multiset(G, wl_iterations=2):
    """Return a list of tokens representing Weisfeiler-Lehman (WL) subtree features for the whole graph.
    Each iteration refines node labels by hashing the multiset of neighbor labels.
    The resulting bag-of-subtrees works as the 'words' for Graph2Vec-style document embeddings.
    """
    labels = {n: str(G.nodes[n].get('label', G.degree[n])) for n in G.nodes()}
    tokens = []
    for it in range(wl_iterations + 1):
        tokens.extend(labels.values())
        if it == wl_iterations:
            break
        new_labels = {}
        for v in G.nodes():
            neigh = sorted(labels[u] for u in G.neighbors(v))
            new_labels[v] = labels[v] + "|" + "|".join(neigh)
        uniq = {lab: i for i, lab in enumerate(sorted(set(new_labels.values())))}
        labels = {v: f"wl{it}_{uniq[new_labels[v]]}" for v in G.nodes()}
    return tokens


def graph2vec_embeddings_sklearn(graphs, dim=128, wl_iterations=2, random_state=42, return_info=False):
    """Graph2Vec-like embeddings using WL subtree tokens + TF-IDF + TruncatedSVD.
    Returns:
      - if return_info=False: Z  (ndarray [N, dim])
      - if return_info=True:  (Z, info_dict)
    info_dict = {
        'n_features': int,
        'svd_explained_var_ratio': list[float],
        'svd_cumulative_explained': float
    }
    """
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.decomposition import TruncatedSVD

    corpus = [' '.join(_wl_feature_multiset(g, wl_iterations=wl_iterations)) for g in graphs]

    hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm=None)
    X = hv.transform(corpus)                      # sparse [N x 2^18]
    X = TfidfTransformer().fit_transform(X)      # TF-IDF weighting

    svd = TruncatedSVD(n_components=dim, random_state=random_state)
    Z = svd.fit_transform(X)                     # dense [N x dim]
    if not return_info:
        return Z
    expl = svd.explained_variance_ratio_
    info = {
        'n_features': int(X.shape[1]),
        'svd_explained_var_ratio': expl.tolist(),
        'svd_cumulative_explained': float(expl.sum())
    }
    return Z, info



# -----------------------------
# Build global book similarity graph
# -----------------------------

def build_global_graph(emb: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (edge_index [2,E], edge_weight [E]) suitable for PyG."""
    n = emb.shape[0]
    k = cfg.knn
    metric = cfg.similarity

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric, algorithm='auto', n_jobs=-1)
    nbrs.fit(emb)
    distances, indices = nbrs.kneighbors(emb)

    # Build undirected edges, skip self (first neighbor is the point itself)
    edges = set()
    weights = []
    for i in range(n):
        for j_idx, dist in zip(indices[i][1:], distances[i][1:]):
            j = int(j_idx)
            if metric == 'cosine':
                # cosine distance → similarity
                w = 1.0 - float(dist)
            else:
                # euclidean → convert to similarity (soft)
                w = 1.0 / (1.0 + float(dist))
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edges:
                edges.add((a, b))
                weights.append(w)

    edge_index = np.array(list(edges)).T  # shape [2, E]
    edge_weight = np.array(weights, dtype=np.float32)
    return edge_index, edge_weight


# -----------------------------
# PyG Data assembly
# -----------------------------

def encode_labels(genres: List[str]) -> Tuple[np.ndarray, Dict[int, str]]:
    le = LabelEncoder()
    y = le.fit_transform(genres)
    idx2label = {i: lab for i, lab in enumerate(le.classes_)}
    return y, idx2label


def make_splits(y: np.ndarray, val_size: float, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    # Ensure at least 1 sample per class in train: use stratified split twice
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(idx, y, test_size=val_size + test_size, stratify=y, random_state=seed)
    rel_test = test_size / (val_size + test_size + 1e-8)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=seed)

    train_mask = np.zeros_like(y, dtype=bool)
    val_mask = np.zeros_like(y, dtype=bool)
    test_mask = np.zeros_like(y, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True
    return train_mask, val_mask, test_mask


def assemble_pyg_data(emb: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray, y: np.ndarray,
                      train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> Data:
    x = torch.from_numpy(emb).float()
    ei = torch.from_numpy(edge_index).long()
    ew = torch.from_numpy(edge_weight).float()
    y_t = torch.from_numpy(y).long()

    data = Data(x=x, edge_index=ei, edge_weight=ew, y=y_t)
    data.train_mask = torch.from_numpy(train_mask)
    data.val_mask = torch.from_numpy(val_mask)
    data.test_mask = torch.from_numpy(test_mask)
    return data


# -----------------------------
# GCN Model for node classification
# -----------------------------
class GCNModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return x


def train_gcn(data: Data, cfg: Config, idx2label: Dict[int, str], out_dir: str) -> Tuple[Dict[str, float], nn.Module]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCNModel(in_dim=data.x.size(1), hidden_dim=cfg.hidden_dim, out_dim=len(idx2label), dropout=cfg.dropout)
    model = model.to(device)
    data = data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = -1.0
    best_state = None
    best_epoch = 0
    log_rows = []

    for epoch in range(1, cfg.gcn_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_weight)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_weight)
            pred = logits.argmax(dim=1)

            def _acc(mask):
                return accuracy_score(data.y[mask].cpu(), pred[mask].cpu()) if mask.sum() > 0 else float('nan')

            def _f1(mask):
                return f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro') if mask.sum() > 0 else float('nan')

            val_acc = _acc(data.val_mask)
            val_f1  = _f1(data.val_mask)
            val_loss = (F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()
                        if int(data.val_mask.sum()) > 0 else float('nan'))

        log_rows.append({
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
        })

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Epoch {epoch:3d}] train_loss={train_loss.item():.4f} val_acc={val_acc:.4f}", flush=True)

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final metrics on best model
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight)
        pred = logits.argmax(dim=1)

        def _acc(mask):
            return accuracy_score(data.y[mask].cpu(), pred[mask].cpu()) if mask.sum() > 0 else float('nan')

        def _f1(mask):
            return f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro') if mask.sum() > 0 else float('nan')

        def _loss(mask):
            return (F.cross_entropy(logits[mask], data.y[mask]).item()
                    if int(mask.sum()) > 0 else float('nan'))

        metrics = {
            'train_acc': _acc(data.train_mask),
            'val_acc': _acc(data.val_mask),
            'test_acc': _acc(data.test_mask),
            'train_f1': _f1(data.train_mask),
            'val_f1': _f1(data.val_mask),
            'test_f1': _f1(data.test_mask),
            'train_loss_best': _loss(data.train_mask),
            'val_loss_best': _loss(data.val_mask),
            'test_loss_best': _loss(data.test_mask),
            'best_val_acc': float(best_val),
            'best_epoch': int(best_epoch),
            'epochs_total': int(cfg.gcn_epochs),
        }

    # Persist the per-epoch log (for your appendix plots/tables)
    try:
        import pandas as pd
        pd.DataFrame(log_rows).to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    except Exception:
        pass

    return metrics, model



# -----------------------------
# Optional: Simple Link Prediction (dot-product decoder)
# -----------------------------

def link_prediction_auc(data: Data, embeddings: Optional[torch.Tensor] = None, neg_ratio: int = 1, seed: int = 42) -> float:
    """Compute a very simple link prediction AUC using dot products on node embeddings.
    If embeddings is None, uses data.x (Graph2Vec‑lite embeddings). In practice, you may want to use the
    hidden layer outputs of the trained GCN instead, by forwarding once and taking the penultimate layer.
    """
    rng = np.random.RandomState(seed)
    edge_index = data.edge_index.cpu().numpy()
    E = edge_index.shape[1]
    n = data.x.size(0)
    # Build set of existing undirected edges
    edges = set()
    for a, b in edge_index.T:
        a, b = int(a), int(b)
        if a > b: a, b = b, a
        edges.add((a, b))
    # Sample negative edges
    num_neg = E * neg_ratio
    neg = set()
    while len(neg) < num_neg:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edges:
            continue
        neg.add((a, b))

    # Scores
    emb = embeddings if embeddings is not None else data.x
    if isinstance(emb, np.ndarray):
        emb = torch.from_numpy(emb).float()
    emb = emb.to(data.x.device)

    def score(pair_list: List[Tuple[int, int]]):
        pairs = torch.tensor(pair_list, dtype=torch.long, device=emb.device)
        s = (emb[pairs[:, 0]] * emb[pairs[:, 1]]).sum(dim=1)  # dot product
        return torch.sigmoid(s).detach().cpu().numpy()

    pos_pairs = list(edges)
    neg_pairs = list(neg)
    y_true = np.array([1] * len(pos_pairs) + [0] * len(neg_pairs))
    y_score = np.concatenate([score(pos_pairs), score(neg_pairs)])

    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float('nan')
    return auc


# -----------------------------
# Orchestration
# -----------------------------

def build_graphs(cfg: Config, meta: pd.DataFrame) -> Tuple[List[nx.Graph], List[str]]:
    graphs = []
    ids = []
    print(f"Building per‑book graphs ({cfg.graph_type})...")
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        g = build_graph_for_book(row["abs_path"], cfg)
        if g.number_of_nodes() == 0 or g.number_of_edges() == 0:
            continue  # skip empty graphs
        graphs.append(g)
        ids.append(row["book_id"])
    print(f"Built {len(graphs)} graphs (out of {len(meta)} books).")
    return graphs, ids


def save_graphs(graphs: List[nx.Graph], ids: List[str], cfg: Config):
    ensure_dir(cfg.artifacts_dir)
    with open(os.path.join(cfg.artifacts_dir, "graphs.pkl"), "wb") as f:
        pickle.dump(graphs, f)
    pd.DataFrame({"book_id": ids}).to_csv(os.path.join(cfg.artifacts_dir, "id_map.csv"), index=False)


def load_graphs(cfg: Config) -> Tuple[List[nx.Graph], List[str]]:
    with open(os.path.join(cfg.artifacts_dir, "graphs.pkl"), "rb") as f:
        graphs = pickle.load(f)
    ids = pd.read_csv(os.path.join(cfg.artifacts_dir, "id_map.csv"))['book_id'].tolist()
    return graphs, ids

def download_pg19(out_books="data/books", out_meta="data/metadata.csv", limit=0):
    """
    Download the Parquet-converted PG-19 from Hugging Face (no loading script)
    and write .txt files + metadata.csv compatible with the pipeline.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please run: pip install datasets")

    import re
    ensure_dir(out_books)
    ensure_dir(os.path.dirname(out_meta) or ".")

    # Parquet version of PG-19 (no Python loader script)
    ds = load_dataset("emozilla/pg19")   # splits: train/validation/test
    rows, total = [], 0

    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        d = ds[split]
        for ex in tqdm(d, desc=f"Writing {split}", total=len(d)):
            title = ex.get("short_book_title") or "unknown"
            url = ex.get("url") or ""
            text = ex.get("text") or ""

            # Derive a stable book_id from the Gutenberg URL if possible
            m = re.search(r"/ebooks/(\\d+)", url)
            book_id = m.group(1) if m else f"pg19_{split}_{total:06d}"

            fname = f"{book_id}.txt"
            fpath = os.path.abspath(os.path.join(out_books, fname))
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)

            rows.append({
                "book_id": book_id,
                "path": os.path.relpath(fpath, start=out_books),
                "title": title,
                "author": "unknown",
                "genre": "unknown",
            })

            total += 1
            if limit and total >= limit:
                break
        if limit and total >= limit:
            break

    pd.DataFrame(rows).to_csv(out_meta, index=False)
    print(f"[OK] Wrote {total} books → {os.path.abspath(out_books)}", flush=True)
    print(f"[OK] Wrote metadata   → {os.path.abspath(out_meta)}", flush=True)


def download_gutenberg_with_genres(out_books="data/books", out_meta="data/metadata.csv", limit=0):
    """
    Download a Gutenberg subset with full text + subjects/themes (acts like genres)
    and write .txt files + metadata.csv for the pipeline.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please run: pip install datasets")

    ensure_dir(out_books)
    ensure_dir(os.path.dirname(out_meta) or ".")

    # This dataset includes full text and standardized subjects/themes.
    ds = load_dataset("Despina/project_gutenberg", "fiction_books", split="train")  # DatasetDict or Dataset
    splits = ds.keys() if hasattr(ds, "keys") else [None]

    rows, total = [], 0
    for split in splits:
        d = ds[split] if split else ds
        for ex in tqdm(d, desc=f"Writing {split or 'data'}", total=len(d)):
            # Be defensive about field names
            book_id = str(ex.get("book_id") or ex.get("id") or total)
            title = ex.get("title") or book_id
            text = ex.get("text") or ex.get("content") or ""
            subjects = ex.get("subjects") or ex.get("themes") or []
            genre = subjects[0] if subjects else "unknown"

            if not text.strip():
                continue  # skip empty texts

            fname = f"{book_id}.txt"
            fpath = os.path.abspath(os.path.join(out_books, fname))
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)

            rows.append({
                "book_id": book_id,
                "path": os.path.relpath(fpath, start=out_books),
                "title": title,
                "author": ex.get("author") or "unknown",
                "genre": genre,
            })
            total += 1
            if limit and total >= limit:
                break
        if limit and total >= limit:
            break

    pd.DataFrame(rows).to_csv(out_meta, index=False)
    print(f"[OK] Wrote {total} books → {os.path.abspath(out_books)}")
    print(f"[OK] Wrote metadata     → {os.path.abspath(out_meta)}")

def download_storyset_with_genres(out_books="data/books", out_meta="data/metadata.csv", limit=0):
    from datasets import load_dataset
    ensure_dir(out_books); ensure_dir(os.path.dirname(out_meta) or ".")

    ds = load_dataset("123Tapwater/StorySet", split="train")  # ~5.6k rows
    if limit and limit > 0:
        ds = ds.shuffle(seed=42).select(range(limit))  # << shuffle before sampling
    else:
        ds = ds.shuffle(seed=42)

    rows = []
    for i, ex in enumerate(tqdm(ds, desc="Writing StorySet", total=len(ds))):
        def g(*keys, default=""):
            for k in keys:
                if k in ex and ex[k] is not None:
                    return ex[k]
            return default

        title = g("title", "Title", default=f"book_{i}")
        genre = g("genre", "Genre", default="unknown")
        text  = g("text", "Text", "raw text", "Raw Text", default="")
        if not str(text).strip():
            continue

        book_id = f"storyset_{i:06d}"  # use i after shuffle/select
        fpath = os.path.abspath(os.path.join(out_books, f"{book_id}.txt"))
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)

        rows.append({
            "book_id": book_id,
            "path": os.path.relpath(fpath, start=out_books),
            "title": str(title),
            "author": "unknown",
            "genre": str(genre).strip().lower(),
        })

    pd.DataFrame(rows).to_csv(out_meta, index=False)
    print(f"[OK] Wrote {len(rows)} books → {os.path.abspath(out_books)}")
    print(f"[OK] Wrote metadata       → {os.path.abspath(out_meta)}")


def make_splits(y: np.ndarray, val_size: float, test_size: float, seed: int,
                min_class_size: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)

    classes, counts = np.unique(y, return_counts=True)
    eligible_classes = classes[counts >= min_class_size]
    elig_mask = np.isin(y, eligible_classes)
    idx_elig = idx[elig_mask]
    y_elig = y[elig_mask]

    # default all-False
    train_mask = np.zeros(n, dtype=bool)
    val_mask   = np.zeros(n, dtype=bool)
    test_mask  = np.zeros(n, dtype=bool)

    try:
        if len(np.unique(y_elig)) >= 2 and len(idx_elig) >= 10:
            # stratified on eligible subset
            idx_train, idx_tmp, y_train, y_tmp = train_test_split(
                idx_elig, y_elig, test_size=val_size + test_size,
                stratify=y_elig, random_state=seed
            )
            rel_test = test_size / (val_size + test_size + 1e-8)
            idx_val, idx_test, y_val, y_test = train_test_split(
                idx_tmp, y_tmp, test_size=rel_test,
                stratify=y_tmp, random_state=seed
            )
            train_mask[idx_train] = True
            val_mask[idx_val]     = True
            test_mask[idx_test]   = True
        else:
            # fallback: non-stratified over all indices
            idx_train, idx_tmp = train_test_split(
                idx, test_size=val_size + test_size, random_state=seed, shuffle=True
            )
            rel_test = test_size / (val_size + test_size + 1e-8)
            idx_val, idx_test = train_test_split(
                idx_tmp, test_size=rel_test, random_state=seed, shuffle=True
            )
            train_mask[idx_train] = True
            val_mask[idx_val]     = True
            test_mask[idx_test]   = True
    except ValueError as e:
        # last-resort fallback
        idx_train, idx_tmp = train_test_split(idx, test_size=val_size + test_size, random_state=seed, shuffle=True)
        rel_test = test_size / (val_size + test_size + 1e-8)
        idx_val, idx_test = train_test_split(idx_tmp, test_size=rel_test, random_state=seed, shuffle=True)
        train_mask[idx_train] = True
        val_mask[idx_val]     = True
        test_mask[idx_test]   = True

    # NOTE: ultra-rare classes remain with all masks False (kept in graph for LP)
    return train_mask, val_mask, test_mask


class StatsCollector:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.data = {}
        self._timers = {}

    def start(self, key: str):
        self._timers[key] = time.time()

    def stop(self, key: str):
        t0 = self._timers.pop(key, None)
        if t0 is not None:
            self.data.setdefault('timings_sec', {})[key] = round(time.time() - t0, 3)

    def add(self, key: str, payload: dict):
        self.data[key] = payload

    def write(self):
        ensure_dir(self.out_dir)
        # JSON (NumPy-safe)
        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, cls=NpEncoder)  # <-- use the encoder
        # Markdown (human-readable)
        md = ["# Pipeline Statistics Report\n"]
        for section, content in self.data.items():
            if section == 'timings_sec':
                continue
            md.append(f"## {section.replace('_', ' ').title()}\n")
            if isinstance(content, dict):
                for k, v in content.items():
                    md.append(f"- **{k}**: {v}")
                md.append("")
        if 'timings_sec' in self.data:
            md.append("## Timings (seconds)\n")
            for k, v in self.data['timings_sec'].items():
                md.append(f"- **{k}**: {v}")
            md.append("")
        with open(os.path.join(self.out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(md))


def _percentiles(arr):
    arr = np.asarray(arr)
    if arr.size == 0: return {}
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def summarize_metadata(meta: pd.DataFrame, books_dir: str) -> dict:
    n = len(meta)
    by_genre = (meta['genre'].astype(str).value_counts().to_dict()
                if 'genre' in meta.columns else {})
    # file sizes (bytes) if available
    sizes = []
    for _, row in meta.iterrows():
        f = os.path.join(books_dir, row['path']) if 'path' in row else None
        if f and os.path.exists(f):
            sizes.append(os.path.getsize(f))
    return {
        "num_books": int(n),
        "num_genres": int(len(by_genre)) if by_genre else 0,
        "genre_counts_top10": dict(list(by_genre.items())[:10]) if by_genre else {},
        "file_size_bytes": _percentiles(np.array(sizes, dtype=np.int64)) if sizes else {},
    }


def summarize_graphs(graphs: List[nx.Graph]) -> dict:
    if not graphs:
        return {}
    n_nodes = np.array([g.number_of_nodes() for g in graphs])
    n_edges = np.array([g.number_of_edges() for g in graphs])

    # edge weight distribution (across all graphs)
    weights = []
    degrees = []
    cc_lcc = []  # clustering coeff on LCC per graph
    for g in graphs:
        degrees.extend([d for _, d in g.degree()])
        for _, _, d in g.edges(data=True):
            w = d.get("weight", 1)
            if w is not None: weights.append(int(w))
        if g.number_of_nodes() > 1:
            lcc_nodes = max(nx.connected_components(g), key=len)
            sub = g.subgraph(lcc_nodes)
            try:
                cc_lcc.append(float(nx.average_clustering(sub)))
            except Exception:
                pass

    return {
        "books_with_graphs": len(graphs),
        "nodes_per_book": _percentiles(n_nodes),
        "edges_per_book": _percentiles(n_edges),
        "degree_over_all_books": _percentiles(np.array(degrees, dtype=int)) if degrees else {},
        "edge_weight_distribution": _percentiles(np.array(weights, dtype=int)) if weights else {},
        "avg_clustering_lcc_per_book": _percentiles(np.array(cc_lcc, dtype=float)) if cc_lcc else {},
    }


def summarize_embeddings(emb: np.ndarray, emb_info: dict) -> dict:
    return {
        "shape": list(emb.shape),
        "svd_cumulative_explained": round(float(emb_info.get('svd_cumulative_explained', float('nan'))), 4),
        "svd_top10": [round(x, 4) for x in emb_info.get('svd_explained_var_ratio', [])[:10]],
        "tfidf_hash_features": int(emb_info.get('n_features', 0)),
    }


def summarize_global_graph(edge_index: np.ndarray, edge_weight: np.ndarray, num_nodes: int) -> dict:
    if edge_index is None or edge_index.size == 0:
        return {}
    E = edge_index.shape[1]
    avg_degree = 2.0 * E / max(1, num_nodes)
    # Build nx Graph for component stats
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)
    n_comp = nx.number_connected_components(G)
    lcc = max(nx.connected_components(G), key=len)
    lcc_size = len(lcc)
    lcc_sub = G.subgraph(lcc)
    try:
        avg_cc = nx.average_clustering(lcc_sub)
    except Exception:
        avg_cc = float("nan")
    density = nx.density(G)
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(E),
        "avg_degree": round(avg_degree, 3),
        "density": round(float(density), 6),
        "num_connected_components": int(n_comp),
        "largest_component_size": int(lcc_size),
        "avg_clustering_on_LCC": round(float(avg_cc), 4),
        "edge_weight": _percentiles(edge_weight) if edge_weight is not None and edge_weight.size > 0 else {},
    }


def summarize_labels(meta: pd.DataFrame) -> dict:
    if 'genre' not in meta.columns:
        return {}
    counts = meta['genre'].astype(str).value_counts()
    imbal_ratio = float(counts.max() / max(1, counts.min())) if len(counts) > 1 else 1.0
    return {
        "num_classes": int(len(counts)),
        "class_counts_top10": dict(counts.head(10)),
        "imbalance_ratio_max_over_min": round(imbal_ratio, 3),
    }


def summarize_classifier(data: Data, logits: torch.Tensor, idx2label: Dict[int, str], topk=(1, 3)) -> dict:
    import numpy as np
    y_true = data.y.cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_proba = torch.softmax(logits, dim=1).cpu().numpy()
    n_classes = y_proba.shape[1]

    out = {}
    for name, mask in [("train", data.train_mask), ("val", data.val_mask), ("test", data.test_mask)]:
        m = mask.cpu().numpy().astype(bool)
        if m.sum() == 0:
            out[f"{name}_support"] = 0
            continue

        yt, yp, yp_prob = y_true[m], y_pred[m], y_proba[m]
        out[f"{name}_support"] = int(m.sum())

        # Core classification metrics
        out[f"{name}_acc"] = float(accuracy_score(yt, yp))
        out[f"{name}_balanced_acc"] = float(balanced_accuracy_score(yt, yp))
        out[f"{name}_f1_macro"] = float(f1_score(yt, yp, average="macro"))
        out[f"{name}_f1_weighted"] = float(f1_score(yt, yp, average="weighted"))

        # Chance-corrected & correlation-like metrics
        try:
            out[f"{name}_mcc"] = float(matthews_corrcoef(yt, yp))
        except Exception:
            out[f"{name}_mcc"] = float("nan")
        try:
            out[f"{name}_kappa"] = float(cohen_kappa_score(yt, yp))
        except Exception:
            out[f"{name}_kappa"] = float("nan")

        # Top-k accuracy (k=1 equals standard accuracy; k=3 is useful with many classes)
        for k in topk:
            if n_classes >= k:
                try:
                    out[f"{name}_top{k}_acc"] = float(
                        top_k_accuracy_score(yt, yp_prob, k=k, labels=np.arange(n_classes))
                    )
                except Exception:
                    out[f"{name}_top{k}_acc"] = float("nan")

        # Probabilistic metrics (guarded for edge cases)
        try:
            yt_bin = np.eye(n_classes)[yt]
            out[f"{name}_auroc_macro_ovr"] = float(
                roc_auc_score(yt_bin, yp_prob, average="macro", multi_class="ovr")
            )
        except Exception:
            out[f"{name}_auroc_macro_ovr"] = float("nan")

        try:
            yt_bin = np.eye(n_classes)[yt]
            out[f"{name}_auprc_macro"] = float(
                average_precision_score(yt_bin, yp_prob, average="macro")
            )
        except Exception:
            out[f"{name}_auprc_macro"] = float("nan")

        try:
            out[f"{name}_log_loss"] = float(log_loss(yt, yp_prob, labels=np.arange(n_classes)))
        except Exception:
            out[f"{name}_log_loss"] = float("nan")

        # Confusion matrices
        cm = confusion_matrix(yt, yp)
        out[f"{name}_confusion_matrix"] = cm.tolist()
        try:
            cm_norm = confusion_matrix(yt, yp, normalize="true")
            out[f"{name}_confusion_matrix_normalized"] = cm_norm.tolist()
        except Exception:
            pass

        # Per-class breakdown
        try:
            target_names = [idx2label[i] for i in range(n_classes)]
            report = classification_report(
                yt, yp, target_names=target_names, output_dict=True, zero_division=0
            )
            out[f"{name}_classification_report"] = report
        except Exception:
            pass

    return out


def evaluate_similarity_distribution(emb: np.ndarray,
                                     meta_aligned: pd.DataFrame,
                                     cfg: Config,
                                     k_eval: int = 10,
                                     out_dir: str = "artifacts") -> dict:
    """
    For each book embedding, find its top-k_eval neighbors (excluding self),
    compute the average similarity, save results, and plot histogram.
    Returns a dict summary (also written to artifacts/similarity_eval.json).
    """
    ensure_dir(out_dir)
    n = emb.shape[0]
    if n <= 1:
        return {}

    k_use = max(1, min(k_eval, n - 1))
    metric = cfg.similarity

    # kNN on embeddings (fresh, to guarantee we have the top-10 even if cfg.knn < 10)
    nbrs = NearestNeighbors(n_neighbors=k_use + 1, metric=metric, algorithm='auto', n_jobs=-1)
    nbrs.fit(emb)
    distances, indices = nbrs.kneighbors(emb)  # shapes: [n, k_use+1]

    # Convert distance -> similarity
    if metric == "cosine":
        sims = 1.0 - distances
    else:  # euclidean -> soft similarity
        sims = 1.0 / (1.0 + distances)

    # Drop self (first neighbor is the point itself)
    sim_k = sims[:, 1:k_use + 1]        # [n, k_use]
    idx_k = indices[:, 1:k_use + 1]     # [n, k_use]
    avg_sim = sim_k.mean(axis=1)        # [n]

    # Save arrays for reuse
    np.save(os.path.join(out_dir, "avg_sim_top10.npy"), avg_sim)
    np.save(os.path.join(out_dir, "knn10_indices.npy"), idx_k)
    np.save(os.path.join(out_dir, "knn10_similarities.npy"), sim_k)

    # Histogram (bar plot with small bins)
    try:
        import matplotlib.pyplot as plt  # local import to avoid hard dependency elsewhere
        bins = np.linspace(0.0, 1.0, 41)  # 40 small bins
        counts, edges = np.histogram(avg_sim, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        plt.figure(figsize=(8, 4))
        plt.bar(centers, counts, width=(edges[1] - edges[0]), align='center')
        plt.xlabel(f"Average similarity to top-{k_use} neighbors ({metric})")
        plt.ylabel("Number of books")
        plt.title(f"Distribution of average top-{k_use} neighbor similarity")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "similarity_avg_hist.png"), dpi=150)
        plt.close()
    except Exception:
        counts, edges = np.histogram(avg_sim, bins=np.linspace(0.0, 1.0, 41))

    # Example: book with highest average similarity
    best_idx = int(np.argmax(avg_sim))
    best_row = meta_aligned.iloc[best_idx]
    best_title = str(best_row.get("title", f"book_{best_idx}"))
    best_genre = str(best_row.get("genre", "unknown"))
    best_avg = float(avg_sim[best_idx])
    nbrs_idx = idx_k[best_idx].tolist()
    nbrs_sim = sim_k[best_idx].tolist()

    # Build a small table and save it
    ex_rows = []
    for rnk, (j, s) in enumerate(zip(nbrs_idx, nbrs_sim), start=1):
        jr = meta_aligned.iloc[int(j)]
        ex_rows.append({
            "rank": rnk,
            "neighbor_index": int(j),
            "neighbor_book_id": str(jr.get("book_id", "")),
            "neighbor_title": str(jr.get("title", "")),
            "neighbor_genre": str(jr.get("genre", "unknown")),
            "similarity": float(s),
        })
    try:
        pd.DataFrame(ex_rows).to_csv(os.path.join(out_dir, "similarity_example_best.csv"), index=False)
    except Exception:
        pass

    summary = {
        "k": int(k_use),
        "metric": metric,
        "avg_similarity": {
            "mean": float(np.mean(avg_sim)),
            "std": float(np.std(avg_sim)),
            "min": float(np.min(avg_sim)),
            "p25": float(np.percentile(avg_sim, 25)),
            "median": float(np.median(avg_sim)),
            "p75": float(np.percentile(avg_sim, 75)),
            "max": float(np.max(avg_sim)),
        },
        "hist_counts": counts.tolist(),
        "hist_bins": edges.tolist(),
        "best_book": {
            "index": best_idx,
            "book_id": str(best_row.get("book_id", "")),
            "title": best_title,
            "genre": best_genre,
            "avg_sim": best_avg,
        },
    }

    # Save JSON summary
    with open(os.path.join(out_dir, "similarity_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console printout of the example (title + its 10 neighbors' genres)
    print(f"[SIM-EVAL] Book with highest avg similarity: '{best_title}' (genre='{best_genre}', avg={best_avg:.4f})")
    print("[SIM-EVAL] Top neighbors (rank, genre, similarity):")
    for r in ex_rows:
        print(f"  {r['rank']:>2d}. {r['neighbor_genre']:<30s}  sim={r['similarity']:.4f}")

    return summary


def write_gcn_results_table(metrics: dict, num_books: int, num_classes: int, out_dir: str):
    import pandas as pd
    row = {
        "Books": int(num_books),
        "Classes": int(num_classes),
        "Epochs": int(metrics.get("epochs_total", 0)),
        "Best Epoch": int(metrics.get("best_epoch", 0)),
        "Train Loss (best)": float(metrics.get("train_loss_best", float('nan'))),
        "Val Loss (best)": float(metrics.get("val_loss_best", float('nan'))),
        "Val Acc (best)": float(metrics.get("best_val_acc", float('nan'))),
        "Test Acc": float(metrics.get("test_acc", float('nan'))),
        "Test F1": float(metrics.get("test_f1", float('nan'))),
    }
    df = pd.DataFrame([row])
    csv_path = os.path.join(out_dir, "gcn_results.csv")
    md_path  = os.path.join(out_dir, "gcn_results.md")
    df.to_csv(csv_path, index=False)
    try:
        md = df.to_markdown(index=False)  # requires 'tabulate'
    except Exception:
        md = "```\n" + df.to_string(index=False) + "\n```"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)


def main(args=None):
    parser = argparse.ArgumentParser(description="BookNet Pipeline — Big Data Graph Project (Py3.13 friendly)")

    # IO
    parser.add_argument('--books-dir', type=str, default=Config.books_dir)
    parser.add_argument('--metadata-csv', type=str, default=Config.metadata_csv)
    parser.add_argument('--artifacts-dir', type=str, default=Config.artifacts_dir)

    # Graph
    parser.add_argument('--graph-type', type=str, default=Config.graph_type, choices=['token', 'char'])
    parser.add_argument('--window-size', type=int, default=Config.window_size)
    parser.add_argument('--min-freq', type=int, default=Config.min_freq)
    parser.add_argument('--max-nodes', type=int, default=Config.max_nodes)
    parser.add_argument('--lowercase', action='store_true', default=True)
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.add_argument('--strip-punct', action='store_true', default=True)
    parser.add_argument('--no-strip-punct', dest='strip_punct', action='store_false')

    # Graph2Vec-lite
    parser.add_argument('--emb-dim', type=int, default=Config.emb_dim)
    parser.add_argument('--wl-iterations', type=int, default=Config.wl_iterations)

    # Global similarity graph
    parser.add_argument('--knn', type=int, default=Config.knn)
    parser.add_argument('--similarity', type=str, default=Config.similarity, choices=['cosine', 'euclidean'])

    # GCN
    parser.add_argument('--hidden-dim', type=int, default=Config.hidden_dim)
    parser.add_argument('--dropout', type=float, default=Config.dropout)
    parser.add_argument('--lr', type=float, default=Config.lr)
    parser.add_argument('--weight-decay', type=float, default=Config.weight_decay)
    parser.add_argument('--gcn-epochs', type=int, default=Config.gcn_epochs)
    parser.add_argument('--val-size', type=float, default=Config.val_size)
    parser.add_argument('--test-size', type=float, default=Config.test_size)
    parser.add_argument('--seed', type=int, default=Config.seed)

    # Steps (build pipeline stages)
    parser.add_argument('--build-graphs', action='store_true',
                        help='Build per-book graphs from text files')
    parser.add_argument('--embed', action='store_true',
                        help='Compute Graph2Vec-lite embeddings for each book-graph')
    parser.add_argument('--build-global', action='store_true',
                        help='Construct the global k-NN similarity graph over book embeddings')
    parser.add_argument('--train-gcn', action='store_true',
                        help='Train a GCN for node classification (e.g., by genre)')

    # Caching
    parser.add_argument('--save-graphs', action='store_true', default=True)
    parser.add_argument('--no-save-graphs', dest='save_graphs', action='store_false')
    parser.add_argument('--load-graphs', action='store_true', default=False)

    # Download (Gutenberg with subjects/genres)
    parser.add_argument('--download-gutenberg', action='store_true',
                        help='Download Gutenberg (with subjects) and create metadata.csv')
    parser.add_argument('--limit', type=int, default=0,
                        help='Stop after N books when downloading (0 = all)')
    parser.add_argument('--download-storyset', action='store_true',
                        help='Download StorySet (full text + genre) and create metadata.csv')

    # ---- parse args ----
    parsed = parser.parse_args(args=args)

    # If no step flags (and no download/load) were given, run full pipeline by default (handy in PyCharm)
    if not any([
        getattr(parsed, 'build_graphs', False),
        getattr(parsed, 'embed', False),
        getattr(parsed, 'build_global', False),
        getattr(parsed, 'train_gcn', False),
        getattr(parsed, 'download_gutenberg', False),
        getattr(parsed, 'download_storyset', False),  # <-- add this
        getattr(parsed, 'download_pg19', False),  # <-- if you still have pg19
        getattr(parsed, 'load_graphs', False),
    ]):
        print("[INFO] No flags given; running full pipeline by default.", flush=True)
        parsed.build_graphs = True
        parsed.embed = True
        parsed.build_global = True
        parsed.train_gcn = True

    # Optionally download Gutenberg (with labels) first
    if parsed.download_gutenberg:
        print("[INFO] Downloading Gutenberg with subjects…", flush=True)
        download_gutenberg_with_genres(out_books=parsed.books_dir,
                                       out_meta=parsed.metadata_csv,
                                       limit=parsed.limit)

    if parsed.download_storyset:
        print("[INFO] Downloading StorySet (with explicit genres)…", flush=True)
        download_storyset_with_genres(out_books=parsed.books_dir,
                                      out_meta=parsed.metadata_csv,
                                      limit=parsed.limit)

    # Build config object
    cfg = Config(
        books_dir=parsed.books_dir,
        metadata_csv=parsed.metadata_csv,
        artifacts_dir=parsed.artifacts_dir,
        graph_type=parsed.graph_type,
        window_size=parsed.window_size,
        min_freq=parsed.min_freq,
        max_nodes=parsed.max_nodes,
        lowercase=parsed.lowercase,
        strip_punct=parsed.strip_punct,
        emb_dim=parsed.emb_dim,
        wl_iterations=parsed.wl_iterations,
        knn=parsed.knn,
        similarity=parsed.similarity,
        hidden_dim=parsed.hidden_dim,
        dropout=parsed.dropout,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        gcn_epochs=parsed.gcn_epochs,
        val_size=parsed.val_size,
        test_size=parsed.test_size,
        seed=parsed.seed,
        save_graphs=parsed.save_graphs,
        load_graphs=parsed.load_graphs,
    )

    # ---- pipeline starts here ----
    seed_everything(cfg.seed)
    ensure_dir(cfg.artifacts_dir)

    stats = StatsCollector(cfg.artifacts_dir)

    # 1) Discover & load metadata
    book_paths = discover_books(cfg.books_dir)
    if len(book_paths) == 0 and not cfg.load_graphs:
        print(f"[WARN] No .txt files found under {cfg.books_dir}. "
              f"Use --download-storyset or --download-gutenberg, or place files in data/books/", flush=True)
        return
    meta = load_metadata(cfg, book_paths)
    print(f"Loaded metadata for {len(meta)} books.", flush=True)
    stats.add("dataset", summarize_metadata(meta, cfg.books_dir))
    stats.add("labels", summarize_labels(meta))

    if 'genre' in meta.columns:
        vc = meta['genre'].astype(str).value_counts()
        print("Genre distribution (top 10):")
        print(vc.head(10).to_string(), flush=True)

    # 2) Build per-book graphs
    if cfg.load_graphs:
        graphs, ids = load_graphs(cfg)
        print(f"Loaded {len(graphs)} cached graphs from artifacts.", flush=True)
    else:
        needs_graphs = parsed.build_graphs or parsed.embed or parsed.build_global or parsed.train_gcn
        if needs_graphs:
            stats.start("build_graphs")
            graphs, ids = build_graphs(cfg, meta)
            stats.stop("build_graphs")
            stats.add("per_book_graphs", summarize_graphs(graphs))
            if cfg.save_graphs:
                save_graphs(graphs, ids, cfg)
        else:
            graphs, ids = [], []

    # Subset metadata to the graphs we actually kept
    if len(ids) > 0:
        meta = meta[meta['book_id'].isin(ids)].reset_index(drop=True)
        meta_aligned = meta.set_index('book_id').loc[ids].reset_index()
    else:
        meta_aligned = meta.copy()

    # 3) Embeddings
    emb_path = os.path.join(cfg.artifacts_dir, "emb.npy")
    need_embed = parsed.embed or (not os.path.exists(emb_path))

    if need_embed:
        if len(graphs) == 0:
            raise SystemExit("No graphs to embed. Run with --build-graphs (or --load-graphs).")
        stats.start("embed_graph2vec")
        # If you added the faster params earlier, pass them here:
        # emb, emb_info = graph2vec_embeddings_sklearn(
        #     graphs, dim=cfg.emb_dim, wl_iterations=cfg.wl_iterations, random_state=cfg.seed,
        #     return_info=True, hv_n_features=cfg.hv_n_features, svd_n_iter=cfg.svd_n_iter
        # )
        emb, emb_info = graph2vec_embeddings_sklearn(
            graphs, dim=cfg.emb_dim, wl_iterations=cfg.wl_iterations, random_state=cfg.seed, return_info=True
        )
        stats.stop("embed_graph2vec")
        np.save(emb_path, emb)
        print(
            f"[EMB] Computed: {emb.shape} • SVD cumulative explained variance: {emb_info['svd_cumulative_explained']:.3f}",
            flush=True)
        stats.add("embeddings", summarize_embeddings(emb, emb_info))
    else:
        emb = np.load(emb_path)
        print(f"[EMB] Loaded cached embeddings: {emb.shape} from {emb_path}", flush=True)

    # 4) Global similarity graph
    edge_idx_path = os.path.join(cfg.artifacts_dir, "edge_index.npy")
    edge_w_path = os.path.join(cfg.artifacts_dir, "edge_weight.npy")
    need_knn = parsed.build_global or (not (os.path.exists(edge_idx_path) and os.path.exists(edge_w_path)))

    if parsed.build_global or parsed.train_gcn:
        if emb is None:
            raise SystemExit("No embeddings available. Run with --embed or provide artifacts/emb.npy")

        if need_knn:
            stats.start("build_global_knn_graph")
            edge_index, edge_weight = build_global_graph(emb, cfg)
            stats.stop("build_global_knn_graph")
            np.save(edge_idx_path, edge_index)
            np.save(edge_w_path, edge_weight)
            print(f"[KNN] Built global graph: {edge_index.shape[1]} edges", flush=True)
        else:
            edge_index = np.load(edge_idx_path)
            edge_weight = np.load(edge_w_path)
            print(f"[KNN] Loaded cached global graph: {edge_index.shape[1]} edges", flush=True)

        stats.add("global_similarity_graph", summarize_global_graph(edge_index, edge_weight, emb.shape[0]))

        # --- Similarity evaluation & histogram over top-10 neighbors ---
        sim_eval = evaluate_similarity_distribution(
            emb=emb,
            meta_aligned=meta_aligned,
            cfg=cfg,
            k_eval=10,
            out_dir=cfg.artifacts_dir
        )
        # Keep a compact subset in the stats file
        stats.add("similarity_eval", {
            "k": sim_eval.get("k"),
            "metric": sim_eval.get("metric"),
            "avg_similarity_mean": sim_eval.get("avg_similarity", {}).get("mean"),
            "avg_similarity_std": sim_eval.get("avg_similarity", {}).get("std"),
            "best_book_id": sim_eval.get("best_book", {}).get("book_id"),
            "best_title": sim_eval.get("best_book", {}).get("title"),
            "best_genre": sim_eval.get("best_book", {}).get("genre"),
            "best_avg_sim": sim_eval.get("best_book", {}).get("avg_sim"),
        })

    else:
        edge_index = edge_weight = None

    # 5) Node classification with GCN (skip if only one genre present)
    if parsed.train_gcn:
        if edge_index is None:
            raise SystemExit("Global graph missing. Run with --build-global")

        genres = meta['genre'].astype(str).tolist()
        y, idx2label = encode_labels(genres)

        if len(np.unique(y)) < 2:
            print("[WARN] Only one genre present; skipping GCN classification. "
                  "Download more data or broaden labels.", flush=True)
            # Still run simple link prediction as an unsupervised sanity check
            data = assemble_pyg_data(
                emb, edge_index, edge_weight, y,
                np.zeros_like(y, dtype=bool),
                np.zeros_like(y, dtype=bool),
                np.zeros_like(y, dtype=bool)
            )
            auc = link_prediction_auc(data)
            with open(os.path.join(cfg.artifacts_dir, "metrics.json"), "w") as f:
                json.dump({"lp_auc_raw_emb": float(auc)}, f, indent=2)
            print(f"Simple link prediction AUC (dot-product, raw emb): {auc:.4f}", flush=True)
            print("Done.", flush=True)
            return

        # Train/val/test split
        train_mask, val_mask, test_mask = make_splits(y, cfg.val_size, cfg.test_size, cfg.seed)

        # Assemble PyG Data
        data = assemble_pyg_data(emb, edge_index, edge_weight, y, train_mask, val_mask, test_mask)
        torch.save(data, os.path.join(cfg.artifacts_dir, "pyg_graph.pt"))

        # Train GCN
        stats.start("train_gcn")
        metrics, model = train_gcn(data, cfg, idx2label, cfg.artifacts_dir)
        stats.stop("train_gcn")
        with open(os.path.join(cfg.artifacts_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        torch.save(model.state_dict(), os.path.join(cfg.artifacts_dir, "gcn_best.pt"))
        print("Final metrics:", metrics, flush=True)

        write_gcn_results_table(metrics, emb.shape[0], len(idx2label), cfg.artifacts_dir)
        print("[GCN] Wrote table → artifacts/gcn_results.csv and artifacts/gcn_results.md", flush=True)

        # Classifier diagnostics
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_weight)
        clf_stats = summarize_classifier(data, logits, idx2label)
        stats.add("classifier", clf_stats)

        # Optional: simple link prediction baseline
        lp_auc = link_prediction_auc(data)
        metrics['lp_auc_raw_emb'] = float(lp_auc)
        with open(os.path.join(cfg.artifacts_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Simple link prediction AUC (dot-product, raw emb): {lp_auc:.4f}", flush=True)
        stats.add("link_prediction_baseline", {"lp_auc_raw_emb": float(lp_auc)})

    stats.write()
    print(f"[STATS] Wrote artifacts/stats.json and artifacts/report.md", flush=True)

    print("Done.", flush=True)



if __name__ == "__main__":
    import sys
    # If you just click Run in PyCharm with no args, do the full pipeline by default
    args = sys.argv[1:]
    if not args:
        print("[INFO] No flags given; running full pipeline by default.", flush=True)
        args = ["--build-graphs", "--embed", "--build-global", "--train-gcn"]
    main(args)
