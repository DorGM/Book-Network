# app_similar_books.py
import os
import difflib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


@st.cache_data(show_spinner=False)
def pca_2d(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=random_state).fit_transform(X)

@st.cache_data(show_spinner=False)
def umap_2d(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    try:
        import umap
    except Exception:
        return None
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(X)

@st.cache_data(show_spinner=False)
def knn_edges_and_degree(emb_norm: np.ndarray, k: int):
    n = emb_norm.shape[0]
    k = max(1, min(k, n-1))
    nbrs = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(emb_norm)
    _, idxs = nbrs.kneighbors(emb_norm)
    edges = set()
    deg = np.zeros(n, dtype=int)
    for i in range(n):
        for j in idxs[i][1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if (a, b) not in edges:
                edges.add((a, b))
                deg[a] += 1
                deg[b] += 1
    return edges, deg

@st.cache_data(show_spinner=False)
def kmeans_labels(X: np.ndarray, k: int, random_state: int = 42):
    k = max(2, min(k, X.shape[0]-1))
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if X.shape[0] > k else float("nan")
    return labels, sil

ART_DIR = "artifacts"
DATA_DIR = "data"
EMB_PATH = os.path.join(ART_DIR, "emb.npy")
IDMAP_PATH = os.path.join(ART_DIR, "id_map.csv")
META_PATH = os.path.join(DATA_DIR, "metadata.csv")

@st.cache_data(show_spinner=False)
def load_data(emb_path: str, idmap_path: str, meta_path: str):
    # Load embeddings
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embeddings at {emb_path}. Run your pipeline to create emb.npy.")

    emb = np.load(emb_path)
    n, d = emb.shape

    # Load id_map (row order for embeddings)
    if not os.path.exists(idmap_path):
        raise FileNotFoundError(f"Missing id_map at {idmap_path}. It is created when graphs are built.")
    idmap = pd.read_csv(idmap_path)
    if "book_id" not in idmap.columns:
        raise ValueError("id_map.csv must contain a 'book_id' column.")
    id_order = idmap["book_id"].astype(str).tolist()

    if len(id_order) != n:
        # We can still proceed if metadata can be aligned by book_id
        st.warning(
            f"id_map length ({len(id_order)}) != embeddings rows ({n}). "
            "Will try to align with metadata by book_id."
        )

    # Load metadata and align it to embedding order
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata at {meta_path}.")
    meta = pd.read_csv(meta_path)
    if "book_id" not in meta.columns or "title" not in meta.columns:
        raise ValueError("metadata.csv must contain at least 'book_id' and 'title' columns.")

    meta["book_id"] = meta["book_id"].astype(str)
    # Keep only books we have embeddings for, and order them by id_order
    meta_on_emb = meta[meta["book_id"].isin(id_order)].copy()
    meta_on_emb.index = pd.CategoricalIndex(meta_on_emb["book_id"], categories=id_order, ordered=True)
    meta_on_emb = meta_on_emb.sort_index()
    # Now meta_on_emb length should match emb rows; if not, we trim/pad carefully
    if len(meta_on_emb) != n:
        # Trim to min len
        m = min(len(meta_on_emb), n)
        meta_on_emb = meta_on_emb.iloc[:m].reset_index(drop=True)
        emb = emb[:m, :]
        n = m
        st.warning(f"Aligned metadata to embeddings: using first {m} rows.")

    # Build nice display name (disambiguate duplicates)
    cols = meta_on_emb.columns
    have_author = "author" in cols
    have_genre = "genre" in cols

    def make_display(row):
        return str(row.get("title", row["book_id"]))

    meta_on_emb["display"] = meta_on_emb.apply(make_display, axis=1)

    # IMPORTANT: ensure a clean integer index to avoid showing file/ids as an index later
    meta_on_emb = meta_on_emb.reset_index(drop=True)

    # Normalize embeddings once for cosine similarity
    emb_norm = normalize(emb, norm="l2", axis=1)
    return emb, emb_norm, meta_on_emb

def cosine_topk(emb_norm: np.ndarray, idx: int, k: int):
    # cos sim = dot on L2-normalized vectors
    sims = emb_norm @ emb_norm[idx]  # shape [N]
    sims[idx] = -1.0  # exclude self
    # argpartition for speed, then sort
    k = int(k)
    k = max(1, min(k, emb_norm.shape[0] - 1))
    top_idx = np.argpartition(-sims, k)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx, sims[top_idx]

def main():
    st.set_page_config(page_title="Similar Books Finder", layout="wide")
    st.title("🔎 Book Similarity Search")
    st.caption("Type a book title or pick from the list, choose **k**, and get the most similar books by embedding.")

    # Load data
    try:
        emb, emb_norm, meta = load_data(EMB_PATH, IDMAP_PATH, META_PATH)
    except Exception as e:
        st.error(str(e))
        st.stop()

    N = emb.shape[0]
    st.write(f"**Books in index:** {N} · **Embedding dim:** {emb.shape[1]}")

    # Sidebar: k selection (informative + dynamic)
    with st.sidebar:
        st.header("Recommendation Size")

        max_k = max(1, N - 1)  # valid range is 1..N-1 (if N==1, clamp to 1 to keep the widget valid)
        default_k = min(10, max_k)

        k = st.number_input(
            label="k (top-N recommendations)",
            min_value=1,
            max_value=max_k,
            value=default_k,
            step=1,
            help=f"Select how many similar books to recommend."
        )

        st.caption(
            f"Choose how many similar books you’d like recommended. "
            f"Enter a number between **1** and **{max_k}**. "
            f"The system currently contains **{N}** book{'s' if N != 1 else ''}."
        )

    st.subheader("Pick a book")
    left, right = st.columns([1, 1])

    # --- Session state defaults ---
    if "title_query" not in st.session_state:
        st.session_state.title_query = ""
    if "dropdown_choice" not in st.session_state:
        st.session_state.dropdown_choice = "—"

    # --- Callbacks to enforce mutual exclusivity ---
    def _on_text_change():
        st.session_state.dropdown_choice = "—"  # user typed → disable dropdown

    def _on_dropdown_change():
        st.session_state.title_query = ""  # user picked from list → disable text

    # Disabled states
    text_disabled = st.session_state.dropdown_choice != "—"
    dropdown_disabled = bool(st.session_state.title_query.strip())

    chosen_idx = None

    # A) Free text search with prefix suggestions (titles only)
    with left:
        st.markdown("**Search by title**")
        title_query = st.text_input(
            "Start typing…",
            placeholder="e.g., The Adventures",
            key="title_query",
            disabled=text_disabled,
            on_change=_on_text_change,
        )

        if not text_disabled and title_query.strip():
            q = title_query.strip().lower()
            title_lc = meta["title"].astype(str).str.lower()

            # 1) Prefix suggestions (titles that START with the query)
            pref_mask = title_lc.str.startswith(q)
            pref_idxs = np.where(pref_mask)[0][:20]

            if len(pref_idxs) > 0:
                options = [(int(i), meta.iloc[i]["title"]) for i in pref_idxs]
                sel = st.selectbox(
                    "Suggestions:",
                    options,
                    format_func=lambda x: x[1] if isinstance(x, tuple) else x,
                    key="prefix_select",
                )
                if isinstance(sel, tuple):
                    chosen_idx = int(sel[0])
            else:
                # 2) Exact match fallback (case-insensitive)
                exact_matches = np.where(title_lc == q)[0]
                if len(exact_matches) == 1:
                    chosen_idx = int(exact_matches[0])
                    st.success(f"Matched “{meta.iloc[chosen_idx]['title']}”.")
                elif len(exact_matches) > 1:
                    st.info("Multiple books share this title — please choose the exact one:")
                    options = [(int(i), meta.iloc[i]["title"]) for i in exact_matches]
                    sel = st.selectbox("Disambiguate:", options,
                                       format_func=lambda x: x[1] if isinstance(x, tuple) else x,
                                       key="exact_select")
                    if isinstance(sel, tuple):
                        chosen_idx = int(sel[0])
                else:
                    # 3) Fuzzy hints
                    suggestions = difflib.get_close_matches(
                        title_query.strip(),
                        meta["title"].astype(str).tolist(),
                        n=5, cutoff=0.6
                    )
                    st.error("Title not found.")
                    if suggestions:
                        st.write("Did you mean:")
                        for s in suggestions:
                            st.write(f"- {s}")

    # B) Dropdown chooser (titles only)
    with right:
        st.markdown("**Or choose from the list**")
        options = ["—"] + [(i, meta.iloc[i]["title"]) for i in range(len(meta))]
        selection = st.selectbox(
            "Pick a book",
            options,
            key="dropdown_choice",
            disabled=dropdown_disabled,
            format_func=lambda x: x if isinstance(x, str) else x[1],
        )
        if not dropdown_disabled and selection != "—" and isinstance(selection, tuple):
            chosen_idx = int(selection[0])

    # Clear selection button (callback)
    def _clear_selection():
        st.session_state.title_query = ""
        st.session_state.dropdown_choice = "—"
        st.session_state.pop("prefix_select", None)
        st.session_state.pop("exact_select", None)

    clear_cols = st.columns([1, 1])
    with clear_cols[0]:
        st.button("Clear selection", on_click=_clear_selection)

    st.divider()

    # === Action button ===
    disable_btn = (chosen_idx is None) or (N < 2)
    if st.button("Find similar books", type="primary", disabled=disable_btn):
        if chosen_idx is None:
            st.warning("Please choose a book first.")
        else:
            # Validate k in [1, N-1]
            kval = int(k)
            if not (1 <= kval <= N - 1):
                st.error(f"k must be between 1 and {N - 1}. There are currently {N} books in the system.")
                st.stop()

            top_idx, top_sims = cosine_topk(emb_norm, chosen_idx, kval)

            # Build results table: title + genre + similarity (NO author, NO file/id index)
            res = meta.iloc[top_idx][["title"]].copy()
            if "genre" in meta.columns:
                res["genre"] = meta.iloc[top_idx]["genre"].values
            res.insert(0, "rank", np.arange(1, len(top_idx) + 1))
            res["similarity"] = np.round(top_sims, 4)

            # Ensure no hidden index with file names/ids appears
            res = res.reset_index(drop=True)

            st.subheader(f"Top {kval} similar to: {meta.iloc[chosen_idx]['title']}")
            st.dataframe(res.set_index("rank"), use_container_width=True)

            # Download CSV (same visible columns)
            csv = res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                data=csv,
                file_name=f"similar_to_{meta.iloc[chosen_idx]['title']}.csv",
                mime="text/csv",
            )

    st.header("📊 Insights & Graphs")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Genre distribution",
        "Embedding map",
        "Similarity profile",
        "Degree distribution",
        "Clusters vs. Genres",
    ])

    # 1) Genre distribution
    with tab1:
        st.subheader("Genre distribution")
        if "genre" in meta.columns:
            counts = meta["genre"].astype(str).value_counts()
            st.write(f"Unique genres: **{counts.shape[0]}**")
            fig, ax = plt.subplots()
            counts.head(20).plot(kind="bar", ax=ax)  # top 20 for readability
            ax.set_xlabel("Genre")
            ax.set_ylabel("Count")
            ax.set_title("Top genres (by count)")
            ax.tick_params(axis='x', labelrotation=45)
            for lbl in ax.get_xticklabels():
                lbl.set_horizontalalignment('right')  # or: lbl.set_ha('right')
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            with st.expander("Full genre table"):
                st.dataframe(counts.rename("count").to_frame())
        else:
            st.info("No 'genre' column found in metadata.")

    # 2) Embedding map (2-D)
    with tab2:
        st.subheader("Embedding map (2-D projection)")
        use_umap = st.checkbox("Use UMAP (if installed)", value=True)
        if use_umap:
            emb2d = umap_2d(emb_norm)  # None if umap-learn not installed
            if emb2d is None:
                st.warning("UMAP not installed. Falling back to PCA.")
                emb2d = pca_2d(emb)
        else:
            emb2d = pca_2d(emb)

        # choose coloring
        color_by = "genre" if "genre" in meta.columns else None
        fig, ax = plt.subplots()
        if color_by:
            for g, sub in meta.groupby(color_by):
                idxs = sub.index.values
                ax.scatter(emb2d[idxs, 0], emb2d[idxs, 1], s=10, label=str(g))
            ax.legend(title=color_by, loc="best", fontsize="small")
        else:
            ax.scatter(emb2d[:, 0], emb2d[:, 1], s=10)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Books projected to 2-D")
        st.pyplot(fig, clear_figure=True)

    # 3) Similarity profile for a chosen book
    with tab3:
        st.subheader("Similarity profile")
        if len(meta) < 2:
            st.info("Need at least 2 books.")
        else:
            pick_title = st.selectbox("Choose a book", meta["title"].tolist(), index=0, key="sim_profile_title")
            idx = int(meta.index[meta["title"] == pick_title][0])
            k_top = st.slider("How many neighbors to display", min_value=1, max_value=min(30, len(meta) - 1), value=10,
                              step=1)
            top_idx, top_sims = cosine_topk(emb_norm, idx, k_top)

            prof = meta.iloc[top_idx][["title"]].copy()
            if "genre" in meta.columns:
                prof["genre"] = meta.iloc[top_idx]["genre"].values
            prof.insert(0, "rank", np.arange(1, len(top_idx) + 1))
            prof["similarity"] = np.round(top_sims, 4)
            prof = prof.reset_index(drop=True)

            # Bar plot
            fig, ax = plt.subplots()
            ax.barh(prof["title"][::-1], prof["similarity"][::-1])
            ax.set_xlabel("Cosine similarity")
            ax.set_title(f"Top-{k_top} similar to “{pick_title}”")
            st.pyplot(fig, clear_figure=True)

            st.dataframe(prof.set_index("rank"), use_container_width=True)

    # 4) Degree distribution of the k-NN graph
    with tab4:
        st.subheader("Degree distribution of k-NN graph")
        if len(meta) < 2:
            st.info("Need at least 2 books.")
        else:
            k_graph = st.slider("k for k-NN graph", min_value=1, max_value=min(50, len(meta) - 1), value=10, step=1)
            _, deg = knn_edges_and_degree(emb_norm, k_graph)
            fig, ax = plt.subplots()
            ax.hist(deg, bins=min(30, deg.max() - deg.min() + 1))
            ax.set_xlabel("Degree")
            ax.set_ylabel("Number of books")
            ax.set_title(f"Degree distribution (k={k_graph})")
            st.pyplot(fig, clear_figure=True)
            st.write(f"Average degree: **{np.mean(deg):.2f}** · Min: **{deg.min()}** · Max: **{deg.max()}**")

    # 5) Clusters vs. Genres
    with tab5:
        st.subheader("K-means clustering on embeddings")
        if len(meta) < 3:
            st.info("Need at least 3 books.")
        else:
            k_clu = st.slider("Number of clusters (k-means)", min_value=2, max_value=min(20, len(meta) - 1), value=6,
                              step=1)
            labels, sil = kmeans_labels(emb, k_clu)

            # Cluster sizes
            sizes = pd.Series(labels).value_counts().sort_index()
            fig, ax = plt.subplots()
            sizes.plot(kind="bar", ax=ax)
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Count")
            ax.set_title(f"Cluster sizes (k={k_clu})  •  Silhouette={sil:.3f}")
            st.pyplot(fig, clear_figure=True)

            # Contingency: clusters vs. genres
            if "genre" in meta.columns:
                crosstab = pd.crosstab(pd.Series(labels, name="cluster"), meta["genre"].astype(str), normalize="index")
                st.write("Share of genres within each cluster (row-normalized):")
                st.dataframe((crosstab * 100).round(1))
            else:
                st.info("No 'genre' column to compare with clusters.")

    # Footer info
    with st.expander("Data sources & expectations"):
        st.write(
            "- Embeddings are loaded from `artifacts/emb.npy` and assumed to align with `artifacts/id_map.csv`.\n"
            "- Titles/authors/genres are read from `data/metadata.csv` and aligned by `book_id`."
        )

if __name__ == "__main__":
    main()
