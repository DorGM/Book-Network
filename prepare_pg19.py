import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser("Prepare PG-19 (Project Gutenberg subset) for BookNet pipeline")
    ap.add_argument("--out-books", type=str, default="data/books", help="Folder to write .txt files")
    ap.add_argument("--out-meta", type=str, default="data/metadata.csv", help="Path to write metadata CSV")
    ap.add_argument("--limit", type=int, default=0, help="Stop after this many books (0 = all)")
    args = ap.parse_args()

    ensure_dir(args.out_books)
    ensure_dir(os.path.dirname(args.out_meta) or ".")

    # Load all splits; PG-19 provides 'train', 'validation', 'test'
    ds = load_dataset("deepmind/pg19")

    rows = []
    total = 0
    for split in ("train", "validation", "test"):
        if split not in ds:
            continue
        d = ds[split]
        for ex in tqdm(d, desc=f"Writing {split}", total=len(d)):
            book_id = str(ex["book_id"])
            text = ex["text"]
            fname = f"{book_id}.txt"
            fpath = os.path.join(args.out_books, fname)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)

            # Title/author/genre are not included as rich labels; keep genre unknown (you can augment later)
            rows.append({
                "book_id": book_id,
                "path": os.path.relpath(fpath, start="data/books"),
                "title": book_id,        # fallback (PG-19 has a metadata.csv in repo; not exposed directly here)
                "author": "unknown",
                "genre": "unknown"
            })

            total += 1
            if args.limit and total >= args.limit:
                break
        if args.limit and total >= args.limit:
            break

    pd.DataFrame(rows).to_csv(args.out_meta, index=False)
    print(f"[OK] Wrote {total} books to {args.out_books}")
    print(f"[OK] Wrote metadata to {args.out_meta}")

if __name__ == "__main__":
    main()
