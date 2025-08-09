# embed_store.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from text_utils import chunk_text, clean_text

# faiss/sklearn import with fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "search_results.json"
META_FILE = DATA_DIR / "chunks_metadata.json"
INDEX_FILE = DATA_DIR / "faiss_index.bin"
EMB_FILE = DATA_DIR / "embeddings.npy"

EMBED_MODEL = "all-MiniLM-L6-v2"

def build_embeddings_and_index(embed_model_name: str = EMBED_MODEL,
                               chunk_size: int = 700, overlap: int = 140,
                               reindex: bool = False):
    """
    Reads cached search results, chunks text, builds embeddings, and persists:
    - metadata (list of dicts with full chunk text and source info)
    - embeddings (numpy)
    - index (FAISS or sklearn NearestNeighbors saved)
    """
    if not CACHE_FILE.exists():
        raise FileNotFoundError("No cached search results found. Run google_search.scrape_google first.")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    metadata = []
    print("[embed_store] Chunking pages ...")
    for p in pages:
        text = clean_text(p.get("text", ""))
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(chunks):
            metadata.append({
                "source_url": p.get("url"),
                "source_title": p.get("title"),
                "chunk_index": i,
                "text": ch
            })
            all_chunks.append(ch)

    if not all_chunks:
        raise ValueError("No chunks extracted from pages.")

    # Embed
    print(f"[embed_store] Loading embedder '{embed_model_name}' ...")
    model = SentenceTransformer(embed_model_name)
    embeddings = model.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
    # normalize for cosine similarity with inner-product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-12)
    embeddings = embeddings.astype("float32")

    # Save metadata & embeddings
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    np.save(EMB_FILE, embeddings)

    # Build index
    if FAISS_AVAILABLE:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors ~ cosine
        index.add(embeddings)
        faiss.write_index(index, str(INDEX_FILE))
        print(f"[embed_store] FAISS index saved to {INDEX_FILE}")
    else:
        # sklearn fallback: persist embeddings; we'll load and build NearestNeighbors at runtime
        print("[embed_store] FAISS not available — embeddings saved and sklearn fallback will be used.")
    print(f"[embed_store] Done. {len(all_chunks)} chunks indexed.")
