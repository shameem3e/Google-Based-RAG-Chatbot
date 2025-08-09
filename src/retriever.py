# retriever.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False
    from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
META_FILE = DATA_DIR / "chunks_metadata.json"
INDEX_FILE = DATA_DIR / "faiss_index.bin"
EMB_FILE = DATA_DIR / "embeddings.npy"

EMBED_MODEL = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, embed_model_name: str = EMBED_MODEL):
        if not META_FILE.exists() or not EMB_FILE.exists():
            raise FileNotFoundError("Missing metadata or embeddings. Run embed_store.build_embeddings_and_index first.")
        self.embedder = SentenceTransformer(embed_model_name)
        with open(META_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.embeddings = np.load(EMB_FILE)
        if FAISS_AVAILABLE and INDEX_FILE.exists():
            self.index = faiss.read_index(str(INDEX_FILE))
            self.use_faiss = True
        else:
            # sklearn nearest neighbor instance will be created on demand
            self.nn = None
            self.use_faiss = False

    def _ensure_nn(self):
        if self.nn is None:
            from sklearn.neighbors import NearestNeighbors
            self.nn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
        q_emb = q_emb.astype("float32")
        results = []
        if self.use_faiss:
            D, I = self.index.search(q_emb, top_k)
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                meta = self.metadata[int(idx)]
                results.append({
                    "score": float(score),
                    "source_url": meta.get("source_url"),
                    "source_title": meta.get("source_title"),
                    "text": meta.get("text")
                })
        else:
            self._ensure_nn()
            distances, indices = self.nn.kneighbors(q_emb, n_neighbors=top_k)
            for dist, idx in zip(distances[0], indices[0]):
                meta = self.metadata[int(idx)]
                # sklearn returns cosine distance; convert to similarity (approx)
                results.append({
                    "score": float(1 - dist),
                    "source_url": meta.get("source_url"),
                    "source_title": meta.get("source_title"),
                    "text": meta.get("text")
                })
        return results
