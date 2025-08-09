# chat.py
from google_search import scrape_google, load_cached_results
from embed_store import build_embeddings_and_index
from retriever import Retriever
from summarizer import Summarizer
import os

def interactive_chat(reindex_on_search: bool = False):
    """
    Terminal chat loop.
    If reindex_on_search=True, each new query will also trigger a fresh scrape->index (costly).
    Otherwise it uses cached pages/index (use 'ingest' command to refresh).
    """
    print("Welcome to RAG Chat (type 'exit' or 'quit' to leave).")
    retriever = None
    summarizer = Summarizer()
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye.")
            break

        # Ask the user whether to run a fresh search for this query or use existing index
        ask_search = input("Search web for this query now? (y/N): ").strip().lower()
        if ask_search == "y":
            # Scrape & reindex
            print("[chat] Scraping web (this may take a while)...")
            scrape_google(q, num_results=5, cache=True)
            print("[chat] Building embeddings & index ...")
            build_embeddings_and_index(reindex=True)
        else:
            # If index doesn't exist, build using cached or prompt user
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            from pathlib import Path
            if not (Path(data_dir) / "faiss_index.bin").exists() and not (Path(data_dir) / "embeddings.npy").exists():
                print("[chat] No index found. Running a one-time ingestion from cached search results (or run a search).")
                build_embeddings_and_index(reindex=True)

        # lazy initialize retriever after index exists
        if retriever is None:
            retriever = Retriever()
        hits = retriever.retrieve(q, top_k=4)
        contexts = [h["text"] for h in hits]
        if not contexts:
            print("No retrieved context found. Try running a search (y) option next time.")
            continue
        answer = summarizer.summarize(q, contexts)
        print("\nAssistant:", answer)
        print("\nSources:")
        for h in hits:
            print(f" - {h['source_title'] or h['source_url']} (score {h['score']:.3f})")
