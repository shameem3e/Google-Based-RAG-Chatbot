# main.py
import argparse
from google_search import scrape_google, load_cached_results
from embed_store import build_embeddings_and_index
from retriever import Retriever
from summarizer import Summarizer
from chat import interactive_chat

def ingest(query: str = None, num_results: int = 5, reindex: bool = True):
    """
    If query provided: search web for that query and cache results.
    Then build embeddings & index.
    """
    if query:
        print("[main] Scraping web ...")
        scrape_google(query, num_results=num_results, cache=True)
    print("[main] Building embeddings & index ...")
    build_embeddings_and_index(reindex=reindex)
    print("[main] Ingest complete.")

def query_once(question: str, top_k: int = 5):
    retr = Retriever()
    hits = retr.retrieve(question, top_k=top_k)
    contexts = [h["text"] for h in hits]
    summarizer = Summarizer()
    ans = summarizer.summarize(question, contexts)
    print("\n=== Answer ===\n")
    print(ans)
    print("\n=== Sources ===")
    for h in hits:
        print(f" - {h['source_title'] or h['source_url']} (score {h['score']:.3f})")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Scrape web (if query provided) and build index")
    p_ingest.add_argument("--query", "-q", type=str, default=None, help="Query to search & cache before ingest")
    p_ingest.add_argument("--num_results", type=int, default=5)

    p_query = sub.add_parser("query", help="Make a single query (uses existing index)")
    p_query.add_argument("--question", "-q", required=True)
    p_query.add_argument("--top_k", type=int, default=5)

    sub.add_parser("chat", help="Interactive chat (terminal)")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(query=args.query, num_results=args.num_results)
    elif args.cmd == "query":
        query_once(args.question, top_k=args.top_k)
    elif args.cmd == "chat":
        interactive_chat()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
