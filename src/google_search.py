# google_search.py
# Search & scrape top pages using googlesearch + requests + BeautifulSoup
# Note: scraping may violate site ToS; prefer Google Custom Search API for production.

from googlesearch import search
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "search_results.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def fetch_page_text(url: str, timeout: int = 8) -> Dict:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join([title] + paragraphs).strip()
        return {"url": url, "title": title, "text": text}
    except Exception:
        return {"url": url, "title": "", "text": ""}

def scrape_google(query: str, num_results: int = 5, cache: bool = True) -> List[Dict]:
    """
    Search Google and scrape top `num_results` pages.
    Saves results to data/search_results.json if cache=True.
    """
    results = []
    print(f"[google_search] Searching for: {query} (top {num_results})")
    urls = list(search(query, num_results=num_results))
    for url in tqdm(urls, total=len(urls)):
        page = fetch_page_text(url)
        if page["text"]:
            results.append(page)
    if cache:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results

def load_cached_results():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
