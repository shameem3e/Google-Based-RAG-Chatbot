# text_utils.py
import re
from typing import List

def clean_text(text: str) -> str:
    """
    Basic cleaning: normalize whitespace, remove strange unicode that can break tokenizers.
    """
    if not text:
        return ""
    # Replace multiple whitespace/newlines with single spaces
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    # Optionally remove control chars
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]+", " ", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 140) -> List[str]:
    """
    Character-based overlapping chunking.
    Returns list of chunks (strings).
    """
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # slide window
        start += chunk_size - overlap
    return chunks
