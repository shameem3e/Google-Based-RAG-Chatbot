# summarizer.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

DEFAULT_MODEL = "google/flan-t5-small"

class Summarizer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"[summarizer] loading {model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def summarize(self, question: str, contexts: List[str], max_new_tokens: int = 150) -> str:
        """
        contexts: list of chunk strings (ordered by relevance).
        We'll combine top chunks into a single prompt. If too long, tokenizer will truncate.
        """
        joined = "\n\n---\n\n".join(contexts)
        prompt = (
            "You are a helpful assistant. Use the CONTEXT to answer the QUESTION concisely. "
            "If the answer is not present in the context, say 'I don't know'.\n\n"
            f"CONTEXT:\n{joined}\n\nQUESTION: {question}\n\nAnswer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4, early_stopping=True)
        answer = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return answer
