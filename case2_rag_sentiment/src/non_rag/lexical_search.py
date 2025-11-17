# src/non_rag/lexical_search.py
from case2_rag_sentiment.src.non_rag.normalize import normalize_text

class LexicalSearcher:
    def __init__(self, feedback_list):
        self.original = feedback_list
        self.normalized = [normalize_text(x) for x in feedback_list]

    def search(self, query: str):
        q = normalize_text(query)
        matches = [
            orig for orig, norm in zip(self.original, self.normalized)
            if q in norm
        ]
        return matches