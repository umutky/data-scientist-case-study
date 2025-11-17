# src/non_rag/pipeline.py
from transformers import pipeline
from collections import Counter
import numpy as np
from case2_rag_sentiment.src.non_rag.lexical_search import LexicalSearcher
from case2_rag_sentiment.src.config.settings import SENTIMENT_MODEL

class NonRAGPipeline:
    def __init__(self, feedback_list):
        print("Loading Non-RAG pipeline...")
        self.searcher = LexicalSearcher(feedback_list)

        print(f"Loading sentiment model: {SENTIMENT_MODEL}")
        self.sentiment_model = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            device=0
        )
        print("Non-RAG pipeline initialized.\n")

    def analyze(self, comments):
        if not comments:
            return {
                "total": 0,
                "label_counts": {},
                "avg_score": 0,
                "scores": [],
                "comments": [],
            }

        results = self.sentiment_model(comments, batch_size=64, truncation=True)

        label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
        scores = []
        labels = []

        for r in results:
            lbl = r["label"].lower()
            labels.append(lbl)
            scores.append(label_to_score.get(lbl, 0))

        return {
            "total": len(comments),
            "label_counts": dict(Counter(labels)),
            "avg_score": float(np.mean(scores)),
            "comments": comments,
            "scores": scores,
        }

    def query(self, query_text):
        matches = self.searcher.search(query_text)
        sentiment_summary = self.analyze(matches)

        return {
            "query": query_text,
            "matches": matches,
            "sentiment": sentiment_summary
        }