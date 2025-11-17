import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from case2_rag_sentiment.src.rag.retriever import FaissRetriever
from case2_rag_sentiment.src.config.settings import (
    TEMPLATES_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL
)


class SentimentRAG:
    def __init__(self):
        # load templates
        self.df_templates = pd.read_csv(TEMPLATES_PATH)

        # load FAISS index
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        # load embedder
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        # retriever
        self.retriever = FaissRetriever(self.index, self.embedder)

    def query(self, query_text, top_k=5):
        ids, scores = self.retriever.search(query_text, top_k)
        sub = self.df_templates[self.df_templates["template_id"].isin(ids)]

        weighted = float((sub["sentiment_score"] * sub["count"]).sum() / sub["count"].sum())

        return {
            "query": query_text,
            "top_ids": ids,
            "similarities": scores,
            "templates": sub,
            "weighted_sentiment": weighted,
            "total_comments": int(sub["count"].sum())
        }