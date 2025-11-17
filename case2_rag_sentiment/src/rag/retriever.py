import faiss
import numpy as np

class FaissRetriever:
    def __init__(self, index, embed_model):
        self.index = index
        self.embed_model = embed_model

    def search(self, query_text, top_k=5):
        q_emb = self.embed_model.encode([query_text])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        distances, ids = self.index.search(q_emb, top_k)
        return ids[0], distances[0]