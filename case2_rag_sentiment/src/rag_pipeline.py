# src/rag_pipeline.py
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, TEMPLATES_PATH, FAISS_INDEX_PATH

class SentimentRAG:
    """
    Önceden oluşturulmuş FAISS indeksini ve şablon skorlarını yükleyerek
    hızlı, düşük kaynaklı duygu analizi sağlayan RAG pipeline'ı.
    """
    def __init__(self):
        print("RAG Pipeline yükleniyor...")
        # Modeli ve artifact'leri yükle
        try:
            self.df_templates = pd.read_csv(TEMPLATES_PATH)
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
            print("Tüm artifact'ler başarıyla yüklendi. Pipeline hazır.")
        except FileNotFoundError:
            print("HATA: 'artifacts' dosyaları bulunamadı.")
            print("Lütfen önce 'src/data_preprocessor.py' ve 'src/index_builder.py' çalıştırın.")
            raise
    
    def rag_search(self, query, top_k=5):
        """FAISS'te anlamsal arama yapar ve şablon ID'lerini döndürür."""
        q_emb = self.embed_model.encode([query])
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        
        distances, indices = self.index.search(q_emb, top_k)
        return indices[0], distances[0] # ID'ler ve Skorlar

    def get_weighted_sentiment(self, template_ids):
        """
        Bulunan şablon ID'lerine göre ağırlıklı duygu skorunu hesaplar.
        (Sizin rag_sentiment_fast fonksiyonunuz)
        """
        subset = self.df_templates[self.df_templates["template_id"].isin(template_ids)]
        
        if subset.empty:
            return 0, 0, []
            
        total_comments = subset["count"].sum()
        
        # Ağırlıklı ortalama
        weighted_sentiment = np.average(
            subset["sentiment_score"], 
            weights=subset["count"]
        )
        
        return weighted_sentiment, total_comments, subset
    
    def query(self, query_text, top_k=5):
        """
        Pipeline'ı uçtan uca çalıştırır ve sonucu basar.
        (Sizin rag_pipeline_fast fonksiyonunuz)
        """
        print(f"\n=== RAG PIPELINE SORGUSU: '{query_text}' ===")
        
        # 1. Arama
        template_ids, scores = self.rag_search(query_text, top_k=top_k)
        
        # 2. Ağırlıklı Sentiment
        sentiment_score, total_comments, subset_df = self.get_weighted_sentiment(template_ids)
        
        # 3. Sonuçları Göster
        print("\n--- Bulunan İlişkili Şablonlar ---")
        for idx, score in zip(template_ids, scores):
            feedback_text = self.df_templates.loc[idx, "Feedback"]
            print(f"  [Benzerlik: {score:.3f}] (ID: {idx}) {feedback_text}")
            
        print("\n--- Hızlı Duygu Analizi Özeti ---")
        print(f"  Toplam Temsil Edilen Yorum: {total_comments:,.0f}")
        print(f"  Ağırlıklı Ortalama Sentiment Skoru: {sentiment_score:.3f} (-1: Neg, +1: Poz)")
        
        return sentiment_score, total_comments