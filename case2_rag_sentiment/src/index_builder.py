# src/index_builder.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from config import EMBEDDING_MODEL, SENTIMENT_MODEL, TEMPLATES_PATH, FAISS_INDEX_PATH
import json
import os

def build_and_save_artifacts():
    """
    Şablonların sentiment skorlarını hesaplar ve FAISS indeksini oluşturur.
    Her ikisini de 'artifacts' klasörüne kaydeder.
    """
    try:
        df_templates = pd.read_csv("artifacts/temp_templates.csv")
    except FileNotFoundError:
        print("HATA: 'artifacts/temp_templates.csv' bulunamadı.")
        print("Lütfen önce 'src/data_preprocessor.py' scriptini çalıştırın.")
        return

    # 1-Sentiment Skorlarını Hesapla 
    print(f"'{SENTIMENT_MODEL}' modeli ile 32 şablon için sentiment hesaplanıyor...")
    sentiment_model = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=0) # GPU
    
    template_texts = df_templates["Feedback"].tolist()
    sentiment_results = sentiment_model(template_texts, batch_size=32)
    
    label_to_score = {"positive": 1, "neutral": 0, "negative": -1}
    label_map = {
        "positive": "positive", "LABEL_1": "positive",
        "neutral": "neutral", "LABEL_2": "neutral",
        "negative": "negative", "LABEL_0": "negative"
    }

    df_templates["sentiment_label"] = [label_map.get(r['label'], 'neutral') for r in sentiment_results]
    df_templates["sentiment_score"] = [label_to_score[label] for label in df_templates["sentiment_label"]]
    
    # Manuel Kural Eklemeleri
    rules_path = "artifacts/manual_rules.json"
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)

        for rule in rules.get("override_labels", []):
            mask = df_templates["Feedback"].str.contains(rule["contains"], case=False, na=False)
            df_templates.loc[mask, "sentiment_label"] = rule["sentiment_label"]
            df_templates.loc[mask, "sentiment_score"] = rule["sentiment_score"]

        print(f"Manual rules uygulandı {len(rules.get('override_labels', []))} adet rule işlendi.")
    else:
        print("Manual rule dosyası bulunamadı, override yapılmadı.")
    

    # Tamamlanmış şablonları kaydet
    df_templates.to_csv(TEMPLATES_PATH, index=False)
    print(f"Sentiment skorları hesaplandı ve '{TEMPLATES_PATH}' dosyasına kaydedildi.")

    # 2-FAISS İndeksini Oluştur
    print(f"'{EMBEDDING_MODEL}' modeli ile FAISS indeksi oluşturuluyor...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    template_embeddings = embed_model.encode(template_texts)
    
    dim = template_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) # Kozinüs benzerliği için

    # Vektörleri normalize et
    template_embeddings_norm = template_embeddings / np.linalg.norm(template_embeddings, axis=1, keepdims=True)
    index.add(template_embeddings_norm)
    
    # İndeksi kaydet
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS indeksi oluşturuldu ve '{FAISS_INDEX_PATH}' dosyasına kaydedildi.")

if __name__ == "__main__":
    # python src/index_builder.py
    build_and_save_artifacts()