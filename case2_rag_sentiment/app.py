# app.py
import sys
import os


# Bu, src/ içindeki importların çalışmasını sağlar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.rag_pipeline import SentimentRAG
import sys

def main():
    # Pipeline'ı bir kez yükle
    try:
        rag = SentimentRAG()
    except FileNotFoundError:
        print("Çıkılıyor.")
        sys.exit(1)

    # Sürekli sorgu al ve çalıştır
    print("\n--- RAG Duygu Analizi Asistanı ---")
    print("Sorgunuzu girin (çıkmak için 'exit' yazın):")
    
    while True:
        query = input("> ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        rag.query(query_text=query, top_k=5)

if __name__ == "__main__":
    main()