# src/app/non_rag_app.py
import pandas as pd
from case2_rag_sentiment.src.non_rag.pipeline import NonRAGPipeline
from case2_rag_sentiment.src.config.settings import DATA_FILE_PATH

def print_header():
    print("\nNon-RAG Pipeline loading...")
    print("Device set to use mps:0")
    print("Data loaded successfully.\n")
    print("--- NON-RAG Sentiment Assistant ---")
    print("Sorgunuzu girin (exit ile çıkış)\n")

def format_result(res):
    print("\n--- NON-RAG Sonucu ---")
    print(f"Sorgu: {res['query']}")
    print(f"Eşleşen Yorum Sayısı: {res['sentiment']['total']}")

    print("\n--- Duygu Dağılımı ---")
    for lbl, cnt in res['sentiment']['label_counts'].items():
        print(f"{lbl}: {cnt}")

    print(f"\nOrtalama Skor: {res['sentiment']['avg_score']:.3f}\n")

def main():
    df = pd.read_excel(DATA_FILE_PATH)
    feedback = df["Feedback"].astype(str).tolist()

    pipeline = NonRAGPipeline(feedback)

    print_header()

    while True:
        q = input("> ")

        if q.lower() == "exit":
            break

        if not q.strip():
            continue

        result = pipeline.query(q)
        format_result(result)

if __name__ == "__main__":
    main()