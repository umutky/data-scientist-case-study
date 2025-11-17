# src/app/rag_app.py
from case2_rag_sentiment.src.rag.rag_pipeline import SentimentRAG

def print_header():
    print("\nRAG Pipeline loading...")
    print("Device set to use mps:0")   # İstersen dinamik olarak da yazabiliriz
    print("Artifacts loaded successfully.\n")
    print("--- RAG Sentiment Assistant ---")
    print("Sorgunuzu girin (exit ile çıkış)\n")

def format_result(result):
    print("\n--- RAG Sonucu ---")
    
    # Convert numpy types to Python types
    top_ids = [int(x) for x in result["top_ids"]]
    similarities = [round(float(x), 3) for x in result["similarities"]]

    print(f"Sorgu: {result['query']}")
    print(f"Benzetilen Template ID'leri: {top_ids}")
    print(f"Benzerlik Skorları: {similarities}")

    print("\n--- İlgili Template Örnekleri ---")
    for _, row in result["templates"].iterrows():
        print(f"[ID: {int(row['template_id'])}] {row['Feedback']} (score={row['sentiment_score']})")

    print("\n--- Özet Duygu Skoru ---")
    print(f"Toplam Temsil Edilen Yorum: {result['total_comments']}")
    print(f"Ağırlıklı Ortalama Sentiment Skoru: {result['weighted_sentiment']:.3f} (-1 Negatif, +1 Pozitif)\n")

def main():
    rag = SentimentRAG()
    print_header()

    while True:
        query = input("> ")
        if query.lower().strip() == "exit":
            print("Çıkılıyor...")
            break
        if not query.strip():
            continue

        result = rag.query(query, top_k=5)
        format_result(result)

if __name__ == "__main__":
    main()