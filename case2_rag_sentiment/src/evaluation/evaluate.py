# src/evaluation/evaluate.py

import os
import time
import pandas as pd

from case2_rag_sentiment.src.config.settings import DATA_FILE_PATH, ARTIFACT_DIR
from case2_rag_sentiment.src.rag.rag_pipeline import SentimentRAG
from case2_rag_sentiment.src.non_rag.pipeline import NonRAGPipeline


RESULTS_PATH = os.path.join(ARTIFACT_DIR, "evaluation_results.csv")


QUERIES = [
    "kredi",
    "kredi durumu",
    "kötüydü",
    "berbat",
    "araç bulunamadı",
    "destek çok kötüydü",
    "memnunum",
    "süperdi",
    "güvenilir",
    "felaket",
    "mükemmeldi",
    "araç",
    "destek",
    "finansman"
]


def main():
    print("=== Evaluation: RAG vs NON-RAG ===")

    # 1) Veriyi yükle (NON-RAG için)
    print("Loading data for Non-RAG pipeline...")
    df = pd.read_excel(DATA_FILE_PATH)
    feedback_list = df["Feedback"].astype(str).tolist()

    # 2) Pipeline'ları initialize et (bir kez)
    print("Initializing RAG pipeline...")
    rag = SentimentRAG()

    print("Initializing Non-RAG pipeline...")
    non_rag = NonRAGPipeline(feedback_list)

    rows = []

    for query in QUERIES:
        print(f"\n--- Evaluating query: '{query}' ---")

        # --- RAG ---
        start = time.perf_counter()
        rag_res = rag.query(query, top_k=5)
        rag_latency = (time.perf_counter() - start) * 1000  # ms

        rag_sent = rag_res["weighted_sentiment"]
        rag_count = rag_res["total_comments"]

        # --- NON-RAG ---
        start = time.perf_counter()
        non_res = non_rag.query(query)
        non_latency = (time.perf_counter() - start) * 1000  # ms

        non_sent = non_res["sentiment"]["avg_score"]
        non_count = non_res["sentiment"]["total"]

        # sentiment farkı
        sent_diff = rag_sent - non_sent

        row = {
            "query": query,
            "rag_sentiment": rag_sent,
            "non_rag_sentiment": non_sent,
            "sentiment_diff": sent_diff,
            "rag_count": rag_count,
            "non_rag_count": non_count,
            "rag_latency_ms": rag_latency,
            "non_rag_latency_ms": non_latency,
        }
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Coverage oranı gibi türev kolonlar ekleyebiliriz (0'a bölme korumalı)
    results_df["count_ratio_rag_over_non"] = results_df.apply(
        lambda r: r["rag_count"] / r["non_rag_count"] if r["non_rag_count"] > 0 else None,
        axis=1,
    )

    # Sonuçları kaydet
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n=== Evaluation Completed ===")
    print(f"Results saved → {RESULTS_PATH}\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()