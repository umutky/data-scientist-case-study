# src/rag/index_builder.py
import os
import numpy as np
import pandas as pd
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import json

from src.config.settings import (
    SENTIMENT_MODEL,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    TEMPLATES_PATH,
    ARTIFACT_DIR,
    MANUAL_RULES_PATH
)


def apply_manual_overrides(df: pd.DataFrame):
    """
    manual_rules.json içindeki sentiment override kurallarını uygular.
    """
    if not os.path.exists(MANUAL_RULES_PATH):
        print("No manual_rules.json found → skipping manual overrides.")
        return df

    print(f"Applying manual sentiment override rules from {MANUAL_RULES_PATH}")

    with open(MANUAL_RULES_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)

    overrides = rules.get("override_labels", [])
    total_overridden = 0

    for rule in overrides:
        keyword = rule.get("contains", "")
        new_label = rule.get("sentiment_label", "neutral")
        new_score = rule.get("sentiment_score", 0)

        mask = df["Feedback"].str.contains(keyword, case=False, na=False)
        overridden_count = mask.sum()

        if overridden_count > 0:
            df.loc[mask, "sentiment_label"] = new_label
            df.loc[mask, "sentiment_score"] = new_score

        total_overridden += overridden_count

    print(f"Manual overrides applied. Total overridden rows: {total_overridden}")
    return df


def build_artifacts():
    # ---- 0. Load templates ----
    df_templates = pd.read_csv(os.path.join(ARTIFACT_DIR, "temp_templates.csv"))
    print(f"Loaded {len(df_templates)} templates.")

    # ---- 1. Sentiment Model ----
    device = 0
    sentiment = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=SENTIMENT_MODEL,
        device=device
    )

    texts = df_templates["Feedback"].tolist()
    results = sentiment(texts, batch_size=32)

    label_map = {"positive": 1, "neutral": 0, "negative": -1}

    df_templates["sentiment_label"] = [r["label"].lower() for r in results]
    df_templates["sentiment_score"] = df_templates["sentiment_label"].map(label_map)

    # ---- 1.1 Apply Manual Rules ----
    df_templates = apply_manual_overrides(df_templates)

    # Save updated template scores
    df_templates.to_csv(TEMPLATES_PATH, index=False)
    print(f"Saved sentiment templates → {TEMPLATES_PATH}")

    # ---- 2. Embeddings + FAISS ----
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embs = embedder.encode(texts)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"Saved FAISS index → {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build_artifacts()