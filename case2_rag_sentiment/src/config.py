# src/config.py
import os

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# --- Model İsimleri ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"

# --- Dosya Yolları ---
DATA_FILE_PATH = "data/musteriyorumlari.xlsx"

# --- Artifact Yolları (Oluşturulacak dosyalar) ---
TEMPLATES_PATH = "artifacts/df_templates_with_scores.csv"
FAISS_INDEX_PATH = "artifacts/faiss_template_index.idx"