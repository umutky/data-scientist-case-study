# case2_rag_sentiment/src/config/settings.py
import os

# Proje root'unun yolunu bul
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# Data dosyası
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "musteriyorumlari.xlsx")

# Artifacts klasörü
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Artifact dosyaları
TEMPLATES_PATH = os.path.join(ARTIFACT_DIR, "df_templates_with_scores.csv")
FAISS_INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss_template_index.idx")
TEMP_TEMPLATES_PATH = os.path.join(ARTIFACT_DIR, "temp_templates.csv")

# Modeller
SENTIMENT_MODEL = "savasy/bert-base-turkish-sentiment-cased"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Manuel Kural
MANUAL_RULES_PATH = os.path.join(ARTIFACT_DIR, "manual_rules.json")