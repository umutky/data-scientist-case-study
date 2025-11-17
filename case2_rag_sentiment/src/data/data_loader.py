import pandas as pd
from case2_rag_sentiment.src.config.settings import DATA_FILE_PATH

def load_feedback():
    df = pd.read_excel(DATA_FILE_PATH)
    df.columns = df.columns.str.strip()
    if "Feedback" not in df.columns:
        raise ValueError("Dataset must contain a 'Feedback' column.")
    return df