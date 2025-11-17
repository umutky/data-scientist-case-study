#data_preprocessor.py
import os
import pandas as pd
from case2_rag_sentiment.src.config.settings import DATA_FILE_PATH, ARTIFACT_DIR

def preprocess_templates():
    print("Preprocessing feedback data...")

    df = pd.read_excel(DATA_FILE_PATH)
    df.columns = df.columns.str.strip()

    required_cols = {"Title", "Feedback", "Score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df_templates = (
        df.groupby(["Title", "Feedback"])
          .agg(count=("Feedback", "size"),
               avg_score=("Score", "mean"))
          .reset_index()
    )

    df_templates["template_id"] = df_templates.index

    out_path = os.path.join(ARTIFACT_DIR, "temp_templates.csv")
    df_templates.to_csv(out_path, index=False)

    print(f"Saved {len(df_templates)} templates → {out_path}")
    return df_templates


if __name__ == "__main__":
    preprocess_templates()