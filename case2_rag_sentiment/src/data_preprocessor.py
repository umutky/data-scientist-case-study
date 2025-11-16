# src/data_preprocessor.py
import pandas as pd
from config import DATA_FILE_PATH, TEMPLATES_PATH
import os

def process_data():
    """
    Ana veri dosyasını okur, 32 şablonu bulur ve
    bunları 'artifacts' klasörüne CSV olarak kaydeder.
    """
    print("Veri işleme başlıyor...")
    df = pd.read_excel(DATA_FILE_PATH)
    print(f"Toplam satır sayısı: {len(df)}")


    # Sütun yapısı kontrolü
    required_cols = {"Title", "Feedback", "Score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolon(lar): {missing}. Lütfen veri dosyasını kontrol edin.")

    # 32 şablonu bulma (Notebook'unuzdaki kod)
    df_templates = (
        df.groupby(["Title", "Feedback"])
          .agg(
              count=("Feedback", "size"),
              avg_score=("Score", "mean") # Gürültülü de olsa orijinalini tutalım
          )
          .reset_index()
    )
    df_templates["template_id"] = df_templates.index
    
    # Artifacts klasörünü kontrol et
    os.makedirs("artifacts", exist_ok=True)
    
    # Şablonları kaydet (Ancak skorlar henüz yok)
    df_templates.to_csv("artifacts/temp_templates.csv", index=False)
    
    print(f"{len(df_templates)} adet benzersiz şablon bulundu ve 'artifacts/temp_templates.csv' olarak kaydedildi.")
    return df_templates

if __name__ == "__main__":
    # Bu script'i doğrudan çalıştırırsak veriyi işler
    # python src/data_preprocessor.py
    process_data()