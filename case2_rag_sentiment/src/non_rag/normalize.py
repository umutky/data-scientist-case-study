# src/non_rag/normalize.py
import unicodedata
import re

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Unicode normalizasyon
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # Lowercase
    text = text.lower()

    # Türkçe karakter değişimleri
    text = (
        text.replace("ı", "i")
            .replace("ğ", "g")
            .replace("ü", "u")
            .replace("ş", "s")
            .replace("ö", "o")
            .replace("ç", "c")
    )
    
    # Özel karakterleri kaldırma
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

