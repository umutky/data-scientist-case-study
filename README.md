# Case Study 1– Kredi Risk Tahminleme
Bu çalışma, kredi riski modellemesi odağında hazırlanmıştır. Veri temizleme, tutarsızlıkların giderilmesi, feature engineering, modelleme, hiperparametre optimizasyonu ve model performans analizini kapsamaktadır.

Case 1 için tüm analizler ve açıklamalar, yorum satırlarıyla birlikte ayrıntılı biçimde Jupyter Notebook içinde sunulmuştur: case1_credit_risk/credit_risk_analysis_and_modeling.ipynb

# Case Study 2– Konu Bazlı Arama + Duygu Analizi + Düşük Kaynak Optimizasyonu

Bu proje, 50.000 müşteri geri bildirimi üzerinde çalışan konu bazlı semantik arama ve duygu analizi sistemi sunar. Yapı hem RAG tabanlı semantik arama hem de Non-RAG anahtar kelime eşleme yöntemlerini içerir. Amaç, düşük donanımlı ortamlarda bile hızlı çalışabilen bir sistem tasarlamaktır.

Projenin amacı:
- Belirli bir konu/anahtar kelime verildiğinde ilgili yorumları bulmak
- Bu yorumlar için duygu analizi yapmak
- Düşük donanımlı ortamlarda bile hızlı çalışacak bir yapı kurmak
- RAG ve Non-RAG yöntemlerini karşılaştırmak


## Proje Mimarisi
```
data-scientist-case-study/
│
├── README.md
├── requirements.txt
├── .venv/
│
├── case1_credit_risk/
│   ├── credit_risk_case.xlsx
│   └── credit_risk_analysis_and_modeling.ipynb
│
├── case2_rag_sentiment/
│   │
│   ├── data/
│   │   └── musteriyorumlari.xlsx
│   │
│   ├── artifacts/
│   │   ├── temp_templates.csv
│   │   ├── df_templates_with_scores.csv
│   │   ├── faiss_template_index.idx
│   │   └── manual_rules.json
│   │
│   ├── images/
│   │   ├── image.png
│   │   ├── image-1.png
│   │   ├── image-2.png
│   │   └── image-3.png
│   │
│   ├── src/
│   │   ├── config/
│   │   │   └── settings.py
│   │   ├── data/
│   │   │   └── data_preprocessor.py
│   │   ├── rag/
│   │   │   ├── index_builder.py
│   │   │   ├── rag_pipeline.py
│   │   │   └── rag_app.py
│   │   ├── non_rag/
│   │   │   └── non_rag_app.py
│   │   └── evaluation/
│   │       └── evaluate.py
│   │
│   └── rag_prototyping_and_analysis.ipynb
```

## Kurulum
Repoyu klonlayın:
git clone https://github.com/umutky/data-scientist-case-study.git

Virtual env oluşturun:
cd data-scientist-case-study

Requirements yükleyin: pip install --upgrade pip
pip install -r requirements.txt


## Pipeline Adımları
### 1. Data Preprocessing
Çalıştır: python -m src.data.data_preprocessor

Bu adım:
- 50.000 yorumdan 32 benzersiz template çıkarır
- Her template tekrar sayısını hesaplar
- Ortalama puanları hesaplar
- Sonuç: artifacts/temp_templates.csv

### 2. FAISS Index + Sentiment Skorları
Çalıştır: python -m src.rag.index_builder

Bu script:
	1.	savasy/bert-base-turkish-sentiment-cased ile her template’in sentiment’i hesaplanır
	2.	Eğer varsa manual_rules.json içindeki kural bazlı düzeltmeler uygulanır
	3.	SentenceTransformer modeli ile template embedding’leri çıkarılır
	4.	FAISS index oluşturulur

Çıktılar:
- df_templates_with_scores.csv
- faiss_template_index.idx

**NOT**: manual_rules.json içerisien bütün eklemeler yapılmamıştır, burada amaç istenildiği zaman rule based bir yapıyla yapının kuvvetlendirilebileceğini göstermektir. Ayrıca bu yapı BERT yerine farklı modeller kullanılarak da kuvvetlendirilebilir.

### 3. RAG Pipeline Çalıştırma
Çalıştır: python -m src.app.rag_app

Örnek kullanım: > kredi

Çıktılar:
- En benzer 5 template
- Her birinin duygu skoru
- Template'in temsil ettiği toplam yorum
- Semantik arama benzerlik skorları

### 4. Non-RAG Pipeline Çalıştırma
Çalıştır: python -m src.app.non_rag_app

Örnek kullanım: > kredi

Çıktılar:
- Sadece substring match yapan yorumlar
- Pozitif/negatif dağılım
- Ortalama duygu skoru
- Eşleşen yorum sayısı

### 5. RAG - Non-RAG Karşılaştırma
Çalıştır: python -m src.evaluation.evaluate

Otomatik kıyaslanan metrikler:
- RAG sentiment
- Non-RAG sentiment
- Sentiment farkı
- Template coverage
- Query başına latency
- RAG / Non-RAG hız kıyaslaması

Örnek Çıktı:
```
=== Evaluation: RAG vs NON-RAG ===
Loading data for Non-RAG pipeline...
Initializing RAG pipeline...
Initializing Non-RAG pipeline...
Loading Non-RAG pipeline...
Loading sentiment model: savasy/bert-base-turkish-sentiment-cased
Device set to use mps:0
Non-RAG pipeline initialized.


--- Evaluating query: 'kredi' ---

--- Evaluating query: 'kredi durumu' ---

--- Evaluating query: 'kötüydü' ---

--- Evaluating query: 'berbat' ---

--- Evaluating query: 'araç bulunamadı' ---

--- Evaluating query: 'destek çok kötüydü' ---

--- Evaluating query: 'memnunum' ---

--- Evaluating query: 'süperdi' ---

--- Evaluating query: 'güvenilir' ---

--- Evaluating query: 'felaket' ---

--- Evaluating query: 'mükemmeldi' ---

--- Evaluating query: 'araç' ---

--- Evaluating query: 'destek' ---

--- Evaluating query: 'finansman' ---

=== Evaluation Completed ===
Results saved → /Users/umutkaya/Documents/data_scientist_case_study/src/artifacts/evaluation_results.csv

             query  rag_sentiment  non_rag_sentiment  sentiment_diff  rag_count  non_rag_count  rag_latency_ms  non_rag_latency_ms  count_ratio_rag_over_non
             kredi      -0.400051          -0.375599       -0.024451       7894          25032      100.287875        23391.018833                  0.315356
      kredi durumu      -0.601624           0.000000       -0.601624       7882              0       22.790417            2.083625                       NaN
           kötüydü      -1.000000          -1.000000        0.000000       7760           1593       26.153959         1284.566458                  4.871312
            berbat      -0.585174           0.000000       -0.585174       7743              0        7.840750            1.986416                       NaN
   araç bulunamadı      -1.000000           0.000000       -1.000000       7754              0       33.186708            2.058125                       NaN
destek çok kötüydü      -1.000000           0.000000       -1.000000       7763              0       30.229208            1.771208                       NaN
          memnunum       0.016067           0.000000        0.016067       7842              0        7.930500            1.873250                       NaN
           süperdi       0.217891           0.000000        0.217891       7825              0        7.714458            2.210041                       NaN
         güvenilir      -0.193260           0.000000       -0.193260       7715              0        7.406208            1.953709                       NaN
           felaket      -1.000000           0.000000       -1.000000       7742              0        7.852458            1.891792                       NaN
        mükemmeldi       0.211849           0.000000        0.211849       7798              0        7.710916            1.981667                       NaN
              araç      -0.183382          -0.414634        0.231252       7847          10947        7.412209         8730.679208                  0.716817
            destek       0.223738           0.018668        0.205071       7768          12428        7.850375         9413.085708                  0.625040
         finansman      -0.596349           0.000000       -0.596349       7779              0        7.957708            2.195917                       NaN
```

## Elde Edilen İç Görüler
![Title Bazlı Ağırlıklı Ortalama Duygu Skoru](case2_rag_sentiment/images/image.png)

![Top 5 Negatif Yorum](case2_rag_sentiment/images/image-1.png)
Burada kullandığımız pre-trained BERT modelinin, bazı cümleleri yanlış sınıflandırdığını görebilmekteyiz. Bu kurmuş olduğumuz yapıdan kaynaklı değil, BERT modelinin cümleyi yanlış sınıflandırmasından kaynaklıdır.

![Top 5 Pozitif Yorum](case2_rag_sentiment/images/image-2.png)

![Tanımlanan Skorlar vs Anlamsal Skorlar](case2_rag_sentiment/images/image-3.png)
Yapılan analizlerde de görülmüştür ki veri seti içerisinde neredeyse her yorum için ortalama Score değeri aynıdır. Bu da aslında bu değerlerin gürültülü (noisy) olduğunun bir göstergesidir. Bu skorların yerine RAG mimarimizin sonucunda elde ettiğimiz skorların daha anlamlı olduğunu görmekteyiz.

RAG tabanlı semantik skorlar çok daha tutarlı sonuçlar vermektedir. Örnek çıktıda görüldüğü üzere Non-RAG anlam değerlendirmesi yapmadan yalnızca kelimelerin birebir uyuştuğu yorumları almaktadır. 

**NOT:** Analizler, pre-trained sentiment modelinin nötr yorumları doğru sınıflandırmada başarısız olduğunu ortaya koymuştur. Bu nedenle ilerleyen aşamalarda rule-based iyileştirmeler veya daha gelişmiş modellerle yeniden kalibrasyon yapılabilir.


1. RAG (Retrieval-Augmented) Pipeline
    - Semantik arama için FAISS + Sentence-BERT kullanır.
	- 50.000 yorumu temsil eden yalnızca 32 şablon üzerinden hızlı değerlendirme yapar.
	- Sorgunun anlamını yakaladığı için daha doğru ve tutarlı sentiment skorları üretir.
	- Sorgu süresi daha kısa.
	- Non-RAG’ın “0 sonuç” verdiği durumlarda bile anlamlı çıktılar sağlar. (Çünkü bağlam da yakalayabiliyor.)

2. Non-RAG Pipeline
	- Tam metin üzerinde kelime-bazlı arama yapar.
	- Eşleşme tamamen string contains mantığına dayanır.
	- Semantik anlam olmadığı için birçok sorguda sonuç üretemez.
	- Sorgu süresi daha uzun.
	- Ayrıştırma hassastır; küçük ifade farklılıklarında bile eşleşme kaçabilir.


## Manuel Kural Sistemi
Örnek:
{
    "override_labels": [
        {
            "contains": "beklentimi karşıladı",
            "sentiment_label": "positive",
            "sentiment_score": 1
        }
    ]
}

Bu mekanizma:
- Modelin hatalı etiketlediği şablonları düzeltmek
- Domain expert bilgisini sisteme katmak
- Üretim ortamında yüksek kontrol sağlamak.
