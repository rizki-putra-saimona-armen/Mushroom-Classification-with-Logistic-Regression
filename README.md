#  Mushroom Classification with Logistic Regression

Proyek klasifikasi untuk memprediksi apakah jamur **aman dikonsumsi (edible)** atau **beracun (poisonous)** berdasarkan ciri-ciri fisiknya, menggunakan **Logistic Regression** (dengan skema One-vs-Rest), dilengkapi analisis eksploratif mendalam untuk seleksi fitur. Dibangun dengan **scikit-learn** dan dibantu library [`jcopml`](https://pypi.org/project/jcopml/) untuk preprocessing, tuning, dan evaluasi model.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-MachineLearning-orange)
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

##  Deskripsi

Proyek ini menjawab pertanyaan klasik dalam machine learning: **"Apakah jamur ini aman dimakan?"** — menggunakan dataset ciri-ciri fisik jamur (warna, bentuk, aroma, dll). Yang membuat notebook ini menarik adalah pendekatan **seleksi fitur berbasis association matrix**, sangat berguna ketika kita tidak memiliki latar belakang domain (misal biologi) untuk menentukan fitur mana yang relevan secara manual.

Proyek ini cocok sebagai referensi belajar:
- Cara membaca dan memanfaatkan **association matrix** untuk seleksi fitur pada data kategorik
- Membangun pipeline klasifikasi dengan **Logistic Regression + One-vs-Rest**
- Evaluasi model klasifikasi secara lengkap: **confusion matrix, classification report, ROC curve, PR curve**
- Analisis **feature importance** dan validasi silang menggunakan **correlation ratio** antara fitur kategorik & numerik

---

##  Struktur Proyek

```
.
├── Part_7_-_Latihan.ipynb   # Notebook utama
├── data/
│   └── mushrooms.csv        # Dataset ciri-ciri jamur
├── model/                   # Output model tersimpan (opsional)
└── README.md
```

> Dataset berisi fitur-fitur kategorik seperti `odor` (aroma), `gill_color` (warna insang), `ring_type` (jenis cincin), `spore_print_color` (warna spora), dengan kolom target **`edible`** (edible/poisonous).

---

##  Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/username/nama-repo.git
cd nama-repo
```

2. Buat virtual environment (opsional tapi disarankan):
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
pip install jcopml luwiji
```

---

##  Exploratory Data Analysis (EDA)

### 1. Cek Missing Value
```python
plot_missing_value(df, return_df=True)
```
Memastikan tidak ada data kosong yang perlu ditangani sebelum modeling.

### 2. Cek Balance Kelas Target
```python
df.edible.value_counts()
```
Karena ini kasus klasifikasi, penting memeriksa apakah distribusi kelas target (`edible` vs `poisonous`) seimbang — kelas yang timpang bisa membuat model bias.

### 3. Association Matrix — Seleksi Fitur Tanpa Domain Knowledge
```python
plot_association_matrix(df, "edible", categoric_col="auto")
```

>  **Kenapa dipakai?** Ketika kita tidak familiar dengan domain data (misal biologi jamur), association matrix membantu melihat fitur mana yang punya hubungan kuat dengan target — sehingga fitur yang tidak relevan bisa dibuang tanpa harus konsultasi ke ahli.

**Temuan penting:**
- Kolom `veil_type` dibuang karena hanya berisi **satu nilai unik** (tidak membawa informasi apapun untuk model)
- Dengan threshold korelasi ≥ 60%, terpilih **4 fitur terkuat**: `odor`, `gill_color`, `ring_type`, `spore_print_color`

---

##  Pipeline & Preprocessing

```python
preprocessor = ColumnTransformer([
    ("categoric", cat_pipe(encoder="onehot"), X_train.columns),
])

pipeline = Pipeline([
    ("prep", preprocessor),
    ("algo", OneVsRestClassifier(LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=42)))
])
```

| Komponen | Detail |
|---|---|
| **Fitur digunakan** | `odor`, `gill_color`, `ring_type`, `spore_print_color` (hasil seleksi association matrix) |
| **Encoding** | One-Hot Encoding untuk seluruh fitur kategorik |
| **Model** | `LogisticRegression` dibungkus `OneVsRestClassifier` |
| **Solver** | `lbfgs` |

---

##  Training & Tuning

Model dituning menggunakan `GridSearchCV`:

```python
param_grid = {
    "algo__estimator__C": [0.1, 1, 10],
    "algo__estimator__fit_intercept": [True, False]
}

model = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)
```

| Parameter | Detail |
|---|---|
| **`C`** | Kekuatan regularisasi (0.1, 1, 10) |
| **`fit_intercept`** | Apakah model menghitung intercept (True/False) |
| **Cross-validation** | 3-fold |

Setelah training, tiga skor dibandingkan: skor train, skor CV terbaik (`best_score_`), dan skor test — untuk memastikan model general dan tidak overfit.

---

## 📈 Evaluasi Model

| Metode | Fungsi |
|---|---|
| **Confusion Matrix** | `plot_confusion_matrix(...)` — melihat jumlah prediksi benar/salah per kelas |
| **Classification Report** | `plot_classification_report(...)` — precision, recall, f1-score per kelas |
| **ROC Curve** | `plot_roc_curve(...)` — trade-off true positive rate vs false positive rate |
| **Precision-Recall Curve** | `plot_pr_curve(...)` — performa model pada data dengan potensi class imbalance |

### Feature Importance
```python
mean_score_decrease(X_train, y_train, model, plot=True, topk=10)
```
Hasil menunjukkan **`odor` (aroma)** sebagai fitur paling berpengaruh terhadap prediksi — sejalan dengan intuisi umum bahwa jamur beracun seringkali memiliki aroma yang khas/menyengat.

### Insight Visual
Notebook memvisualisasikan distribusi 4 fitur utama terhadap target menggunakan `countplot`, mempermudah interpretasi pola mana yang cenderung mengarah ke jamur beracun vs aman dikonsumsi.

### Correlation Ratio (Validasi Tambahan)
```python
plot_correlation_ratio(df_subset, cols_cat, ["ring_number"])
```
Digunakan untuk mengecek hubungan antara fitur kategorik dengan fitur numerik (`ring_number`) yang sebelumnya tidak dipakai.

>  **Insight kunci:** Hasil correlation ratio mengonfirmasi bahwa `ring_number` **tidak berpengaruh signifikan** terhadap `edible` — memvalidasi keputusan sebelumnya untuk tidak menyertakan fitur ini dalam model.

---

##  Tech Stack

- **scikit-learn** — Logistic Regression, OneVsRestClassifier, Pipeline, GridSearchCV
- **jcopml** — Preprocessing pipeline, association matrix, correlation ratio, evaluasi model
- **pandas** & **NumPy** — Manipulasi data
- **Matplotlib** & **Seaborn** — Visualisasi EDA & hasil evaluasi
- **luwiji** — Ilustrasi pendukung pembelajaran

---

##  Lisensi

Proyek ini menggunakan lisensi **MIT** — bebas digunakan, dimodifikasi, dan didistribusikan ulang untuk keperluan pembelajaran maupun pengembangan lebih lanjut.

---

##  Kontribusi

Kontribusi, saran, dan diskusi sangat terbuka! Silakan buat *issue* atau *pull request* — misalnya untuk mencoba algoritma klasifikasi lain (Random Forest, XGBoost) sebagai perbandingan performa dengan Logistic Regression.

---

