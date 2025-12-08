# 🔐 Email Spam Detection & Mitigation using Machine Learning

Selamat datang di repository **UAS Keamanan Data** yang berfokus pada **Deteksi dan Mitigasi Ancaman Email Spam**.  
Proyek ini mengembangkan sistem yang dapat mengklasifikasikan email **spam / berbahaya** dan menerjemahkannya menjadi **aksi mitigasi keamanan** yang dapat digunakan oleh tim SOC (Security Operations Center).

Output utama:
- 🧠 Model Machine Learning (**XGBoost Balanced**)
- 🌐 Aplikasi Streamlit untuk analisis & deteksi email secara real-time

---

## 👤 Mahasiswa Pengembang
- **Vinsensius Erik Kie**  
- Kelas: **5PDS1**  
- Mata Kuliah: **DSF01 — Keamanan Data**

---

## 📌 Latar Belakang

Email menjadi saluran utama penyebaran:
✔ Spam  
✔ Phishing  
✔ Malware berbasis social engineering  

Dengan volume email yang besar, diperlukan sistem otomatis yang membantu mendeteksi ancaman sebelum sampai ke pengguna.

Proyek ini menggunakan data asli dunia kerja (Enron Email Dataset) untuk membangun model pendeteksi spam yang **akurat dan dapat dioperasionalkan**.

---

## 📂 Struktur Direktori

📦 UAS_KeamananData_SPAMDetection/
│
├── 📁 data/
│ └── emails.csv # Data hasil preprocessing
│
├── 📁 models/
│ ├── model_xgb_balanced.pkl # Model final siap pakai
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer untuk teks
│
├── 🌐 app_streamlit.py # Aplikasi UI Streamlit
├── 📓 notebook.ipynb # Notebook analisis & pelatihan model
└── 📄 README.md # Dokumentasi proyek


---

## ✨ Fitur Utama Aplikasi Web

| Fitur | Deskripsi |
|------|-----------|
| **Real-Time Spam Detection** | Input subject + body → model langsung prediksi |
| **Risk Scoring** | Probabilitas spam → HIGH / MEDIUM / LOW |
| **Mitigation Action** | QUARANTINE / ALERT REVIEW / ALLOW |
| **Analisis Email** | Menampilkan domain, link, dan indikator risiko |
| **Decision Support System** | Membantu triase oleh SOC |

---

## ⚙️ Alur Kerja Proyek

1. **Pengumpulan & Pemuatan Data**  
   Enron Email dataset (517.401 data)

2. **Preprocessing & Feature Extraction**
   - Ekstraksi `from`, `subject`, `body`
   - TF-IDF text vectorization
   - Feature engineering keamanan (num_urls, is_internal_sender, dll.)

3. **Pelabelan Spam/Ham**
   - Berdasarkan keyword dan pola URL → *heuristic labeling*

4. **Training & Model Selection**
   - Logistic Regression
   - Random Forest
   - **XGBoost Balanced** → *model terbaik*

5. **Evaluasi**
   - Accuracy 0.9957
   - Recall 0.9809 (prioritas keamanan)
   - ROC-AUC 0.9973

6. **Deployment**
   - Model disimpan `.pkl`
   - Integrasi ke aplikasi **Streamlit**

---

## 🛡 Mitigation Mapping

| Probabilitas Spam | Level Risiko | Action           |
|-------------------|:------------:|------------------|
| > 0.90            | High         | **QUARANTINE**   |
| 0.70 – 0.90       | Medium       | **ALERT REVIEW** |
| < 0.70            | Low          | **ALLOW**        |

Domain berisiko tinggi → *candidate blocklist*

---

## ▶️ Instalasi, Pengaturan & Cara Menjalankan Aplikasi

### 1️⃣ Prasyarat
- Python 3.9+
- pip (package manager)
- Git (opsional)

### 2️⃣ Kloning Repositori
```bash
git clone https://github.com/yourusername/UAS_KeamananData_SPAMDetection.git
cd UAS_KeamananData_SPAMDetection
```

### 3️⃣ Instalasi Dependensi
Proyek ini sudah menyediakan requirements.txt.
Install seluruh library dengan:

```bash
pip install -r requirements.txt
```

### 4️⃣ Menjalankan Sistem Deteksi Spam
```bash
streamlit run app.py
```
