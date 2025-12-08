import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import hstack

# =========================
# 0. Konfigurasi & Konstanta
# =========================

APP_TITLE = "Enron Email Security – Spam & Risk Detector"
MODEL_NAME = "XGBoost (Balanced)"
DATASET_NAME = "Enron Email Dataset (Kaggle)"

# Statistik dataset (diambil dari notebook)
DATASET_STATS = {
    "n_total": 517_401,
    "n_ham": 408_976,
    "n_spam": 108_425,
    "prop_ham": 0.7904,
    "prop_spam": 0.2096,
    "avg_len": {
        "ham": {"subject_len": 25.9412, "body_len": 1067.7633},
        "spam": {"subject_len": 33.1696, "body_len": 4766.0449},
    },
}

# Metrik model (diambil dari notebook)
MODEL_METRICS = {
    "Logistic Regression": {
        "Accuracy": 0.9429,
        "Precision": 0.9233,
        "Recall": 0.7933,
        "F1-score": 0.8534,
        "ROC-AUC": 0.9739,
    },
    "Random Forest": {
        "Accuracy": 0.9941,
        "Precision": 0.9930,
        "Recall": 0.9787,
        "F1-score": 0.9858,
        "ROC-AUC": 0.9986,
    },
    "XGBoost (Baseline)": {
        "Accuracy": 0.9953,
        "Precision": 0.9998,
        "Recall": 0.9778,
        "F1-score": 0.9887,
        "ROC-AUC": 0.9970,
    },
    "XGBoost (Balanced)": {
        "Accuracy": 0.9957,
        "Precision": 0.9984,
        "Recall": 0.9809,
        "F1-score": 0.9896,
        "ROC-AUC": 0.9973,
    },
}

# Contoh blocklist domain eksternal dari hasil mitigasi
BLOCKLIST_DOMAINS = pd.DataFrame(
    {
        "domain": [
            "hotmail.com",
            "nymex.com",
            "carrfut.com",
            "ccomad3.uu.commissioner.com",
            "aol.com",
            "yahoo.com",
            "intcx.com",
            "williams.com",
            "concureworkplace.com",
            "msn.com",
            "reply.pm0.net",
            "wordsmith.org",
        ],
        "count_quarantined_emails": [
            652,
            348,
            309,
            283,
            188,
            179,
            147,
            138,
            128,
            80,
            79,
            75,
        ],
    }
)

# =========================
# 1. Load Model & Dataset
# =========================

@st.cache_resource
def load_artifacts():
    """
    Load model final dan TF-IDF vectorizer dari file .pkl.
    Sesuaikan path jika kamu memindahkan file ke folder lain.
    """
    model = joblib.load("models/model_xgb_balanced.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    return model, tfidf


@st.cache_resource
def load_raw_dataset():
    """
    Optional: load raw emails.csv hanya untuk EDA ringan.
    Jika file tidak ditemukan, halaman EDA akan menampilkan info statis saja.
    """
    try:
        df = pd.read_csv("data/emails.csv")
        return df
    except Exception:
        return None


model, tfidf = load_artifacts()
raw_df = load_raw_dataset()

# =========================
# 2. Helper Functions
# =========================

def count_urls(text: str) -> int:
    if text is None:
        return 0
    url_pattern = r"http[s]?://\S+"
    return len(re.findall(url_pattern, str(text)))


def is_internal_sender(from_addr: str, internal_domain: str = "enron.com") -> int:
    if from_addr is None:
        return 0
    return int(internal_domain.lower() in str(from_addr).lower())


def extract_domain(from_addr: str) -> str:
    if from_addr is None:
        return ""
    m = re.search(r"@([A-Za-z0-9._-]+)", str(from_addr))
    if m:
        return m.group(1).lower()
    return ""


def get_risk_level(p: float) -> str:
    if p > 0.90:
        return "HIGH"
    elif p >= 0.70:
        return "MEDIUM"
    else:
        return "LOW"


def choose_action(risk_level: str, is_internal: bool) -> str:
    """
    Mapping kebijakan mitigasi:
    - HIGH + eksternal -> QUARANTINE
    - HIGH + internal -> ALERT_REVIEW
    - MEDIUM         -> ALERT_REVIEW
    - LOW            -> ALLOW
    """
    if risk_level == "HIGH":
        if is_internal:
            return "ALERT_REVIEW (Internal – perlu review manusia)"
        else:
            return "QUARANTINE (Eksternal – isolasi otomatis)"
    elif risk_level == "MEDIUM":
        return "ALERT_REVIEW (Borderline – kirim ke SOC analyst)"
    else:
        return "ALLOW (Diteruskan ke user, bisa tetap dipantau)"


def build_feature_vector(from_addr: str, subject: str, body: str):
    """
    Mengubah input user menjadi fitur yang konsisten dengan pipeline training:
    - TF-IDF dari subject + body
    - Fitur numerik: subject_len, body_len, num_urls, is_internal_sender
    """
    subject = subject or ""
    body = body or ""
    from_addr = from_addr or ""

    text = (subject + " " + body).strip()

    # TF-IDF (sparse)
    X_text = tfidf.transform([text])

    # fitur numerik
    subject_len = len(subject)
    body_len = len(body)
    num_urls = count_urls(body)
    internal_flag = is_internal_sender(from_addr)

    X_num = np.array([[subject_len, body_len, num_urls, internal_flag]])

    # gabung fitur teks + numerik
    X_all = hstack([X_text, X_num])

    feature_info = {
        "subject_len": subject_len,
        "body_len": body_len,
        "num_urls": num_urls,
        "is_internal_sender": internal_flag,
    }

    return X_all, feature_info


# =========================
# 3. Halaman: Overview
# =========================

def render_overview():
    st.title("📌 Project Overview")
    st.markdown(f"### {APP_TITLE}")

    st.markdown(
        f"""
        Aplikasi ini merupakan prototipe Email Security Analytics yang dikembangkan
        sebagai bagian dari UAS Mata Kuliah Keamanan Data.

        Sistem memanfaatkan {DATASET_NAME} untuk:
        - Mengidentifikasi email yang berpotensi spam / berbahaya di lingkungan korporat, dan  
        - Memberikan rekomendasi tindakan mitigasi yang dapat dijadikan bahan keputusan
          oleh tim keamanan (SOC – Security Operations Center).
        """
    )

    st.markdown("#### 🎯 Tujuan Sistem")
    st.write(
        """
        1. Membangun model machine learning untuk klasifikasi email menjadi:
           - **ham (0)** / email normal  
           - **spam (1)** / email spam / email mencurigakan  
        2. Menghubungkan output model dengan kebijakan keamanan yang konkret:
           - Karantina email berisiko tinggi,  
           - Mengirim alert ke analis keamanan,  
           - Tetap mengizinkan email dengan pemantauan.
        """
    )

    st.markdown("#### ⚙️ Ringkasan Teknis Singkat")
    st.write(
        f"""
        - Tipe masalah   : *Binary Classification* (spam vs ham)  
        - Model final    : **{MODEL_NAME}**  
        - Representasi teks: TF-IDF pada gabungan subject + body (unigram & bigram)  
        - Fitur tambahan (numerik):
          - Panjang subject (`subject_len`)
          - Panjang body (`body_len`)
          - Jumlah URL di body (`num_urls`)
          - Indikator pengirim internal (`is_internal_sender`)
        """
    )

    st.markdown("#### 🔐 Perspektif Keamanan Siber")
    st.write(
        """
        Dari sudut pandang keamanan, fokus utama sistem ini adalah:
        - Meminimalkan False Negative (FN) / sebisa mungkin tidak ada spam/phishing
          yang lolos ke inbox.
        - Menjaga False Positive (FP) tetap rendah / email penting tidak ikut
          terblokir secara agresif, terutama dari domain internal perusahaan.
        """
    )

# =========================
# 4. Halaman: Dataset & EDA
# =========================

def render_dataset_eda():
    st.title("📂 Dataset & Exploratory Data Analysis")

    st.markdown("### Ringkasan Dataset")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Total Email", f"{DATASET_STATS['n_total']:,}")
    with cols[1]:
        st.metric("Ham (0)", f"{DATASET_STATS['n_ham']:,}")
    with cols[2]:
        st.metric("Spam (1)", f"{DATASET_STATS['n_spam']:,}")

    st.write(
        f"""
        Proporsi kelas:
        - Ham: **{DATASET_STATS['prop_ham']:.2%}**
        - Spam: **{DATASET_STATS['prop_spam']:.2%}**
        """
    )

    class_counts = pd.Series(
        {
            "ham (0)": DATASET_STATS["n_ham"],
            "spam (1)": DATASET_STATS["n_spam"],
        }
    )
    st.markdown("#### Distribusi Kelas (Ham vs Spam)")
    st.bar_chart(class_counts)

    st.markdown("#### Rata-rata Panjang Subject & Body per Kelas")
    avg_len_df = pd.DataFrame(
        {
            "subject_len": {
                "ham (0)": DATASET_STATS["avg_len"]["ham"]["subject_len"],
                "spam (1)": DATASET_STATS["avg_len"]["spam"]["subject_len"],
            },
            "body_len": {
                "ham (0)": DATASET_STATS["avg_len"]["ham"]["body_len"],
                "spam (1)": DATASET_STATS["avg_len"]["spam"]["body_len"],
            },
        }
    )
    st.dataframe(avg_len_df.style.format("{:.2f}"))
    st.bar_chart(avg_len_df)

    st.markdown("#### Contoh Raw Data (emails.csv)")
    if raw_df is not None:
        st.write("Kolom pada file asli:", list(raw_df.columns))
        st.dataframe(raw_df.head())
    else:
        st.warning(
            "File `emails.csv` tidak ditemukan, sehingga contoh raw data tidak dapat ditampilkan."
        )


# =========================
# 5. Halaman: Analyze Email
# =========================

def render_analyze_email():
    st.title("🧪 Analyze Email")
    st.write(
        """
        Masukkan detail email di bawah ini. Aplikasi akan:
        - Menghitung probabilitas spam,
        - Mengklasifikasikan SPAM / HAM,
        - Menentukan level risiko,
        - Memberikan rekomendasi aksi mitigasi.
        """
    )

    st.markdown("### Input Email")

    from_addr = st.text_input("From (alamat pengirim)", value="user@example.com")
    subject = st.text_input("Subject", value="Special offer just for you")
    body = st.text_area(
        "Body (isi email)",
        height=200,
        value="Hi, click this link to claim your free prize: http://example.com/free",
    )

    if st.button("🔎 Analisis Email"):
        if not subject and not body:
            st.warning("Silakan isi minimal subject atau body email.")
            return

        X_all, feat_info = build_feature_vector(from_addr, subject, body)

        proba_spam = model.predict_proba(X_all)[0, 1]
        pred_label = model.predict(X_all)[0]

        risk_level = get_risk_level(proba_spam)
        action_text = choose_action(risk_level, bool(feat_info["is_internal_sender"]))
        domain = extract_domain(from_addr)

        st.markdown("### 🔐 Hasil Analisis")
        st.write(f"**Probabilitas Spam:** `{proba_spam:.4f}`")
        st.write(f"**Prediksi Kelas:** `{'SPAM' if pred_label == 1 else 'HAM (normal)'}`")
        st.write(f"**Risk Level:** `{risk_level}`")
        st.write(f"**Rekomendasi Aksi Mitigasi:** **{action_text}**")

        st.markdown("### 🧩 Informasi Tambahan")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"- Domain pengirim: `{domain or '-'}`")
            st.write(f"- Panjang subject: `{feat_info['subject_len']}` karakter")
            st.write(f"- Panjang body: `{feat_info['body_len']}` karakter")
        with col2:
            st.write(f"- Jumlah URL di body: `{feat_info['num_urls']}`")
            st.write(
                f"- Pengirim internal Enron? `{bool(feat_info['is_internal_sender'])}`"
            )

        st.markdown("---")
        st.info(
            "Email dengan risk level **HIGH** dan berasal dari domain eksternal "
            "sebaiknya dikarantina dan sumber domainnya dipertimbangkan "
            "untuk masuk blocklist."
        )


# =========================
# 6. Halaman: Models & Performance
# =========================

def render_model_summary():
    st.title("📊 Models & Performance")

    st.markdown("### Perbandingan Model")
    df_metrics = pd.DataFrame(MODEL_METRICS).T
    st.dataframe(df_metrics.style.format("{:.4f}"))

    st.write(
        f"""
        Model final yang digunakan adalah **{MODEL_NAME}**, karena:
        - Memiliki Recall spam tertinggi (≈ {MODEL_METRICS[MODEL_NAME]['Recall']:.4f}),
        - Tetap menjaga Precision sangat tinggi,
        - ROC-AUC yang mendekati 1 menunjukkan pemisahan kelas yang sangat baik.
        """
    )

    st.markdown("#### Visualisasi F1-score per Model")
    f1_series = df_metrics["F1-score"]
    st.bar_chart(f1_series)

    st.markdown("#### Visualisasi ROC-AUC per Model")
    roc_series = df_metrics["ROC-AUC"]
    st.bar_chart(roc_series)

    st.info(
        "Logistic Regression dan Random Forest digunakan sebagai baseline, "
        "sedangkan XGBoost (Balanced) dipilih sebagai model produksi "
        "karena memberikan trade-off terbaik antara keamanan dan akurasi."
    )


# =========================
# 7. Halaman: Mitigation Policy & Workflow
# =========================

def render_mitigation_policy():
    st.title("🛡 Mitigation Policy & SOC Workflow")

    st.markdown("### Kebijakan Mitigasi Berbasis Skor Risiko")
    st.write(
        """
        Probabilitas spam dari model diterjemahkan menjadi level risiko dan aksi mitigasi:

        - Risk Level HIGH (P(spam) > 0,90)
          - Eksternal → `QUARANTINE`
          - Internal → `ALERT_REVIEW` (butuh review SOC analyst)
        - Risk Level MEDIUM (0,70–0,90)
          - Semua → `ALERT_REVIEW`
        - Risk Level LOW (P(spam) < 0,70)
          - Semua → `ALLOW` (email dilepas, bisa tetap dipantau)
        """
    )

    st.markdown("### Candidate Blocklist Domains (External)")
    st.write(
        "Domain berikut sering muncul sebagai sumber email yang dikarantina dan "
        "dapat dipertimbangkan sebagai candidate blocklist pada mail gateway:"
    )
    st.dataframe(BLOCKLIST_DOMAINS)

    st.markdown("### SOC Workflow")
    st.code(
        """
Incoming Email
    ↓
ML Spam Classifier (XGBoost Balanced)
    ↓
Hitung Probabilitas Spam → Tentukan Risk Level (HIGH / MEDIUM / LOW)
    ↓
+----------------------------+------------------------------+--------------------+
| HIGH + External            | HIGH + Internal              | MEDIUM             |
| → QUARANTINE               | → ALERT_REVIEW (SOC)         | → ALERT_REVIEW     |
+----------------------------+------------------------------+--------------------+
                                   ↓
                            Human-in-the-loop
                                   ↓
                      Update policy / blocklist jika perlu

LOW → ALLOW (email diteruskan ke user)
        """,
        language="text",
    )

    st.info(
        "Pendekatan ini menyeimbangkan antara keamanan (minim spam lolos) "
        "dan kelancaran operasional (email internal tidak langsung diblok)."
    )


# =========================
# 8. Halaman: About 
# =========================

def render_about():
    st.title("ℹ️ About")

    st.markdown(
        """
        ### Informasi Proyek

        - **Mata kuliah** : Keamanan Data  
        - **Topik**       : Deteksi Spam & Mitigasi Email Berbasis Machine Learning  
        - **Dataset**     : Enron Email Dataset (Kaggle)  
        - **Model Final** : XGBoost (Balanced)  

        
        ### Catatan

        Aplikasi ini adalah prototipe **Decision Support System** untuk tim SOC,
        bukan pengganti penuh sistem keamanan email produksi.
        """
    )


# =========================
# 9. Main App
# =========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Pilih Halaman",
        [
            "📌 Overview",
            "📂 Dataset & EDA",
            "🧪 Analyze Email",
            "📊 Models & Performance",
            "🛡 Mitigation Policy",
            "ℹ️ About",
        ],
    )

    if page == "📌 Overview":
        render_overview()
    elif page == "📂 Dataset & EDA":
        render_dataset_eda()
    elif page == "🧪 Analyze Email":
        render_analyze_email()
    elif page == "📊 Models & Performance":
        render_model_summary()
    elif page == "🛡 Mitigation Policy":
        render_mitigation_policy()
    else:
        render_about()


if __name__ == "__main__":
    main()
