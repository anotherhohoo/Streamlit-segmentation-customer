import streamlit as st
import pandas as pd
import joblib

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="CustomerIQ · Segmentation Engine",
    page_icon="🎯",
    layout="centered"
)

# ─── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #07090f;
    color: #e2e8f0;
}

/* ── Ambient blobs ── */
.stApp::before {
    content: '';
    position: fixed; top: -25%; left: -15%;
    width: 65vw; height: 65vw;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 68%);
    pointer-events: none; z-index: 0;
    animation: blob 14s ease-in-out infinite alternate;
}
.stApp::after {
    content: '';
    position: fixed; bottom: -20%; right: -10%;
    width: 50vw; height: 50vw;
    background: radial-gradient(circle, rgba(168,85,247,0.08) 0%, transparent 68%);
    pointer-events: none; z-index: 0;
    animation: blob 18s ease-in-out infinite alternate-reverse;
}
@keyframes blob {
    from { transform: translate(0,0) scale(1); }
    to   { transform: translate(3%,4%) scale(1.07); }
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.28);
    color: #38bdf8;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.18em; text-transform: uppercase;
    padding: 0.32rem 1rem; border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.1rem, 5.5vw, 3.4rem);
    font-weight: 800; line-height: 1.08;
    background: linear-gradient(140deg, #fff 25%, #38bdf8 75%, #a855f7 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    color: #475569;
    font-size: 0.93rem; font-weight: 300; font-style: italic;
    max-width: 420px; margin: 0 auto 0.5rem;
    line-height: 1.65;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
}

/* ── Glass card ── */
.glass {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px; padding: 1.8rem;
    margin-bottom: 1.2rem;
}

/* ── Input overrides ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #64748b !important;
    font-size: 0.75rem !important; font-weight: 500 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important; color: #f1f5f9 !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(56,189,248,0.5) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12) !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(130deg, #0ea5e9 0%, #7c3aed 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.9rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.75rem 1.5rem !important;
    box-shadow: 0 6px 28px rgba(124,58,237,0.35) !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 36px rgba(124,58,237,0.5) !important;
    transform: translateY(-2px) !important;
}

/* ── Result ── */
.result-wrap {
    animation: up 0.45s cubic-bezier(.22,.68,0,1.2) both;
}
@keyframes up {
    from { opacity:0; transform: translateY(22px) scale(0.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}
.result-card {
    background: linear-gradient(145deg, rgba(14,165,233,0.09), rgba(124,58,237,0.09));
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px; padding: 2.2rem 1.8rem;
    text-align: center;
}
.result-eyebrow {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: #475569; margin-bottom: 0.6rem;
}
.result-name {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 6vw, 3rem);
    font-weight: 800; line-height: 1.1;
    background: linear-gradient(130deg, #f8fafc 20%, #38bdf8 60%, #a855f7 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 0.3rem;
}
.result-cid {
    display: inline-block;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 6px;
    font-size: 0.7rem; color: #475569;
    padding: 0.18rem 0.6rem; margin-bottom: 1.6rem;
    letter-spacing: 0.1em;
}

/* ── Metric pills ── */
.pills {
    display: flex; flex-wrap: wrap; gap: 0.5rem;
    justify-content: center; margin-bottom: 1.5rem;
}
.pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px; padding: 0.38rem 0.85rem;
    font-size: 0.76rem; color: #64748b;
}
.pill b { color: #cbd5e1; font-weight: 500; }

/* ── Recommendation ── */
.rec {
    background: rgba(14,165,233,0.06);
    border-left: 3px solid #38bdf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem; text-align: left;
}
.rec.amber { background: rgba(245,158,11,0.06); border-left-color: #f59e0b; }
.rec-head {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 0.4rem;
}
.rec.amber .rec-head { color: #f59e0b; }
.rec-body { font-size: 0.87rem; color: #94a3b8; line-height: 1.6; }

/* ── Stat strip ── */
.stat-row {
    display: flex; gap: 0.5rem; margin-top: 1.2rem;
}
.stat-block {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.8rem 0.5rem;
    text-align: center;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-key {
    font-size: 0.62rem; color: #334155;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center; color: #1e293b;
    font-size: 0.68rem; margin-top: 3rem; padding-bottom: 2.5rem;
    letter-spacing: 0.07em;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('rf_segmentation_model.pkl')

try:
    model = load_model()
except Exception:
    st.markdown("""
    <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.25);
    border-radius:14px;padding:1.2rem 1.5rem;color:#fca5a5;font-size:0.88rem;margin-top:2rem;">
    ⚠️ <b>Model tidak ditemukan.</b> Pastikan file
    <code style="background:rgba(255,255,255,0.06);padding:0.1rem 0.4rem;border-radius:4px;">
    rf_segmentation_model.pkl</code> sudah di-upload ke Colab sebelum menjalankan app ini.
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─── Segment Config ──────────────────────────────────────────────
SEGMENTS = {
    0: {"name": "High Value",          "emoji": "👑", "color": "info",
        "rec": "Pelanggan Sultan — prioritas utama. Tawarkan layanan eksklusif, early access produk baru, dan program loyalti VIP untuk retensi jangka panjang."},
    1: {"name": "Target Growth",       "emoji": "🚀", "color": "info",
        "rec": "Potensi naik kelas tinggi. Lakukan up-selling bertahap dengan edukasi produk premium — mereka siap transisi ke tier lebih tinggi."},
    2: {"name": "Impulse Buyers",      "emoji": "⚡", "color": "info",
        "rec": "Responsif terhadap stimulus pembelian. Manfaatkan flash sale, limited-time offer, dan bundle deal untuk memaksimalkan konversi cepat."},
    3: {"name": "Moderate Spender",    "emoji": "📊", "color": "info",
        "rec": "Pelanggan stabil dan konsisten. Fokus pada cross-selling produk relevan dan program referral untuk memperluas jaringan."},
    4: {"name": "Low Engagement",      "emoji": "💤", "color": "amber",
        "rec": "Aktivitas belanja sangat rendah. Jalankan kampanye re-engagement agresif dengan diskon besar dan konten personal yang relevan."},
    5: {"name": "Highly Conservative", "emoji": "🔒", "color": "amber",
        "rec": "Sangat selektif dalam pengeluaran. Tonjolkan value proposition, social proof, dan garansi kepuasan untuk membangun kepercayaan."},
}


# ─── Hero ────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎯 &nbsp; Random Forest Classifier</div>
    <div class="hero-title">Customer Intelligence<br>Segmentation Engine</div>
    <div class="hero-sub">
        Identifikasi segmen pelanggan secara instan<br>dari profil demografi & perilaku belanja mereka.
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Input ───────────────────────────────────────────────────────
st.markdown('<div class="glass"><div class="section-label">📋 &nbsp; Profil Pelanggan</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")
with c1:
    gender_raw = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    gender_val = 1 if gender_raw == "Male" else 0
    age        = st.number_input("Usia", min_value=17, max_value=100, value=28, step=1)
with c2:
    income   = st.number_input("Pendapatan Tahunan (ribu $)", min_value=10, max_value=200, value=65, step=1)
    spending = st.slider("Skor Pengeluaran", min_value=1, max_value=100, value=55)

st.markdown('</div>', unsafe_allow_html=True)

btn = st.button("Analisis Segmen Pelanggan →", use_container_width=True)


# ─── Prediction ──────────────────────────────────────────────────
if btn:
    df_in = pd.DataFrame({
        'Gender': [gender_val], 'Age': [age],
        'Annual_Income': [income], 'Spending_Score': [spending]
    })

    with st.spinner("Memproses dengan model AI..."):
        cid        = int(model.predict(df_in)[0])
        proba      = model.predict_proba(df_in)[0]
        confidence = round(float(proba[cid]) * 100, 1)

    seg     = SEGMENTS.get(cid, {"name": f"Cluster {cid}", "emoji": "🔍",
                                  "color": "info", "rec": "Segmen tidak terdefinisi."})
    rec_cls = "amber" if seg["color"] == "amber" else ""

    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-card">
        <div class="result-eyebrow">Segmen Terdeteksi</div>
        <div class="result-name">{seg['emoji']} &nbsp;{seg['name']}</div>
        <div class="result-cid">CLUSTER · {cid}</div>
        <div class="pills">
            <div class="pill">Usia &nbsp;<b>{age} thn</b></div>
            <div class="pill">Gender &nbsp;<b>{gender_raw}</b></div>
            <div class="pill">Income &nbsp;<b>${income}K</b></div>
            <div class="pill">Spending &nbsp;<b>{spending}/100</b></div>
        </div>
        <div class="rec {rec_cls}">
            <div class="rec-head">💡 Rekomendasi Strategi</div>
            <div class="rec-body">{seg['rec']}</div>
        </div>
        <div class="stat-row">
            <div class="stat-block">
                <div class="stat-val">{confidence}%</div>
                <div class="stat-key">Confidence</div>
            </div>
            <div class="stat-block">
                <div class="stat-val">{cid}</div>
                <div class="stat-key">Cluster ID</div>
            </div>
            <div class="stat-block">
                <div class="stat-val">{spending}</div>
                <div class="stat-key">Spending</div>
            </div>
            <div class="stat-block">
                <div class="stat-val">${income}K</div>
                <div class="stat-key">Income</div>
            </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CustomerIQ Engine &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
