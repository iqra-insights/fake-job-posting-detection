import streamlit as st
import joblib
import re
import time
import os

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Support running from any directory — look in the model/ folder next to this script
    base = os.path.dirname(os.path.abspath(__file__))
    model_path      = os.path.join(base, "model", "fake_job_model.pkl")
    vectorizer_path = os.path.join(base, "model", "tfidf_vectorizer.pkl")
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}\n\nMake sure `fake_job_model.pkl` and `tfidf_vectorizer.pkl` are inside the `model/` folder next to `app.py`.")
    st.stop()
    
# ── Text Cleaner ─────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 760px; }

.stApp {
    background: #070a12;
    background-image:
        radial-gradient(ellipse 80% 40% at 50% 0%, rgba(79,70,229,0.12), transparent),
        radial-gradient(ellipse 50% 30% at 90% 90%, rgba(16,185,129,0.06), transparent);
}

[data-testid="stSidebar"] {
    background: #0b0f1c !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebarContent"] { padding: 20px 14px; }

.hero {
    background: linear-gradient(160deg, #0d1130 0%, #1a1660 60%, #0d1130 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 20px;
    padding: 36px 32px 32px;
    text-align: center;
    margin-bottom: 20px;
}
.hero-icon { font-size: 44px; margin-bottom: 10px; display: block; }
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem !important; font-weight: 800 !important;
    color: #f1f5f9 !important; margin: 0 0 8px !important;
}
.hero h1 span { color: #818cf8; }
.hero p { color: #94a3b8; font-size: 0.95rem; margin: 0; }

.stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 20px; }
.stat { background: rgba(15,23,42,0.7); border: 1px solid rgba(99,102,241,0.18);
        border-radius: 12px; padding: 12px; text-align: center; }
.stat b { font-family: 'Syne', sans-serif; font-size: 1.3rem; color: #818cf8; display: block; }
.stat small { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }

.lbl { font-size: 0.68rem; letter-spacing: 2px; text-transform: uppercase;
       color: #6366f1; margin-bottom: 8px; font-weight: 600; }

.stTextArea textarea {
    background: rgba(15,23,42,0.85) !important;
    border: 1px solid rgba(99,102,241,0.22) !important;
    border-radius: 14px !important; color: #e2e8f0 !important;
    font-size: 0.93rem !important; padding: 14px !important; line-height: 1.65 !important;
}
.stTextArea textarea:focus { border-color: #6366f1 !important; outline: none !important; }
.stTextArea textarea::placeholder { color: #334155 !important; }
.stTextArea label { display: none; }

div[data-testid="column"] .stButton > button {
    border: 1px solid rgba(99,102,241,0.28) !important;
    background: rgba(15,23,42,0.8) !important; color: #94a3b8 !important;
    border-radius: 10px !important; font-size: 0.88rem !important; padding: 9px 14px !important;
    width: 100%;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #6366f1 !important; color: #a5b4fc !important;
}

.detect-wrap .stButton > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    border: none !important; color: #fff !important;
    border-radius: 12px !important; font-size: 1rem !important;
    font-weight: 600 !important; padding: 13px !important;
    width: 100%; box-shadow: 0 4px 20px rgba(99,102,241,0.35);
}
.detect-wrap .stButton > button:hover {
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px);
}

.card-fake {
    background: linear-gradient(135deg, #1c0808, #2f0f0f);
    border: 1px solid #dc2626; border-radius: 18px;
    padding: 28px; text-align: center;
    box-shadow: 0 0 36px rgba(220,38,38,0.18);
}
.card-real {
    background: linear-gradient(135deg, #071a0f, #0e2d18);
    border: 1px solid #16a34a; border-radius: 18px;
    padding: 28px; text-align: center;
    box-shadow: 0 0 36px rgba(22,163,74,0.18);
}
.card-fake .ci { font-size: 2.6rem; display: block; margin-bottom: 8px; }
.card-real .ci { font-size: 2.6rem; display: block; margin-bottom: 8px; }
.card-fake h2 { color: #f87171; font-family: 'Syne', sans-serif; font-size: 1.6rem; margin: 0 0 6px; }
.card-real h2 { color: #4ade80; font-family: 'Syne', sans-serif; font-size: 1.6rem; margin: 0 0 6px; }
.card-fake .cf { color: #fca5a5; font-size: 1.05rem; font-weight: 600; }
.card-real .cf { color: #86efac; font-size: 1.05rem; font-weight: 600; }
.card-fake .cm { color: #94a3b8; font-size: 0.85rem; margin-top: 8px; }
.card-real .cm { color: #94a3b8; font-size: 0.85rem; margin-top: 8px; }

.prob-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
.prob-card { background: rgba(15,23,42,0.85); border: 1px solid rgba(255,255,255,0.06);
             border-radius: 12px; padding: 14px; text-align: center; }
.prob-card b { font-family: 'Syne', sans-serif; font-size: 1.8rem; display: block; }
.prob-card small { font-size: 0.72rem; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }
.c-real { color: #4ade80; }
.c-fake { color: #f87171; }

.prog { background: rgba(15,23,42,0.8); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 14px 18px; margin-top: 12px; }
.prog-lbl { display: flex; justify-content: space-between; font-size: 0.8rem; color: #64748b; margin-bottom: 5px; }
.prog-bg { height: 7px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; margin-bottom: 10px; }
.prog-real { height: 100%; border-radius: 99px; background: #16a34a; }
.prog-fake { height: 100%; border-radius: 99px; background: #dc2626; }

.warn { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 14px; }
.warn-item { background: rgba(15,23,42,0.8); border: 1px solid rgba(239,68,68,0.15);
             border-radius: 10px; padding: 9px 12px; font-size: 0.82rem; color: #94a3b8; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.25), transparent); margin: 20px 0; }
.footer { text-align: center; color: #1e293b; font-size: 0.75rem; padding: 16px 0 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px'>
      <div style='font-size:2rem;'>🛡️</div>
      <div style='font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:#818cf8;margin:6px 0 2px'>Job Guard AI</div>
      <div style='font-size:0.72rem;color:#334155'>Powered by ML + NLP</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:0 0 14px'>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>About</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.82rem;color:#64748b;line-height:1.65'>Detects whether a job posting is genuine or fraudulent using Machine Learning and Natural Language Processing.</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>Model info</div>", unsafe_allow_html=True)
    st.markdown("""
    <table style='width:100%;font-size:0.8rem;border-collapse:collapse'>
      <tr><td style='color:#334155;padding:3px 0'>Algorithm</td><td style='text-align:right;color:#c7d2fe'>Logistic Regression</td></tr>
      <tr><td style='color:#334155;padding:3px 0'>Vectorizer</td><td style='text-align:right;color:#c7d2fe'>TF-IDF</td></tr>
      <tr><td style='color:#334155;padding:3px 0'>Accuracy</td><td style='text-align:right;color:#4ade80;font-weight:600'>100%</td></tr>
      <tr><td style='color:#334155;padding:3px 0'>Training data</td><td style='text-align:right;color:#c7d2fe'>10,000 rows</td></tr>
    </table>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:12px 0'>", unsafe_allow_html=True)
    st.markdown("<div class='lbl'>🚨 Fake job signs</div>", unsafe_allow_html=True)
    for s in ["Earn ₹50,000/week from home", "No experience required",
              "Pay a fee to get the job", "Urgent — apply today only",
              "No interview needed", "WhatsApp-only contact"]:
        st.markdown(f"<div style='font-size:0.8rem;color:#334155;padding:2px 0'>🔴 {s}</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <span class='hero-icon'>🛡️</span>
  <h1>Fake Job <span>Detector</span></h1>
  <p>Paste any job description — instantly know if it's real or fraudulent</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────
st.markdown("""
<div class='stats'>
  <div class='stat'><b>100%</b><small>Accuracy</small></div>
  <div class='stat'><b>10K</b><small>Trained on</small></div>
  <div class='stat'><b>NLP</b><small>Technology</small></div>
  <div class='stat'><b>AI</b><small>Powered</small></div>
</div>
""", unsafe_allow_html=True)

# ── Sample Buttons ────────────────────────────────────────────────
st.markdown("<div class='lbl'>Quick test examples</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("✅ Load Real Job Example"):
        st.session_state.job_text = """Senior Software Engineer at Infosys located in Bengaluru Karnataka.

We are hiring a qualified professional with 3 to 6 years of experience in Python Django REST framework.

Required qualification: Bachelors degree in Computer Science.

Responsibilities include project delivery, team collaboration and client communication.

Benefits include health insurance, provident fund, paid leaves and annual bonus.

Salary: 10 to 18 LPA based on experience. Apply through our official careers portal."""

with col2:
    if st.button("❌ Load Fake Job Example"):
        st.session_state.job_text = """URGENT HIRING!!!

Earn 50000 per week working from home!
No experience needed. No degree required.
No interview. Immediate joining.

Pay registration fee 2000 now to get started.
WhatsApp us immediately. Limited seats apply today only!"""

# ── Text Input ────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='lbl'>Enter job description</div>", unsafe_allow_html=True)

job_description = st.text_area(
    "job",
    value=st.session_state.get("job_text", ""),
    height=210,
    placeholder="Paste the job posting here...\n\nExample: Senior Python Developer at Infosys, Bengaluru. 4+ years experience required. Salary 15 LPA. Apply via official careers portal.",
    label_visibility="collapsed"
)

# ── Detect Button ─────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("<div class='detect-wrap'>", unsafe_allow_html=True)
detect = st.button("🔎 Analyse & Detect", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Result ────────────────────────────────────────────────────────
if detect:
    if not job_description.strip():
        st.warning("⚠️ Please enter a job description first.")
    else:
        with st.spinner("Analysing job posting..."):
            time.sleep(0.5)
            cleaned     = clean_text(job_description)
            vectorized  = vectorizer.transform([cleaned])
            prediction  = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0]
            real_pct    = round(probability[0] * 100, 1)
            fake_pct    = round(probability[1] * 100, 1)
            confidence  = round(max(probability) * 100, 2)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(f"""
            <div class='card-fake'>
              <span class='ci'>⚠️</span>
              <h2>FAKE Job Posting</h2>
              <p class='cf'>Confidence: {confidence}%</p>
              <p class='cm'>This posting shows signs of fraud.<br>
              Do <b>NOT</b> share personal details or pay any registration fees.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card-real'>
              <span class='ci'>✅</span>
              <h2>REAL Job Posting</h2>
              <p class='cf'>Confidence: {confidence}%</p>
              <p class='cm'>This posting appears to be legitimate.<br>
              Always verify through the company's <b>official website</b> before applying.</p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='prob-row'>
          <div class='prob-card'><b class='c-real'>{real_pct}%</b><small>✅ Real probability</small></div>
          <div class='prob-card'><b class='c-fake'>{fake_pct}%</b><small>⚠️ Fake probability</small></div>
        </div>
        <div class='prog'>
          <div class='prog-lbl'><span>✅ Real</span><span>{real_pct}%</span></div>
          <div class='prog-bg'><div class='prog-real' style='width:{real_pct}%'></div></div>
          <div class='prog-lbl'><span>⚠️ Fake</span><span>{fake_pct}%</span></div>
          <div class='prog-bg'><div class='prog-fake' style='width:{fake_pct}%'></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='lbl'>🚨 Common signs of a fake job</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='warn'>
          <div class='warn-item'>💰 Unrealistic salary claims</div>
          <div class='warn-item'>📋 No experience required</div>
          <div class='warn-item'>⚡ Urgent — apply today only</div>
          <div class='warn-item'>📧 Gmail or personal email</div>
          <div class='warn-item'>💳 Pay a fee to get the job</div>
          <div class='warn-item'>🏠 Vague work from home offer</div>
        </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class='divider'></div>
<div class='footer'>Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; Always verify jobs through official company websites</div>
""", unsafe_allow_html=True)
