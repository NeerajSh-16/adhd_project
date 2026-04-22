"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ADHD Pre-Screening Assessment Tool — Streamlit App                        ║
║  Built for: Final Year University Project                                  ║
║  Model: Logistic Regression (81.4% Accuracy)                               ║
║  Features: ASRS Part B + BAI + BDI + AAS + TF-IDF Clinical Notes           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pickle
import os
import scipy.sparse as sp
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScreen — ADHD Pre-Screening Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Clinical, calming theme with medical blues
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import premium font ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Hide default Streamlit chrome ──────────────────────────────────────── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Global typography & background ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #f0f4f8 0%, #e8eef5 50%, #f5f7fa 100%);
}

/* ── Sidebar styling ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2940 0%, #163d5c 40%, #1a4a6e 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] label {
    color: #c8dce8 !important;
    font-weight: 400;
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    color: #ffffff !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Card containers ────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    margin-bottom: 1rem;
    overflow: hidden;
}
div[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #1e3a5f;
    padding: 0.8rem 1rem;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    text-align: center;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #1a4971 !important;
    font-weight: 700 !important;
}

/* ── Primary action button ──────────────────────────────────────────────── */
div.stButton > button[kind="primary"],
div.stButton > button {
    background: linear-gradient(135deg, #1a6fb5 0%, #2584d0 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2.5rem;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(26,111,181,0.3);
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1);
    width: 100%;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #155c98 0%, #1e73b8 100%);
    box-shadow: 0 6px 20px rgba(26,111,181,0.4);
    transform: translateY(-1px);
}

/* ── Text input / number input / textarea styling ───────────────────────── */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1.5px solid #d0d9e4 !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: #2584d0 !important;
    box-shadow: 0 0 0 3px rgba(37,132,208,0.12) !important;
}

/* ── Slider customisation ───────────────────────────────────────────────── */
div[data-testid="stSlider"] > div > div > div {
    color: #1a4971;
}

/* ── Alert boxes polish ─────────────────────────────────────────────────── */
.stAlert {
    border-radius: 12px !important;
}

/* ── Tab styling ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    font-weight: 500;
    color: #64748b;
}
.stTabs [aria-selected="true"] {
    color: #1a4971 !important;
    font-weight: 600;
    border-bottom: 3px solid #2584d0;
}

/* ── Utility classes injected via markdown ──────────────────────────────── */
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e0f0ff 0%, #d0e7fb 100%);
    color: #1a5d96;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #0f2940;
    line-height: 1.18;
    margin-bottom: 0.5rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #4a6a8a;
    font-weight: 400;
    line-height: 1.6;
    max-width: 640px;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #7a8fa8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.card-glass {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a6fb5;
    line-height: 1;
}
.stat-label {
    font-size: 0.8rem;
    color: #7a8fa8;
    font-weight: 500;
    margin-top: 0.25rem;
}
.risk-high {
    background: linear-gradient(135deg, #fff5f5 0%, #fee2e2 100%);
    border-left: 4px solid #e53e3e;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.risk-low {
    background: linear-gradient(135deg, #f0fff4 0%, #dcfce7 100%);
    border-left: 4px solid #38a169;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.disclaimer-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 0.82rem;
    color: #64748b;
    line-height: 1.65;
    margin-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():
    """
    Load the saved model, vectorizer, and feature selector.

    TODO ────────────────────────────────────────────────────────────────────
    1. Save your fitted TfidfVectorizer and SelectKBest from the training
       notebook alongside the model in the models/ folder:
           pickle.dump(vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))
           pickle.dump(selector,   open('models/feature_selector.pkl', 'wb'))
    2. Update the paths below if yours differ.
    ────────────────────────────────────────────────────────────────────────
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # ── Load logistic regression model ───────────────────────────────────
    model_path = os.path.join(models_dir, "logistic_regression.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ── Load fitted TF-IDF vectorizer ────────────────────────────────────
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # ── Load fitted SelectKBest selector ─────────────────────────────────
    selector_path = os.path.join(models_dir, "feature_selector.pkl")
    with open(selector_path, "rb") as f:
        selector = pickle.load(f)

    return model, vectorizer, selector


model, vectorizer, selector = load_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE MAPS  —  must match exact LabelEncoder order from training
# ─────────────────────────────────────────────────────────────────────────────

# TODO: Verify these label-encoded values match your training run.
# LabelEncoder sorts categories alphabetically by default.
# Run `le.classes_` on each fitted encoder to confirm the mapping.
SEX_MAP = {"Female": 0, "Male": 1}
DIAGNOSED_MAP = {"No": 0, "Yes": 1}
MEDICATION_MAP = {"No": 0, "Yes": 1}
PRIOR_MH_MAP = {"No": 0, "Yes": 1}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.6rem;">
        <span style="font-size:2.6rem;">🧠</span>
        <h2 style="margin:0.3rem 0 0; font-size:1.4rem; font-weight:800;
                    letter-spacing:-0.02em;">NeuroScreen</h2>
        <p style="font-size:0.76rem; opacity:0.65; margin-top:0.2rem;
                  letter-spacing:0.04em;">ADHD PRE-SCREENING TOOL</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Project Overview", "📝 Patient Assessment", "🔬 Analysis & Results"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Sidebar footer — model info
    st.markdown("""
    <div style="padding:0.5rem 0; font-size:0.72rem; opacity:0.5;">
        <p>Model: Logistic Regression</p>
        <p>Accuracy: 81.4%</p>
        <p>Features: 20 (selected)</p>
        <p style="margin-top:0.6rem;">Final Year Project · 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — PROJECT OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

if page == "📊 Project Overview":

    # Hero section
    st.markdown('<div class="hero-badge">University Research Project</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Intelligent ADHD Pre-Screening<br>'
                'for University Students</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'An NLP-powered clinical decision support tool that combines '
        'psychometric assessments with natural language analysis to '
        'pre-screen university students for ADHD — reducing diagnostic '
        'bottlenecks at campus health clinics.'
        '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key statistics row ───────────────────────────────────────────────
    cols = st.columns(4)
    stats = [
        ("81.4%", "Model Accuracy"),
        ("506", "Survey Responses"),
        ("20", "Selected Features"),
        ("5", "Models Compared"),
    ]
    for col, (value, label) in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div class="card-glass" style="text-align:center; padding:1.5rem 1rem;">
                <div class="stat-number">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Why this matters ─────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-label">The Problem</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card-glass">
            <h3 style="color:#0f2940; margin-top:0;">Why Pre-Screening Matters</h3>
            <p style="color:#4a6a8a; line-height:1.7; font-size:0.95rem;">
                ADHD is one of the most under-diagnosed conditions among university
                students. Studies estimate that <strong>2–8%</strong> of university
                populations have undiagnosed ADHD, leading to academic struggles,
                anxiety, and dropout. Campus clinics face long waiting lists, with
                students often waiting <strong>6–12 months</strong> for a formal
                assessment.
            </p>
            <p style="color:#4a6a8a; line-height:1.7; font-size:0.95rem;">
                This tool acts as a <strong>first-line triage system</strong> —
                analysing validated psychometric instruments (ASRS, BAI, BDI, AAS)
                alongside free-text clinical notes to flag high-risk individuals for
                priority referral.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Data Pipeline</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card-glass">
            <h3 style="color:#0f2940; margin-top:0;">How It Works</h3>
            <ol style="color:#4a6a8a; line-height:2; font-size:0.92rem; padding-left:1.2rem;">
                <li><strong>Data Collection</strong> — 506 student survey responses</li>
                <li><strong>Feature Engineering</strong> — Numeric scores + TF-IDF on clinical text</li>
                <li><strong>Feature Selection</strong> — Mutual Information (124 → 20 features)</li>
                <li><strong>Model Training</strong> — Logistic Regression, SVM, XGBoost, MLP, Stacking</li>
                <li><strong>Evaluation</strong> — 5-fold cross-validation, ROC/PR curves</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature breakdown ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Feature Categories</div>',
                unsafe_allow_html=True)

    feat_cols = st.columns(4)
    features_info = [
        ("🧪", "ASRS Part B", "Items 7–18",
         "Adult ADHD Self-Report Scale inattention & hyperactivity items"),
        ("😰", "BAI Scores", "Total + Items 4, 8",
         "Beck Anxiety Inventory — comorbidity indicators"),
        ("😔", "BDI Scores", "Total + Item 19",
         "Beck Depression Inventory — mood-related features"),
        ("📝", "Clinical Text", "TF-IDF Vectorised",
         "Free-text diagnosis field processed with NLP"),
    ]
    for col, (icon, title, subtitle, desc) in zip(feat_cols, features_info):
        with col:
            st.markdown(f"""
            <div class="card-glass" style="text-align:center; min-height:220px;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">{icon}</div>
                <h4 style="color:#0f2940; margin:0 0 0.15rem;">{title}</h4>
                <p style="color:#2584d0; font-size:0.78rem; font-weight:600;
                   margin:0 0 0.6rem;">{subtitle}</p>
                <p style="color:#64748b; font-size:0.84rem; line-height:1.55;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ───────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚕️ Medical Disclaimer:</strong> This tool is a predictive screening
        instrument based on the ASRS Part A scoring threshold and is intended for
        educational and research purposes only. It does <em>not</em> constitute a
        clinical diagnosis of ADHD. Any positive result should be followed up with a
        licensed healthcare professional for formal evaluation.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — PATIENT ASSESSMENT FORM
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📝 Patient Assessment":

    st.markdown('<div class="hero-badge">Patient Assessment</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Clinical Data Entry</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">'
        'Complete all sections below. Psychometric scores should be taken '
        'directly from validated instruments. All fields are required.'
        '</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Layout: two main columns ─────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    # ────────────────────── LEFT COLUMN ──────────────────────────────────
    with col_left:

        # ── Demographics & History ───────────────────────────────────────
        with st.expander("👤  Demographics & History", expanded=True):
            demo_c1, demo_c2 = st.columns(2)
            with demo_c1:
                sex = st.selectbox("Sex", list(SEX_MAP.keys()),
                                   help="Biological sex as recorded.")
                prior_mh = st.selectbox(
                    "Prior Mental Health History",
                    list(PRIOR_MH_MAP.keys()),
                    help="Any mental health difficulties before university?")
            with demo_c2:
                diagnosed = st.selectbox(
                    "Prior Diagnosis",
                    list(DIAGNOSED_MAP.keys()),
                    help="Ever been diagnosed with a mental illness?")
                medication = st.selectbox(
                    "Current Psychiatric Medication",
                    list(MEDICATION_MAP.keys()),
                    help="Currently using prescribed psychiatric medication?")

        # ── BAI Scores ───────────────────────────────────────────────────
        with st.expander("😰  Beck Anxiety Inventory (BAI)", expanded=True):
            bai_total = st.slider(
                "BAI Total Score", 0, 63, 10,
                help="Sum of all 21 BAI items (0–63)")
            bai_c1, bai_c2 = st.columns(2)
            with bai_c1:
                bai_item_4 = st.slider(
                    "BAI Item 4 — Unable to Relax", 0, 3, 0,
                    help="0 = Not at all, 3 = Severely")
            with bai_c2:
                bai_item_8 = st.slider(
                    "BAI Item 8 — Unsteady", 0, 3, 0,
                    help="0 = Not at all, 3 = Severely")

        # ── BDI Scores ───────────────────────────────────────────────────
        with st.expander("😔  Beck Depression Inventory (BDI)", expanded=True):
            bdi_total = st.slider(
                "BDI Total Score", 0, 63, 8,
                help="Sum of all 21 BDI-II items (0–63)")
            bdi_item_19 = st.slider(
                "BDI Item 19 — Concentration Difficulty", 0, 3, 0,
                help="0 = I can concentrate as well as ever, "
                     "3 = I can't concentrate on anything")

    # ────────────────────── RIGHT COLUMN ─────────────────────────────────
    with col_right:

        # ── ASRS Part B ──────────────────────────────────────────────────
        with st.expander("🧪  ASRS Part B — Items 7 to 18", expanded=True):
            st.caption("Rate each item: **0** = Never · **1** = Rarely · "
                       "**2** = Sometimes · **3** = Often · **4** = Very Often")

            asrs_labels = {
                7:  "Difficulty getting things in order for a task",
                8:  "Difficulty keeping attention on tasks",
                9:  "Difficulty concentrating on what people say",
                10: "Misplace or have difficulty finding things at home or work",
                11: "Distracted by activity or noise around you",
                12: "Leave your seat in meetings or situations",
                13: "Feel restless or fidgety",
                14: "Difficulty unwinding and relaxing when free time",
                15: "Find yourself talking too much in social situations",
                16: "Finish other people's sentences before they can",
                17: "Difficulty waiting your turn in queued situations",
                18: "Interrupt others when they are busy",
            }

            asrs_values = {}
            # Arrange ASRS items in a two-column grid
            asrs_items = list(asrs_labels.items())
            for i in range(0, len(asrs_items), 2):
                a_c1, a_c2 = st.columns(2)
                item_num, label = asrs_items[i]
                with a_c1:
                    asrs_values[item_num] = st.slider(
                        f"Item {item_num}", 0, 4, 0,
                        key=f"asrs_{item_num}",
                        help=label)
                if i + 1 < len(asrs_items):
                    item_num2, label2 = asrs_items[i + 1]
                    with a_c2:
                        asrs_values[item_num2] = st.slider(
                            f"Item {item_num2}", 0, 4, 0,
                            key=f"asrs_{item_num2}",
                            help=label2)

        # ── AAS Scores ───────────────────────────────────────────────────
        with st.expander("📎  Adult Attachment Scale (AAS)", expanded=True):
            aas_item_3 = st.slider(
                "AAS Item 3", 1, 5, 3,
                help="Likert scale: 1 = Not at all characteristic, "
                     "5 = Very characteristic")
            aas_c1, aas_c2 = st.columns(2)
            with aas_c1:
                aas_item_4 = st.slider(
                    "AAS Item 4", 1, 5, 3,
                    help="1–5 Likert scale")
            with aas_c2:
                aas_item_6 = st.slider(
                    "AAS Item 6", 1, 5, 3,
                    help="1–5 Likert scale")

        # ── Clinical Notes (free text) ───────────────────────────────────
        with st.expander("💬  Clinical Notes (Free Text)", expanded=True):
            clinical_text = st.text_area(
                "If you have been diagnosed formally or informally, "
                "please list the diagnosis/diagnoses:",
                height=120,
                placeholder="e.g., anxiety, depression, autism spectrum disorder…\n"
                            "Type 'none' if not applicable.",
                help="This text is processed with TF-IDF for NLP features.")

    # ── Submit button ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        run_assessment = st.button("🔬  Run Assessment", use_container_width=True)

    # ── Process on submit ────────────────────────────────────────────────
    if run_assessment:
        with st.spinner("Processing psychometric data and analysing clinical text…"):

            # ── 1. Build numeric feature vector (18 values) ──────────────
            # Order must match feature_engineering.py: ASRS B (7-18),
            # BAI (total, item4, item8), BDI (total, item19),
            # AAS (item3, item4, item6)
            numeric_features = []
            for i in range(7, 19):
                numeric_features.append(asrs_values.get(i, 0))
            numeric_features.extend([
                bai_total, bai_item_4, bai_item_8,
                bdi_total, bdi_item_19,
                aas_item_3, aas_item_4, aas_item_6,
            ])

            # ── 2. Encode categorical features (4 values) ───────────────
            cat_features = [
                SEX_MAP[sex],
                DIAGNOSED_MAP[diagnosed],
                MEDICATION_MAP[medication],
                PRIOR_MH_MAP[prior_mh],
            ]

            # ── 3. TF-IDF transform the clinical text ───────────────────
            text_input = clinical_text.strip().lower() if clinical_text else "none"

            # TODO ──────────────────────────────────────────────────────
            # Replace this placeholder with actual vectorizer transform:
            #     tfidf_vector = vectorizer.transform([text_input])
            # For now, create a zero vector as placeholder.
            # ──────────────────────────────────────────────────────────
            if vectorizer is not None:
                tfidf_vector = vectorizer.transform([text_input])
            else:
                # Placeholder: 100 zeros matching max_features=100 in training
                tfidf_vector = sp.csr_matrix(np.zeros((1, 100)))
                st.warning("⚠️ TF-IDF vectorizer not loaded. Using placeholder "
                           "zeros. Save your fitted vectorizer to "
                           "`models/tfidf_vectorizer.pkl`.")

            # ── 4. Combine into full feature matrix ──────────────────────
            numeric_sparse = sp.csr_matrix(
                np.array(numeric_features, dtype=float).reshape(1, -1))
            cat_sparse = sp.csr_matrix(
                np.array(cat_features, dtype=float).reshape(1, -1))
            X_full = sp.hstack([numeric_sparse, cat_sparse, tfidf_vector])

            # ── 5. Apply feature selection ───────────────────────────────
            # TODO ──────────────────────────────────────────────────────
            # Replace this placeholder with actual selector transform:
            #     X_selected = selector.transform(X_full)
            # ──────────────────────────────────────────────────────────
            if selector is not None:
                X_selected = selector.transform(X_full)
            else:
                X_selected = X_full
                st.warning("⚠️ Feature selector not loaded. Passing all "
                           "features. Save your fitted selector to "
                           "`models/feature_selector.pkl`.")

            # ── 6. Predict ───────────────────────────────────────────────
            try:
                probability = model.predict_proba(X_selected)[0][1]
                prediction = int(probability >= 0.5)

                # Store results in session state so they persist on page switch
                st.session_state["result"] = {
                    "probability": probability,
                    "prediction": prediction,
                    "numeric": numeric_features,
                    "categorical": cat_features,
                    "text": text_input,
                }
                st.success("✅ Assessment complete! Navigate to "
                           "**🔬 Analysis & Results** to view the report.")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}\n\n"
                         f"This likely means the feature dimensions don't "
                         f"match the trained model. Ensure the vectorizer "
                         f"and selector are loaded correctly.")


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ANALYSIS & RESULTS
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Analysis & Results":

    st.markdown('<div class="hero-badge">Assessment Report</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Analysis & Results</div>',
                unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📋 No assessment data available yet. Please complete the "
                "**Patient Assessment** form first and click **Run Assessment**.")
    else:
        result = st.session_state["result"]
        prob = result["probability"]
        pred = result["prediction"]
        prob_pct = prob * 100

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Top metrics row ──────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("ADHD Probability", f"{prob_pct:.1f}%")
        with m2:
            risk_label = "Elevated" if pred == 1 else "Low"
            st.metric("Risk Level", risk_label)
        with m3:
            st.metric("Model Confidence", f"{max(prob, 1-prob)*100:.1f}%")
        with m4:
            st.metric("Model Used", "Logistic Reg.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge + Risk Profile side by side ────────────────────────────
        gauge_col, profile_col = st.columns([3, 2], gap="large")

        with gauge_col:
            st.markdown('<div class="section-label">Probability Gauge</div>',
                        unsafe_allow_html=True)

            # Build a Plotly gauge chart
            if prob_pct < 30:
                gauge_color = "#38a169"  # green
            elif prob_pct < 50:
                gauge_color = "#d69e2e"  # amber
            elif prob_pct < 70:
                gauge_color = "#e53e3e"  # red
            else:
                gauge_color = "#c53030"  # dark red

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 52, "color": "#0f2940",
                                                 "family": "Inter"}},
                delta={"reference": 50, "increasing": {"color": "#e53e3e"},
                       "decreasing": {"color": "#38a169"},
                       "suffix": "%", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1,
                             "tickcolor": "#cbd5e0",
                             "tickfont": {"size": 12, "color": "#94a3b8"}},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "#f1f5f9",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#dcfce7"},
                        {"range": [30, 50], "color": "#fef9c3"},
                        {"range": [50, 70], "color": "#fee2e2"},
                        {"range": [70, 100], "color": "#fecaca"},
                    ],
                    "threshold": {
                        "line": {"color": "#1a4971", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
                title={"text": "ADHD Likelihood Score",
                       "font": {"size": 16, "color": "#64748b",
                                "family": "Inter"}},
            ))
            fig.update_layout(
                height=340,
                margin=dict(l=30, r=30, t=60, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter"},
            )
            st.plotly_chart(fig, use_container_width=True)

        with profile_col:
            st.markdown('<div class="section-label">Risk Profile</div>',
                        unsafe_allow_html=True)

            if pred == 1:
                st.markdown(f"""
                <div class="risk-high">
                    <h3 style="color:#c53030; margin:0 0 0.5rem;">
                        ⚠️ Elevated Risk Detected
                    </h3>
                    <p style="color:#7f1d1d; font-size:0.92rem; line-height:1.7;">
                        The model predicts a <strong>{prob_pct:.1f}%</strong>
                        probability of ADHD based on the ASRS Part A screening
                        threshold. This individual may benefit from
                        <strong>priority referral</strong> for a formal clinical
                        assessment.
                    </p>
                    <p style="color:#7f1d1d; font-size:0.85rem; line-height:1.6;
                       margin-top:0.5rem;">
                        <strong>Suggested next steps:</strong><br>
                        • Schedule comprehensive ADHD evaluation<br>
                        • Review comorbid anxiety/depression indicators<br>
                        • Consider academic accommodations evaluation
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-low">
                    <h3 style="color:#276749; margin:0 0 0.5rem;">
                        ✅ Low Risk Indicated
                    </h3>
                    <p style="color:#22543d; font-size:0.92rem; line-height:1.7;">
                        The model predicts a <strong>{prob_pct:.1f}%</strong>
                        probability of ADHD based on the ASRS Part A screening
                        threshold. Current scores do not indicate elevated risk.
                    </p>
                    <p style="color:#22543d; font-size:0.85rem; line-height:1.6;
                       margin-top:0.5rem;">
                        <strong>Recommendation:</strong><br>
                        • Routine follow-up as needed<br>
                        • Re-screen if new symptoms emerge<br>
                        • Continue monitoring well-being
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Model accuracy context
            st.markdown("""
            <div style="background:#f8fafc; border:1px solid #e2e8f0;
                        border-radius:12px; padding:1rem; margin-top:1rem;">
                <p style="color:#64748b; font-size:0.8rem; margin:0;">
                    <strong>Model Performance Context</strong><br>
                    Accuracy: 81.4% · Validated via 5-fold cross-validation ·
                    Feature selection reduced 124 → 20 features using Mutual
                    Information scoring.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Input Summary ────────────────────────────────────────────────
        st.markdown('<div class="section-label">Input Summary</div>',
                    unsafe_allow_html=True)

        with st.expander("📋  View Submitted Assessment Data", expanded=False):
            sum_c1, sum_c2, sum_c3 = st.columns(3)

            with sum_c1:
                st.markdown("**ASRS Part B Scores (Items 7–18)**")
                for i, val in enumerate(result["numeric"][:12], start=7):
                    bar_len = "█" * val + "░" * (4 - val)
                    st.code(f"Item {i:2d}: {bar_len}  ({val})", language=None)

            with sum_c2:
                st.markdown("**Anxiety & Depression**")
                labels = ["BAI Total", "BAI Item 4", "BAI Item 8",
                          "BDI Total", "BDI Item 19"]
                for label, val in zip(labels, result["numeric"][12:17]):
                    st.markdown(f"- **{label}:** {val}")

                st.markdown("**Attachment**")
                aas_labels = ["AAS Item 3", "AAS Item 4", "AAS Item 6"]
                for label, val in zip(aas_labels, result["numeric"][17:20]):
                    st.markdown(f"- **{label}:** {val}")

            with sum_c3:
                st.markdown("**Demographics**")
                cat_labels = ["Sex", "Prior Diagnosis", "Medication",
                              "Prior MH History"]
                cat_maps = [
                    {v: k for k, v in SEX_MAP.items()},
                    {v: k for k, v in DIAGNOSED_MAP.items()},
                    {v: k for k, v in MEDICATION_MAP.items()},
                    {v: k for k, v in PRIOR_MH_MAP.items()},
                ]
                for label, val, rev_map in zip(cat_labels,
                                                result["categorical"],
                                                cat_maps):
                    st.markdown(f"- **{label}:** {rev_map.get(val, val)}")

                st.markdown("**Clinical Text**")
                st.text(result["text"][:200] if result["text"] else "none")

        # ── Disclaimer ───────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer-box">
            <strong>⚕️ Medical Disclaimer:</strong> This result is generated by a
            machine learning model trained on survey data using the ASRS Part A
            scoring threshold as its target. It is a <strong>predictive screening
            tool</strong> only and does <strong>not</strong> replace formal clinical
            diagnosis. ADHD can only be diagnosed by a qualified healthcare
            professional through comprehensive evaluation including clinical
            interviews, behavioural observations, and standardised testing. If you
            or a student scores highly, please seek a referral to your campus
            health services or a licensed psychologist/psychiatrist.
        </div>
        """, unsafe_allow_html=True)
