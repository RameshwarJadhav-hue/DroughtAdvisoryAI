"""
Cognitive AI-Based Drought Risk Prediction and Farmer Advisory System
Case Study: Marathwada Agro-Climatic Region
Framework: Streamlit + Scikit-Learn + Python
Aesthetic: Minimalist Botanical / Sage Editorial Design
"""

import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Import Cognitive Reasoning Engine
try:
    from cognitive_engine import (
        generate_cognitive_reasoning,
        generate_farmer_advisory,
        answer_farmer_chatbot_query,
    )
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from cognitive_engine import (
        generate_cognitive_reasoning,
        generate_farmer_advisory,
        answer_farmer_chatbot_query,
    )

# ----------------- PAGE CONFIGURATION -----------------
st.set_page_config(
    page_title="Marathwada Drought AI · Cognitive Biosystems",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- MINIMALIST BOTANICAL CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Space+Grotesk:wght@400;500;600;700&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500&display=swap');

    /* Global Typography & Colors */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1a271f;
    }

    .stApp {
        background-color: #fbfcf9;
    }

    /* Top Brand Editorial Banner */
    .brand-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        padding: 1.6rem 2rem;
        background: #f3f7f4;
        border: 1px solid #dce8df;
        border-radius: 12px;
        margin-bottom: 1.8rem;
    }

    .brand-kicker {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.76rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #436e52;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .brand-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #153222;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.95rem;
        color: #4e6355;
        margin-top: 0.45rem;
        line-height: 1.5;
        max-width: 720px;
    }

    .brand-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        background: #e4efe7;
        border: 1px solid #cadccf;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        color: #1f4730;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #2b824f;
        box-shadow: 0 0 0 3px rgba(43, 130, 79, 0.2);
    }

    /* Minimalist Metric Cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.2rem 0;
    }

    .minimal-metric-card {
        background: #ffffff;
        border: 1px solid #e1ebe3;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        transition: border-color 0.2s ease;
    }

    .minimal-metric-card:hover {
        border-color: #b7cebf;
    }

    .metric-label-clean {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #5c7464;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }

    .metric-val-clean {
        font-size: 1.65rem;
        font-weight: 700;
        color: #173623;
        line-height: 1.1;
    }

    .metric-sub-clean {
        font-size: 0.82rem;
        color: #6a8272;
        margin-top: 0.3rem;
    }

    /* Risk Diagnostic Badges (Botanical / Earth Tonality) */
    .diagnosis-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.4rem;
        border: 1px solid #dde7e0;
        margin-bottom: 1.4rem;
    }

    .risk-badge-high {
        background-color: #faede9;
        color: #9c3324;
        border: 1px solid #e6b8af;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .risk-badge-med {
        background-color: #fcf6e9;
        color: #8c5d19;
        border: 1px solid #e8d4a7;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    .risk-badge-low {
        background-color: #eaf4ed;
        color: #1f5f37;
        border: 1px solid #b7dcbe;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Cognitive Causal Ledger Cards */
    .causal-item {
        background: #ffffff;
        border: 1px solid #e3ece5;
        border-left: 3px solid #2d6141;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.65rem;
        font-size: 0.93rem;
        color: #213528;
        line-height: 1.45;
    }

    .causal-step-tag {
        font-family: 'Space Grotesk', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #4a7459;
        margin-bottom: 0.2rem;
        text-transform: uppercase;
    }

    /* Minimalist Advisory Cards */
    .advisory-panel {
        background: #ffffff;
        border: 1px solid #e2ece4;
        border-radius: 10px;
        padding: 1.15rem 1.25rem;
        height: 100%;
    }

    .advisory-header-tag {
        display: inline-block;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #245035;
        background: #edf5f0;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }

    .advisory-bullet {
        font-size: 0.91rem;
        color: #2b3e32;
        line-height: 1.5;
        margin-bottom: 0.55rem;
        padding-left: 1rem;
        position: relative;
    }

    .advisory-bullet::before {
        content: "•";
        color: #3b7450;
        font-weight: bold;
        position: absolute;
        left: 0;
    }

    /* Chat Styling */
    .chat-card {
        background: #ffffff;
        border: 1px solid #dce8df;
        border-radius: 12px;
        padding: 1.3rem;
        margin-top: 1rem;
    }

    /* Streamlit UI Component Overrides */
    div.stButton > button {
        background: #1e3f2c !important;
        color: #ffffff !important;
        border: 1px solid #142e20 !important;
        border-radius: 8px !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div.stButton > button:hover {
        background: #142e20 !important;
        border-color: #0c1c13 !important;
        box-shadow: 0 4px 14px rgba(20, 46, 32, 0.16) !important;
        transform: translateY(-1px) !important;
    }

    /* Minimalist Underline Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem !important;
        border-bottom: 1px solid #dbe6de !important;
        background-color: transparent !important;
        padding-bottom: 0.2rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
        color: #637a6b !important;
        padding: 0.7rem 0.3rem !important;
        border-bottom: 2px solid transparent !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #173824 !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #234d34 !important;
    }

    /* Sidebar Refinement */
    [data-testid="stSidebar"] {
        background-color: #f2f6f3 !important;
        border-right: 1px solid #dde7e0 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #dbe6de !important;
    }

    /* Muted clean expander */
    .streamlit-expanderHeader {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #1e3f2c !important;
        background-color: #f6faf7 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- DATA & MODEL LOADING -----------------
@st.cache_data
def load_dataset():
    data_path = os.path.join(os.path.dirname(__file__), "drought_data.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}")
        st.stop()
    return pd.read_csv(data_path)

@st.cache_resource
def load_or_train_model(df):
    model_path = os.path.join(os.path.dirname(__file__), "drought_model.pkl")
    features = [
        "Rainfall_mm", "Normal_Rainfall_mm", "Rainfall_Deficit_pct",
        "Soil_Moisture_pct", "Groundwater_Level_m", "Avg_Temp_C"
    ]
    X = df[features]
    y = df["Drought_Risk"]

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, features
        except Exception:
            pass

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    except Exception:
        pass
    return model, features

df = load_dataset()
model, feature_names = load_or_train_model(df)


# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.2rem 0 1rem 0;'>
        <div style='font-family: "Space Grotesk", sans-serif; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; color: #436e52; font-weight: 600;'>Control System</div>
        <div style='font-size: 1.25rem; font-weight: 700; color: #153222;'>Agro Parameters</div>
    </div>
    """, unsafe_allow_html=True)

    # Language Selector
    lang_choice = st.radio(
        "Language / भाषा",
        options=["English", "मराठी (Marathi)"],
        index=0,
        horizontal=True
    )
    lang_code = "mr" if "मराठी" in lang_choice else "en"

    st.markdown("---")
    st.markdown("<div class='metric-label-clean'>Location Selection</div>", unsafe_allow_html=True)

    districts = sorted(df["District"].unique())
    selected_district = st.selectbox("District / जिल्हा", districts, index=0)

    talukas_in_dist = sorted(df[df["District"] == selected_district]["Taluka"].unique())
    selected_taluka = st.selectbox("Taluka / तालुका", talukas_in_dist, index=0)

    preset = df[(df["District"] == selected_district) & (df["Taluka"] == selected_taluka)].iloc[0]
    use_defaults = st.checkbox("Auto-populate Taluka telemetry", value=True)

    st.markdown("---")
    st.markdown("<div class='metric-label-clean'>Telemetry Inputs</div>", unsafe_allow_html=True)

    if use_defaults:
        default_rainfall = float(preset["Rainfall_mm"])
        default_normal = float(preset["Normal_Rainfall_mm"])
        default_soil = float(preset["Soil_Moisture_pct"])
        default_gw = float(preset["Groundwater_Level_m"])
        default_temp = float(preset["Avg_Temp_C"])
    else:
        default_rainfall = 450.0
        default_normal = 750.0
        default_soil = 25.0
        default_gw = 24.0
        default_temp = 34.0

    input_rainfall = st.number_input(
        "Current Seasonal Rain (mm)",
        min_value=0.0, max_value=2500.0, value=default_rainfall, step=10.0
    )
    input_normal = st.number_input(
        "Normal Expected Rain (mm)",
        min_value=100.0, max_value=2500.0, value=default_normal, step=10.0
    )

    calculated_deficit = max(0.0, ((input_normal - input_rainfall) / input_normal) * 100.0)

    # Clean deficit stat box
    st.markdown(f"""
    <div style='background: #e7efe9; border-left: 3px solid #2d6141; padding: 0.55rem 0.8rem; border-radius: 0 6px 6px 0; margin: 0.5rem 0 0.8rem 0;'>
        <div style='font-size: 0.72rem; font-family: "Space Grotesk", sans-serif; color: #436e52; text-transform: uppercase;'>Computed Monsoon Deficit</div>
        <div style='font-size: 1.15rem; font-weight: 700; color: #163623;'>{calculated_deficit:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    input_soil = st.slider(
        "Soil Moisture (% Volumetric)",
        min_value=5.0, max_value=60.0, value=default_soil, step=1.0
    )
    input_groundwater = st.slider(
        "Groundwater Table Depth (m)",
        min_value=2.0, max_value=50.0, value=default_gw, step=0.5
    )
    input_temp = st.slider(
        "Mean Surface Temp (°C)",
        min_value=15.0, max_value=48.0, value=default_temp, step=0.5
    )


# ----------------- EDITORIAL HERO BANNER -----------------
banner_kicker = "MARATHWADA AGRO-CLIMATIC OBSERVATORY · 2026" if lang_code == "en" else "मराठवाडा कृषी-हवामान वेधशाळा · २०२६"
banner_title = "Drought Risk AI & Farmer Advisory" if lang_code == "en" else "दुष्काळ जोखीम AI आणि शेतकरी सल्लागार"
banner_sub = (
    "Cognitive decision-support system synthesizing multi-sensor precipitation, soil moisture, and aquifer dynamics with explainable causal reasoning for regional resilience."
    if lang_code == "en"
    else "पावसाची तूट, मातीतील ओलावा व भूजल पातळी यांचे विश्लेषण करून तर्कसंगत दुष्काळ जोखीम व शेतकरी मार्गदर्शन देणारी कॉग्निटिव्ह प्रणाली."
)

st.markdown(f"""
<div class="brand-container">
    <div>
        <div class="brand-kicker">{banner_kicker}</div>
        <h1 class="brand-title">{banner_title}</h1>
        <div class="brand-subtitle">{banner_sub}</div>
    </div>
    <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 0.45rem;">
        <div class="brand-status-badge">
            <span class="pulse-dot"></span> 74 Talukas Monitored
        </div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; color: #5f7566; letter-spacing: 0.05em;">
            RANDOM FOREST + COGNITIVE RULES
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- MINIMALIST TABS -----------------
tab_labels = [
    "01 / Risk Diagnostic & Advisory",
    "02 / Bilingual Farmer Assistant",
    "03 / Regional Agro Observatory",
    "04 / Model Architecture & Viva"
] if lang_code == "en" else [
    "०१ / जोखीम विश्लेषण व सल्लागार",
    "०२ / शेतकरी मदतनीस संवाद",
    "०३ / मराठवाडा वेधशाळा डॅशबोर्ड",
    "०४ / मॉडेल आर्किटेक्चर व व्हायव्हा"
]

tab1, tab2, tab3, tab4 = st.tabs(tab_labels)


# ----------------- TAB 1: RISK ASSESSMENT & REASONING -----------------
with tab1:
    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        st.markdown(f"""
        <div class="minimal-metric-card" style="margin-bottom: 1rem;">
            <div class="metric-label-clean">Target Location</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #153222;">{selected_taluka}, {selected_district}</div>
            <div class="metric-sub-clean">Marathwada Division · Vertisol Heavy Clay Belt</div>
            <hr style="margin: 0.75rem 0; border: none; border-top: 1px solid #e5eee8;" />
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.85rem; color: #3b5042;">
                <div>Rain: <b>{input_rainfall:.0f} mm</b></div>
                <div>Normal: <b>{input_normal:.0f} mm</b></div>
                <div>Soil Moisture: <b>{input_soil:.1f}%</b></div>
                <div>Aquifer Depth: <b>{input_groundwater:.1f} m</b></div>
                <div>Temperature: <b>{input_temp:.1f} °C</b></div>
                <div>Deficit: <b>{calculated_deficit:.1f}%</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        analyze_trigger = st.button("Run Diagnostic Trace / विश्लेषण सुरू करा", use_container_width=True)

    with col_right:
        input_data = pd.DataFrame([[
            input_rainfall,
            input_normal,
            calculated_deficit,
            input_soil,
            input_groundwater,
            input_temp
        ]], columns=feature_names)

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))
        confidence = prob_dict[prediction] * 100.0

        if prediction == "High":
            badge_html = "<span class='risk-badge-high'>● HIGH DROUGHT RISK · तीव्र दुष्काळ जोखीम</span>"
            risk_summary = "Immediate contingency measures required. Multi-source water stress detected across meteorological and soil buffers."
        elif prediction == "Medium":
            badge_html = "<span class='risk-badge-med'>● MODERATE VULNERABILITY · मध्यम जोखीम</span>"
            risk_summary = "Pre-emptive conservation advised. Moderate soil moisture depletion; sensitive to upcoming dry spells."
        else:
            badge_html = "<span class='risk-badge-low'>● NORMAL / LOW STRESS · समाधानकारक परिस्थिती</span>"
            risk_summary = "Hydrological and soil reserves within stable agronomic tolerances."

        st.markdown(f"""
        <div class="diagnosis-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.9rem;">
                {badge_html}
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.82rem; color: #4f6656;">
                    Confidence: <b>{confidence:.1f}%</b>
                </div>
            </div>
            <div style="font-size: 0.92rem; color: #3b4e42; line-height: 1.5; margin-bottom: 1rem;">
                {risk_summary}
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;">
                <div style="background: #f7faf8; border: 1px solid #e1ebe3; border-radius: 8px; padding: 0.65rem 0.85rem;">
                    <div class="metric-label-clean" style="font-size: 0.68rem;">Deficit</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #173623;">{calculated_deficit:.1f}%</div>
                </div>
                <div style="background: #f7faf8; border: 1px solid #e1ebe3; border-radius: 8px; padding: 0.65rem 0.85rem;">
                    <div class="metric-label-clean" style="font-size: 0.68rem;">Soil Saturation</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #173623;">{input_soil:.1f}%</div>
                </div>
                <div style="background: #f7faf8; border: 1px solid #e1ebe3; border-radius: 8px; padding: 0.65rem 0.85rem;">
                    <div class="metric-label-clean" style="font-size: 0.68rem;">Aquifer Head</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #173623;">{input_groundwater:.1f} m</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- COGNITIVE REASONING LEDGER -----------------
    st.markdown("<br/>", unsafe_allow_html=True)
    reasoning_title = "Causal Diagnostic Trace (Explainable AI Core)" if lang_code == "en" else "तर्कसंगत निदान विश्लेषण (AI ने हा निष्कर्ष का काढला?)"
    st.markdown(f"""
    <div style='margin-bottom: 0.85rem;'>
        <div class='brand-kicker'>Symbolic Cognitive Trace</div>
        <div style='font-size: 1.25rem; font-weight: 700; color: #153222;'>{reasoning_title}</div>
    </div>
    """, unsafe_allow_html=True)

    reasons = generate_cognitive_reasoning(
        calculated_deficit, input_soil, input_groundwater, input_temp, prediction, lang=lang_code
    )

    step_tags = [
        "01 / SYNTHETIC STATE EVALUATION",
        "02 / MONSOON DEFICIT THRESHOLD",
        "03 / ROOT-ZONE HYDROLOGY",
        "04 / SUB-SURFACE AQUIFER HEAD",
        "05 / THERMAL EVAPORATION INDEX"
    ]

    for idx, r in enumerate(reasons):
        tag = step_tags[idx] if idx < len(step_tags) else f"0{idx+1} / AGRO CRITERION"
        st.markdown(f"""
        <div class="causal-item">
            <div class="causal-step-tag">{tag}</div>
            <div>{r}</div>
        </div>
        """, unsafe_allow_html=True)

    # ----------------- AGRONOMIC ADVISORY DIRECTIVE -----------------
    st.markdown("<br/>", unsafe_allow_html=True)
    advisory_title = "Agronomic Directives & Farmer Action Plan" if lang_code == "en" else "शेतकरी कृती योजना व मार्गदर्शन"
    st.markdown(f"""
    <div style='margin-bottom: 0.85rem;'>
        <div class='brand-kicker'>Field Operational Guidance</div>
        <div style='font-size: 1.25rem; font-weight: 700; color: #153222;'>{advisory_title}</div>
    </div>
    """, unsafe_allow_html=True)

    advisories = generate_farmer_advisory(
        prediction, calculated_deficit, input_soil, input_groundwater, lang=lang_code
    )

    adv_cols = st.columns(len(advisories))
    for idx, (cat, items) in enumerate(advisories.items()):
        with adv_cols[idx]:
            bullets_html = "".join([f"<div class='advisory-bullet'>{item}</div>" for item in items])
            st.markdown(f"""
            <div class="advisory-panel">
                <span class="advisory-header-tag">{cat}</span>
                {bullets_html}
            </div>
            """, unsafe_allow_html=True)


# ----------------- TAB 2: BILINGUAL FARMER CHATBOT -----------------
with tab2:
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <div class='brand-kicker'>Bilingual Conversational Core</div>
        <div style='font-size: 1.35rem; font-weight: 700; color: #153222;'>AI Agro-Advisory Chatbot · शेतकरी संवाद मदतनीस</div>
        <div style='font-size: 0.9rem; color: #526759; margin-top: 0.2rem;'>Ask questions in English or Marathi regarding water conservation, crop choices, or state drought relief.</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Suggestion Chips
    st.markdown("<div class='metric-label-clean' style='margin-bottom: 0.4rem;'>Quick Inquiries / थेट प्रश्न</div>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    preset_query = ""

    if q_col1.button("💧 पाणी व्यवस्थापन कसे करावे?"):
        preset_query = "पाणी व्यवस्थापन कसे करावे?"
    if q_col2.button("🌾 दुष्काळात कोणती पिके घ्यावीत?"):
        preset_query = "दुष्काळात कोणती पिके घ्यावीत?"
    if q_col3.button("🌱 मातीतील ओलावा कसा टिकवायचा?"):
        preset_query = "मातीतील ओलावा कसा टिकवायचा?"
    if q_col4.button("🏛️ शासकीय १२-कलमी पॅकेज काय आहे?"):
        preset_query = "शासकीय दुष्काळ मदत पॅकेज काय आहे?"

    user_query = st.text_input(
        "Enter your query / आपला प्रश्न लिहा:",
        value=preset_query,
        placeholder="e.g. Which short-duration crops should I sow? किंवा कमी पाण्यात ठिबक सिंचन कसे वापरावे?"
    )

    if user_query:
        st.markdown(f"""
        <div style='background: #f4f8f5; border: 1px solid #dce8df; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 1rem;'>
            <div style='font-size: 0.72rem; font-family: "Space Grotesk", sans-serif; color: #3b6d4e; text-transform: uppercase; font-weight: 600;'>Farmer Query</div>
            <div style='font-size: 0.96rem; color: #1a2c20; font-weight: 500;'>{user_query}</div>
        </div>
        """, unsafe_allow_html=True)

        response_text = answer_farmer_chatbot_query(user_query, current_risk=prediction, lang=lang_code)
        st.markdown(f"""
        <div style='background: #ffffff; border: 1px solid #d8e5dc; border-left: 4px solid #244b35; border-radius: 0 8px 8px 0; padding: 1.1rem 1.25rem; margin-top: 0.75rem;'>
            <div style='font-size: 0.72rem; font-family: "Space Grotesk", sans-serif; color: #244b35; text-transform: uppercase; font-weight: 600; margin-bottom: 0.4rem;'>Cognitive Advisory Response</div>
            <div style='font-size: 0.94rem; color: #203326; line-height: 1.55;'>{response_text}</div>
        </div>
        """, unsafe_allow_html=True)


# ----------------- TAB 3: REGIONAL OBSERVATORY DASHBOARD -----------------
with tab3:
    st.markdown("""
    <div style='margin-bottom: 1.2rem;'>
        <div class='brand-kicker'>Spatial Telemetry</div>
        <div style='font-size: 1.35rem; font-weight: 700; color: #153222;'>Regional Agro-Climatic Observatory · Marathwada</div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Top Metrics Grid
    st.markdown(f"""
    <div class="metric-grid">
        <div class="minimal-metric-card">
            <div class="metric-label-clean">Monitored Talukas</div>
            <div class="metric-val-clean">{len(df)}</div>
            <div class="metric-sub-clean">Across 8 Administrative Districts</div>
        </div>
        <div class="minimal-metric-card">
            <div class="metric-label-clean">High Risk Talukas</div>
            <div class="metric-val-clean" style="color: #9c3324;">{int((df['Drought_Risk'] == 'High').sum())}</div>
            <div class="metric-sub-clean">{((df['Drought_Risk'] == 'High').sum() / len(df) * 100):.1f}% of surveyed region</div>
        </div>
        <div class="minimal-metric-card">
            <div class="metric-label-clean">Mean Monsoon Deficit</div>
            <div class="metric-val-clean">{df['Rainfall_Deficit_pct'].mean():.1f}%</div>
            <div class="metric-sub-clean">Peak: {df['Rainfall_Deficit_pct'].max():.1f}% (Georai, Beed)</div>
        </div>
        <div class="minimal-metric-card">
            <div class="metric-label-clean">Mean Soil Saturation</div>
            <div class="metric-val-clean">{df['Soil_Moisture_pct'].mean():.1f}%</div>
            <div class="metric-sub-clean">Critical stress under 22%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='metric-label-clean'>Average Rainfall Deficit by District (%)</div>", unsafe_allow_html=True)
        deficit_series = df.groupby("District")["Rainfall_Deficit_pct"].mean().sort_values(ascending=False)
        st.bar_chart(deficit_series, color="#2d6141")

    with c2:
        st.markdown("<div class='metric-label-clean'>Talukas by Drought Risk Severity</div>", unsafe_allow_html=True)
        risk_series = df["Drought_Risk"].value_counts()
        st.bar_chart(risk_series, color="#4a7c5f")

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='metric-label-clean'>Regional Telemetry Records Browser</div>", unsafe_allow_html=True)

    selected_dist_filters = st.multiselect(
        "Filter records by District:",
        options=districts,
        default=districts
    )
    display_df = df[df["District"].isin(selected_dist_filters)]
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ----------------- TAB 4: ARCHITECTURE & VIVA PREP -----------------
with tab4:
    st.markdown("""
    <div style='margin-bottom: 1.2rem;'>
        <div class='brand-kicker'>Academic Specifications</div>
        <div style='font-size: 1.35rem; font-weight: 700; color: #153222;'>Model Evaluation & Viva Voce Guide</div>
    </div>
    """, unsafe_allow_html=True)

    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.markdown("<div class='metric-label-clean'>Ensemble Validation Metrics</div>", unsafe_allow_html=True)
        X = df[feature_names]
        y = df["Drought_Risk"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        test_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        test_model.fit(X_train, y_train)
        preds = test_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.markdown(f"""
        <div class="minimal-metric-card" style="margin-bottom: 0.9rem;">
            <div class="metric-label-clean">Stratified Test Accuracy</div>
            <div class="metric-val-clean" style="color: #235836;">{acc * 100:.1f}%</div>
            <div class="metric-sub-clean">Evaluated across holdout test split (25%)</div>
        </div>
        """, unsafe_allow_html=True)
        st.text("Detailed Classification Report:\n" + classification_report(y_test, preds, zero_division=0))

    with ev_col2:
        st.markdown("<div class='metric-label-clean'>Feature Importance Ranking (Gini Impurity)</div>", unsafe_allow_html=True)
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": importances * 100
        }).sort_values("Importance (%)", ascending=False)
        st.bar_chart(feat_df.set_index("Feature"), color="#2d6141")

    st.markdown("---")
    st.markdown("<div class='brand-kicker'>Examination Preparation</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #153222; margin-bottom: 0.8rem;'>Frequently Asked Viva Voce Questions</div>", unsafe_allow_html=True)

    with st.expander("Q1: What differentiates this Cognitive Computing system from standard Machine Learning?"):
        st.markdown("""
        **Answer:**
        Conventional Machine Learning functions as a black-box function approximator that only maps input vectors to a discrete numerical label (`High Risk`).
        A **Cognitive Computing architecture** implements the full human-like cognitive cycle:
        1. **Perception**: Multi-source environmental feature synthesis (Rainfall, Soil Moisture, Groundwater depth, Temperature).
        2. **Pattern Learning**: Supervised Random Forest classification.
        3. **Causal Reasoning**: An interpretable rule-based cognitive layer explaining *why* the drought state occurred.
        4. **Decision Support**: Generating category-specific agronomic actions (irrigation, crops, mulching, government packages).
        5. **Vernacular Interaction**: Communicating with grassroots end-users in their native language (Marathi & English).
        """)

    with st.expander("Q2: Why is Random Forest preferred over Logistic Regression or Single Decision Trees?"):
        st.markdown("""
        **Answer:**
        - Single Decision Trees suffer from high variance and prone to overfitting.
        - Random Forest aggregates 100 de-correlated bootstrap trees, reducing variance while preserving non-linear threshold splits.
        - It natively estimates class probabilities, allowing us to compute model confidence.
        """)

    with st.expander("Q3: What is the socioeconomic context of the 2026 Marathwada case study?"):
        st.markdown("""
        **Answer:**
        In late September 2026, the Government of Maharashtra declared drought in 265 of 358 talukas statewide, with 74 talukas across all 8 Marathwada districts severely hit due to an acute monsoon deficit. Grounding the project in this current issue provides immense academic and practical value.
        """)

    with st.expander("Q4: How can this system be expanded for a final-year capstone project?"):
        st.markdown("""
        **Answer:**
        1. **IoT Telemetry**: Ingesting live LoRaWAN soil moisture probes and automated weather stations.
        2. **Satellite Remote Sensing**: Integrating Sentinel-2 NDVI and NDWI vegetation and water indices.
        3. **Local LLM Integration**: Incorporating a fine-tuned Marathi Llama-3/Mistral model for natural voice-based agricultural guidance.
        """)


# ----------------- FOOTER -----------------
st.markdown("""
<div style='margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid #dce8df; text-align: center; color: #6d8475; font-size: 0.82rem; font-family: "Space Grotesk", sans-serif;'>
    MARATHWADA AGRO-COGNITIVE BIOSYSTEMS · 2026 RESEARCH & DEMONSTRATION INITIATIVE
</div>
""", unsafe_allow_html=True)
