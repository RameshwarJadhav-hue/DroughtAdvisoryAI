"""
Cognitive AI-Based Drought Risk Prediction and Farmer Advisory System
Case Study: Marathwada Agro-Climatic Region
Framework: Streamlit + Scikit-Learn + Python
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
    page_title="Marathwada Drought AI & Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1rem;
    }
    .high-risk-badge {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .med-risk-badge {
        background-color: #fef3c7;
        color: #b45309;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .low-risk-badge {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.4rem 0.8rem;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .advisory-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
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

    # Try loading existing pickle model
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model, features
        except Exception:
            pass

    # Train Random Forest if pickle doesn't exist
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
st.sidebar.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("⚙️ System Control Panel")

# Language Selection
lang_choice = st.sidebar.radio(
    "🌐 भाषा निवडा / Select Language:",
    options=["English", "मराठी (Marathi)"],
    index=0
)
lang_code = "mr" if "मराठी" in lang_choice else "en"

st.sidebar.markdown("---")
st.sidebar.subheader("📍 स्थान निवडा / Select Location")

districts = sorted(df["District"].unique())
selected_district = st.sidebar.selectbox("जिल्हा / District", districts, index=0)

talukas_in_dist = sorted(df[df["District"] == selected_district]["Taluka"].unique())
selected_taluka = st.sidebar.selectbox("तालुका / Taluka", talukas_in_dist, index=0)

# Retrieve preset data for the selected taluka
preset = df[(df["District"] == selected_district) & (df["Taluka"] == selected_taluka)].iloc[0]

use_defaults = st.sidebar.checkbox("तालुक्यातील अधिकृत डेटा वापरा / Load Taluka Historical Defaults", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ पर्यावरणीय घटक / Environmental Inputs")

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

input_rainfall = st.sidebar.number_input(
    "हंगामातील पाऊस / Season Rainfall (mm)",
    min_value=0.0, max_value=2500.0, value=default_rainfall, step=10.0
)
input_normal = st.sidebar.number_input(
    "सरासरी पाऊस / Normal Rainfall (mm)",
    min_value=100.0, max_value=2500.0, value=default_normal, step=10.0
)

# Real-time deficit calculation
calculated_deficit = max(0.0, ((input_normal - input_rainfall) / input_normal) * 100.0)
st.sidebar.info(f"📊 पावसातील तूट / Deficit: **{calculated_deficit:.1f}%**")

input_soil = st.sidebar.slider(
    "मातीतील ओलावा / Soil Moisture (%)",
    min_value=5.0, max_value=60.0, value=default_soil, step=1.0
)
input_groundwater = st.sidebar.slider(
    "भूजल खोली / Groundwater Depth (meters below ground)",
    min_value=2.0, max_value=50.0, value=default_gw, step=0.5
)
input_temp = st.sidebar.slider(
    "सरासरी तापमान / Average Temperature (°C)",
    min_value=15.0, max_value=48.0, value=default_temp, step=0.5
)


# ----------------- MAIN HEADER -----------------
title_text = "🌾 Cognitive AI-Based Drought Risk & Farmer Advisory System" if lang_code == "en" else "🌾 मराठवाडा दुष्काळ जोखीम अंदाज व शेतकरी कॉग्निटिव्ह सल्लागार"
st.markdown(f"<div class='main-title'>{title_text}</div>", unsafe_allow_html=True)

sub_text = (
    "A Cognitive Computing + Machine Learning Mini-Project | Case Study: 2026 Marathwada Drought Management"
    if lang_code == "en"
    else "कॉग्निटिव्ह संगणन + मशिन लर्निंग मिनी-प्रकल्प | अभ्यास: मराठवाडा दुष्काळ व्यवस्थापन २०२६"
)
st.markdown(f"<div class='sub-title'>{sub_text}</div>", unsafe_allow_html=True)


# ----------------- TABS NAVIGATION -----------------
tab_labels = [
    "🔮 Risk Assessment & Reasoning",
    "💬 Bilingual Farmer Chatbot",
    "📊 Regional Agro Dashboard",
    "🧪 Model Evaluation & Viva Prep"
] if lang_code == "en" else [
    "🔮 जोखीम अंदाज व तर्कसंगती",
    "💬 शेतकरी मदतनीस संवाद",
    "📊 प्रादेशिक डॅशबोर्ड",
    "🧪 मॉडेल मूल्यमापन व व्हायव्हा"
]

tab1, tab2, tab3, tab4 = st.tabs(tab_labels)


# ----------------- TAB 1: RISK ASSESSMENT & REASONING -----------------
with tab1:
    col_pred_left, col_pred_right = st.columns([1.1, 1.9])

    with col_pred_left:
        st.subheader("📍 Target Location Details" if lang_code == "en" else "📍 निवडलेल्या ठिकाणाचा तपशील")
        st.markdown(f"""
        - **District / जिल्हा:** `{selected_district}`
        - **Taluka / तालुका:** `{selected_taluka}`
        - **Actual Rainfall:** `{input_rainfall:.1f} mm`
        - **Normal Rainfall:** `{input_normal:.1f} mm`
        - **Rainfall Deficit:** `{calculated_deficit:.1f}%`
        - **Soil Moisture:** `{input_soil:.1f}%`
        - **Groundwater Depth:** `{input_groundwater:.1f} m`
        - **Average Temperature:** `{input_temp:.1f} °C`
        """)

        analyze_btn = st.button("🚀 Analyze Drought Risk / जोखीम विश्लेषण करा", type="primary", use_container_width=True)

    with col_pred_right:
        # Prepare input sample
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

        st.subheader("🎯 Prediction Output" if lang_code == "en" else "🎯 अंदाज व निष्कर्ष")

        badge_class = "high-risk-badge" if prediction == "High" else ("med-risk-badge" if prediction == "Medium" else "low-risk-badge")
        risk_label_mr = "तीव्र दुष्काळ जोखीम (HIGH RISK)" if prediction == "High" else ("मध्यम जोखीम (MEDIUM RISK)" if prediction == "Medium" else "कमी जोखीम / सामान्य (LOW RISK)")
        risk_label_en = f"{prediction.upper()} DROUGHT RISK"

        display_label = risk_label_mr if lang_code == "mr" else risk_label_en

        st.markdown(f"""
        <div style='margin-bottom: 15px;'>
            <span class='{badge_class}' style='font-size: 1.3rem;'>
                {display_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Category", prediction)
        m2.metric("Rainfall Deficit", f"{calculated_deficit:.1f}%")
        m3.metric("Model Confidence", f"{confidence:.1f}%")

        st.write("**Prediction Probabilities across classes:**")
        prob_df = pd.DataFrame([prob_dict]).rename(index={0: "Probability"})
        st.dataframe(prob_df.style.format("{:.1%}"), use_container_width=True)

    st.markdown("---")

    # Cognitive Reasoning Section
    st.subheader("🧠 Cognitive Reasoning Layer (तर्कसंगत स्पष्टीकरण)" if lang_code == "en" else "🧠 कॉग्निटिव्ह तर्कसंगत विश्लेषण (AI ने हा निर्णय का घेतला?)")
    st.caption("Unlike a black-box model, the cognitive layer generates causal explanations by inspecting agro-climatic boundaries.")

    reasons = generate_cognitive_reasoning(
        calculated_deficit, input_soil, input_groundwater, input_temp, prediction, lang=lang_code
    )

    for r in reasons:
        st.markdown(f"🔹 {r}")

    st.markdown("---")

    # Agronomic Farmer Advisory Section
    st.subheader("🌱 Farmer Advisory & Action Plan (शेतकरी कृती योजना)" if lang_code == "en" else "🌱 शेतकरी कृषी सल्ला व मार्गदर्शन योजना")
    advisories = generate_farmer_advisory(
        prediction, calculated_deficit, input_soil, input_groundwater, lang=lang_code
    )

    adv_cols = st.columns(2)
    categories = list(advisories.keys())

    for idx, cat in enumerate(categories):
        target_col = adv_cols[idx % 2]
        with target_col:
            st.markdown(f"#### 📌 {cat}")
            for item in advisories[cat]:
                st.markdown(f"- {item}")
            st.write("")


# ----------------- TAB 2: BILINGUAL FARMER CHATBOT -----------------
with tab2:
    st.subheader("💬 AI Farmer Advisory Assistant (शेतकरी मदतनीस चॅटबॉट)")
    st.markdown(
        "Ask questions in **English** or **मराठी (Marathi)** regarding water conservation, crop choices, soil management, or government relief packages."
        if lang_code == "en"
        else "पाणी बचत, पिकांची निवड, सेंद्रिय आच्छादन किंवा शासकीय मदत याविषयी मराठी अथवा इंग्रजीत थेट प्रश्न विचारा."
    )

    st.write("**💡 Quick Suggestion Questions / जलद प्रश्न:**")
    quick_col1, quick_col2, quick_col3 = st.columns(3)
    preset_query = ""

    if quick_col1.button("💧 पाणी व्यवस्थापन कसे करावे?"):
        preset_query = "पाणी व्यवस्थापन कसे करावे?"
    if quick_col2.button("🌾 दुष्काळात कोणती पिके घ्यावीत?"):
        preset_query = "दुष्काळात कोणती पिके घ्यावीत?"
    if quick_col3.button("🏛️ शासकीय दुष्काळ मदत पॅकेज काय आहे?"):
        preset_query = "शासकीय दुष्काळ मदत पॅकेज काय आहे?"

    user_query = st.text_input(
        "Type your question here / आपला प्रश्न येथे लिहा:",
        value=preset_query,
        placeholder="उदा. How to save water with drip? किंवा दुष्काळात जनावरांचा चारा कसा नियोजित करावा?"
    )

    if user_query:
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            response_text = answer_farmer_chatbot_query(user_query, current_risk=prediction, lang=lang_code)
            st.markdown(response_text)


# ----------------- TAB 3: REGIONAL AGRO DASHBOARD -----------------
with tab3:
    st.subheader("📊 Marathwada Regional Agro-Climatic Dashboard (मराठवाडा डॅशबोर्ड)")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Monitored Talukas", len(df))
    d2.metric("Districts Represented", df["District"].nunique())
    d3.metric("High-Risk Talukas", int((df["Drought_Risk"] == "High").sum()))
    d4.metric("Avg Regional Deficit", f"{df['Rainfall_Deficit_pct'].mean():.1f}%")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("#### 🌧️ Average Rainfall Deficit by District (%)")
        district_deficit = df.groupby("District")["Rainfall_Deficit_pct"].mean().sort_values(ascending=False)
        st.bar_chart(district_deficit)

    with chart_col2:
        st.write("#### ⚠️ Drought Risk Category Distribution")
        risk_counts = df["Drought_Risk"].value_counts()
        st.bar_chart(risk_counts)

    st.markdown("---")
    st.write("#### 📋 Marathwada Taluka Dataset Browser")

    filter_dist = st.multiselect(
        "Filter by District / जिल्हानुसार फिल्टर करा:",
        options=districts,
        default=districts
    )
    filtered_df = df[df["District"].isin(filter_dist)]
    st.dataframe(filtered_df, use_container_width=True)


# ----------------- TAB 4: MODEL EVALUATION & VIVA PREPARATION -----------------
with tab4:
    st.subheader("🧪 Machine Learning Performance & Cognitive Architecture")

    eval_col1, eval_col2 = st.columns(2)

    with eval_col1:
        st.markdown("#### 📈 Model Validation Metrics")
        X = df[feature_names]
        y = df["Drought_Risk"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        test_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        test_model.fit(X_train, y_train)
        preds = test_model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.metric("Test Set Accuracy", f"{acc * 100:.2f}%")
        st.text("Detailed Classification Report:\n" + classification_report(y_test, preds, zero_division=0))

    with eval_col2:
        st.markdown("#### 🔍 Feature Importance Ranking")
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": importances * 100
        }).sort_values("Importance (%)", ascending=True)
        st.bar_chart(data=feat_df.set_index("Feature"))

    st.markdown("---")
    st.subheader("🎓 College Mini-Project Viva & Presentation Guide")

    with st.expander("❓ Q1: What makes this a 'Cognitive Computing' project rather than just standard Machine Learning?"):
        st.markdown("""
        **Answer:**
        Standard Machine Learning only outputs a statistical class label or number (e.g. `High Risk`), acting as a black box.
        A **Cognitive Computing system** integrates 4 core human-like cognitive abilities:
        1. **Perception:** Ingests multiple heterogeneous environmental indicators (Rainfall, Soil Moisture, Groundwater depth, Temperature).
        2. **Learning:** Extracts patterns from historical drought data using Random Forest classification.
        3. **Reasoning:** Contains an explicit causal rule-based engine that explains *why* the drought risk is severe (e.g., deficit > 35% compounded by critical soil depletion).
        4. **Decision Support & Interaction:** Generates actionable agronomic advisories and interacts with farmers in natural language (English & Marathi).
        """)

    with st.expander("❓ Q2: Why choose Random Forest over other algorithms?"):
        st.markdown("""
        **Answer:**
        - Random Forest is an ensemble of decorrelated decision trees that effectively mitigates overfitting on tabular data.
        - It natively handles non-linear relationships and interactions between multi-sensor features (e.g. high heat accelerating low soil moisture).
        - It outputs well-calibrated class probabilities, allowing the system to measure confidence alongside predictions.
        """)

    with st.expander("❓ Q3: What is the relevance of the 2026 Marathwada context in this project?"):
        st.markdown("""
        **Answer:**
        In late September 2026, Maharashtra declared drought in 265 out of 358 talukas statewide, with 74 talukas across all 8 Marathwada districts severely hit due to severe monsoon deficit (nearly 38% shortfall in parts of Marathwada). This real-world challenge makes the project highly topical, impactful, and socially relevant.
        """)

    with st.expander("❓ Q4: How can this system be scaled up for a final-year project?"):
        st.markdown("""
        **Answer:**
        1. **IoT Sensor Integration:** Connect automated soil moisture probes and LoRaWAN weather stations for live telemetry.
        2. **Satellite Remote Sensing:** Integrate Sentinel-2 / Landsat NDVI (Normalized Difference Vegetation Index) and NDWI (Normalized Difference Water Index).
        3. **LLM Integration:** Integrate an open-source fine-tuned Marathi LLM (e.g., Llama-3 or Mistral) for expanded natural voice interactions.
        4. **SMS/WhatsApp Gateway:** Automatically broadcast localized advisories to registered farmers' feature phones.
        """)


# ----------------- FOOTER -----------------
st.markdown("---")
st.caption("🌾 Marathwada Cognitive AI Project | Developed for Academic Demonstrations & Farmer Advisory Research | 2026")
