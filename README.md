# 🌾 Cognitive AI-Based Drought Risk Prediction & Farmer Advisory System
**Case Study: Marathwada Agro-Climatic Region**

A beginner-friendly, end-to-end **Cognitive Computing & Machine Learning Mini-Project** designed for engineering and computer science students.

---

## 📌 Project Overview

This project demonstrates how **Cognitive Computing** principles extend conventional Machine Learning. Instead of functioning solely as a "black-box" risk predictor, the system combines:
1. **Perception**: Multi-source environmental feature ingestion (Rainfall, Soil Moisture, Groundwater Depth, Temperature).
2. **Learning**: Random Forest classification predicting drought severity categories (**Low / Medium / High**).
3. **Cognitive Reasoning**: An explainable causal reasoning engine explaining *why* the drought risk occurs.
4. **Farmer Advisory Engine**: Actionable, localized agronomic recommendations (crop selection, micro-irrigation, mulching, government relief packages).
5. **Bilingual Human-AI Interface**: Interactive Streamlit dashboard and conversational advisory chatbot supporting both **English and Marathi (मराठी)**.

---

## 📁 Project Structure

```text
marathwada_drought_cognitive_ai/
│
├── app.py                      # Interactive Streamlit Web Application
├── cognitive_engine.py         # Cognitive reasoning & bilingual advisory rules
├── train_model.py              # ML training, evaluation metrics & model export
├── drought_data.csv            # Representative dataset for 8 Marathwada districts
├── drought_model.pkl           # Saved trained Random Forest model (generated)
├── requirements.txt            # Python dependencies
├── README.md                   # Quickstart guide
└── PROJECT_DOCUMENTATION.md    # Complete academic project report & viva guide
```

---

## 🚀 Quickstart Guide (How to Run on Windows)

### 1. Prerequisites
Ensure you have **Python 3.9+** installed. If not yet installed, you can install it via Windows PowerShell using `winget`:
```powershell
winget install Python.Python.3.11
```
*(Or download and install Python from https://www.python.org/downloads/ ensuring you check "Add Python to PATH")*

### 2. Navigate to Project Directory
Open PowerShell or Command Prompt:
```powershell
cd C:\Users\rameshwar\.gemini\antigravity\scratch\marathwada_drought_cognitive_ai
```

### 3. (Optional but Recommended) Create a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Train the Machine Learning Model
```powershell
python train_model.py
```
*This will evaluate baseline Decision Tree vs. Random Forest (~95-100% accuracy on prototype dataset), print confusion matrices, feature importances, and save `drought_model.pkl`.*

### 6. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
*Your browser will automatically open at `http://localhost:8501`.*

---

## 💡 Key Features of the Web App

- **Tab 1: 🔮 Risk Assessment & Reasoning:**
  Select any of the 8 Marathwada districts (Chhatrapati Sambhajinagar, Jalna, Beed, Latur, Dharashiv, Nanded, Parbhani, Hingoli) and specific talukas. Adjust rainfall, soil moisture, groundwater, and temperature. Get real-time risk predictions with **causal explanations** and **actionable farmer advisories**.
- **Tab 2: 💬 Bilingual Farmer Chatbot:**
  Ask questions in English or Marathi regarding water conservation, drought-tolerant crop varieties (बाजरी, ज्वारी, तूर), and government subsidy packages.
- **Tab 3: 📊 Regional Agro Dashboard:**
  Inspect average rainfall deficits, risk distributions across districts, and explore historical taluka records.
- **Tab 4: 🧪 Model Evaluation & Viva Prep:**
  View test-set accuracy, classification metrics, feature importance rankings, and top college viva questions & answers.

---

## 📚 Real-World Background (Context)
In late September 2026, the Maharashtra government officially notified drought in 265 of 358 talukas statewide, with 74 talukas across all 8 Marathwada districts severely affected due to significant monsoon deficits. This project provides an intelligent decision-support system to aid agricultural stakeholders and policymakers during such environmental stresses.
