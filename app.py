"""
LPU Student Academic Early-Warning & Intervention System
Course: INT395 Supervised Learning | Lovely Professional University
Production-Ready Web Application: app.py

Enterprise-grade Academic Dashboard:
- Clean, minimalist flat design (Linear/Stripe style)
- Solid colors only: #FFFFFF surfaces, #F8FAFC background, #0F172A text, #1E3A8A navy accent
- Semantic risk badges: #DC2626 (High), #D97706 (Medium), #16A34A (Low)
- Multi-page navigation:
  1. Executive Overview
  2. Individual Student Risk & SHAP Profiler
  3. Batch Prediction & CSV Upload
  4. Model Performance & EDA Lab
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from src.custom_transformers import AcademicFeatureEngineer, IQROutlierCapper
from src.explain import explain_student_risk, generate_actionable_interventions, safe_val

# -----------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & ENTERPRISE FLAT CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LPU Academic Early-Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise-grade CSS (strictly flat, solid colors, NO gradients, NO glassmorphism)
CUSTOM_CSS = """
<style>
/* Global resets & typography */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
    background-color: #F8FAFC;
}

.stApp {
    background-color: #F8FAFC;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #0F172A;
}

/* Modern Flat Sidebar Navigation Menu */
[data-testid="stSidebar"] div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 10px 14px !important;
    border-radius: 6px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    background-color: transparent !important;
    margin-bottom: 2px !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}

/* Hide default circular radio button */
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Radio label typography */
[data-testid="stSidebar"] div[role="radiogroup"] > label div p {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #334155 !important;
    margin: 0 !important;
    letter-spacing: 0.01em;
}

/* Hover state */
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background-color: #F8FAFC !important;
    border-color: #E2E8F0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:hover div p {
    color: #0F172A !important;
}

/* Active / Selected state */
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background-color: #EFF6FF !important;
    border: 1px solid #BFDBFE !important;
    border-left: 4px solid #1E3A8A !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) div p {
    color: #1E3A8A !important;
    font-weight: 600 !important;
}

/* Flat solid cards */
.flat-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.flat-card-header {
    font-size: 0.875rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #475569;
    margin-bottom: 0.5rem;
}

.flat-metric-value {
    font-size: 1.875rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

.flat-metric-subtext {
    font-size: 0.8125rem;
    color: #64748B;
    margin-top: 0.25rem;
}

/* Flat Semantic Badges */
.badge-high {
    display: inline-block;
    background-color: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FCA5A5;
    padding: 0.25rem 0.625rem;
    border-radius: 4px;
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}

.badge-medium {
    display: inline-block;
    background-color: #FFFBEB;
    color: #D97706;
    border: 1px solid #FCD34D;
    padding: 0.25rem 0.625rem;
    border-radius: 4px;
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}

.badge-low {
    display: inline-block;
    background-color: #F0FDF4;
    color: #16A34A;
    border: 1px solid #86EFAC;
    padding: 0.25rem 0.625rem;
    border-radius: 4px;
    font-size: 0.8125rem;
    font-weight: 600;
    letter-spacing: 0.025em;
}

/* Intervention Card */
.intervention-item {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #1E3A8A;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.intervention-item-critical {
    background-color: #FFFFFF;
    border: 1px solid #FCA5A5;
    border-left: 4px solid #DC2626;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.intervention-item-medium {
    background-color: #FFFFFF;
    border: 1px solid #FCD34D;
    border-left: 4px solid #D97706;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.intervention-item-low {
    background-color: #FFFFFF;
    border: 1px solid #86EFAC;
    border-left: 4px solid #16A34A;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #F1F5F9;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 4px;
    font-weight: 500;
    color: #475569;
    background-color: transparent;
    border: none;
    padding: 0 16px;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #1E3A8A !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

/* Flat Button overrides */
.stButton>button {
    background-color: #1E3A8A;
    color: #FFFFFF;
    border: 1px solid #1E3A8A;
    border-radius: 4px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    box-shadow: none !important;
}

.stButton>button:hover {
    background-color: #172554;
    border-color: #172554;
    color: #FFFFFF;
}

/* Table styling */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    background-color: #FFFFFF;
}

/* Metric labels */
[data-testid="stMetricLabel"] {
    font-size: 0.8125rem;
    color: #475569;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CACHED ASSET LOADING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    pipe_path = os.path.join("artifacts", "best_model_pipeline.joblib")
    if not os.path.exists(pipe_path):
        st.error(f"Model pipeline not found at {pipe_path}. Execute src/train.py first.")
        st.stop()
    return joblib.load(pipe_path)

@st.cache_resource
def load_shap_explainer():
    shap_path = os.path.join("artifacts", "shap_explainer.joblib")
    if not os.path.exists(shap_path):
        st.error(f"SHAP explainer not found at {shap_path}. Execute src/train.py first.")
        st.stop()
    return joblib.load(shap_path)

# Multi-Cohort Academic Surveillance Registry
COHORT_REGISTRY = {
    "🏛️ Institutional Baseline (2,500 Students)": {
        "file": os.path.join("data", "student_academic_data.csv"),
        "label": "Institutional Baseline (All 7 Departments)",
        "badge": "2,500 Undergraduates"
    },
    "💻 School of CSE (949 Students)": {
        "file": os.path.join("data", "cohort_cse_engineering.csv"),
        "label": "School of Computer Science & Engineering",
        "badge": "949 Engineers"
    },
    "⚡ Core Engineering: ECE & ME (607 Students)": {
        "file": os.path.join("data", "cohort_ece_me_core.csv"),
        "label": "Electronics & Mechanical Engineering",
        "badge": "607 Engineers"
    },
    "📈 Management & Commerce (211 Students)": {
        "file": os.path.join("data", "cohort_management_commerce.csv"),
        "label": "School of Management & Business Studies",
        "badge": "211 Students"
    },
    "🌱 First-Year Freshmen (839 Students)": {
        "file": os.path.join("data", "cohort_first_year_freshmen.csv"),
        "label": "First-Year Freshmen Foundation (Sem 1 & 2)",
        "badge": "839 Freshmen"
    },
    "⚠️ High-Attrition Remedial Triage (200 Students)": {
        "file": os.path.join("data", "cohort_high_risk_remedial.csv"),
        "label": "Remedial Coaching & Academic Probation Triage",
        "badge": "200 At-Risk Focus"
    },
    "📋 Section Alpha Audit Sample (50 Students)": {
        "file": os.path.join("data", "sample_batch_upload.csv"),
        "label": "Section Alpha Audit Sample",
        "badge": "50 Students"
    }
}

@st.cache_data
def load_cohort_dataset(file_path: str):
    if not os.path.exists(file_path):
        file_path = os.path.join("data", os.path.basename(file_path))
    df = pd.read_csv(file_path)
    if "Risk_Level" not in df.columns:
        pipe = load_ml_pipeline()
        req_cols = [
            "Department", "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks",
            "CA3_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
            "Active_Backlogs", "Study_Hours_Per_Week", "Participation_Score"
        ]
        X_df = df[req_cols].copy()
        if "LMS_Logins_Per_Week" not in X_df.columns:
            X_df["LMS_Logins_Per_Week"] = 14
        pipeline_features = [
            "Department", "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks",
            "CA3_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
            "LMS_Logins_Per_Week", "Active_Backlogs", "Study_Hours_Per_Week", "Participation_Score"
        ]
        X_df = X_df[pipeline_features]
        preds = pipe.predict(X_df)
        inv_map = {0: "Low", 1: "Medium", 2: "High"}
        df["Risk_Level"] = [inv_map[p] for p in preds]
    return df

@st.cache_data
def load_student_dataset():
    data_path = os.path.join("data", "student_academic_data.csv")
    if not os.path.exists(data_path):
        st.error(f"Dataset not found at {data_path}. Execute src/data_prep.py first.")
        st.stop()
    return load_cohort_dataset(data_path)

@st.cache_data
def load_json_artifact(filename: str):
    path = os.path.join("artifacts", filename)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# Load assets
try:
    pipeline = load_ml_pipeline()
    explainer = load_shap_explainer()
    raw_default_df = load_student_dataset()
    if "selected_cohort_key" not in st.session_state:
        st.session_state.selected_cohort_key = "🏛️ Institutional Baseline (2,500 Students)"
    if "active_students" not in st.session_state:
        st.session_state.active_students = raw_default_df.copy()
    if "is_custom_cohort" not in st.session_state:
        st.session_state.is_custom_cohort = False
    
    df_students = st.session_state.active_students
    feature_names = load_json_artifact("feature_names.json")
    eda_stats = load_json_artifact("eda_stats.json")
    model_comparison = load_json_artifact("model_comparison.json")
    confusion_matrix_data = load_json_artifact("confusion_matrix.json")
    roc_data = load_json_artifact("roc_curves.json")
    error_analysis = load_json_artifact("error_analysis.json")
    global_shap_data = load_json_artifact("global_shap.json")
    smote_summary = load_json_artifact("smote_summary.json")
except Exception as e:
    st.error(f"Error loading system artifacts: {e}")
    st.stop()

# Helper for Plotly layout consistency (Flat enterprise style)
def apply_flat_plotly_layout(fig, title: str = "", height: int = 380):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=14, color="#0F172A", weight=600)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#E2E8F0",
            tickfont=dict(color="#475569")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            linecolor="#E2E8F0",
            tickfont=dict(color="#475569")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569")
        )
    )
    return fig

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION & SYSTEM BADGE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 1.1rem; border-bottom: 1px solid #E2E8F0; margin-bottom: 1.25rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.4rem;">
            <span style="font-size: 1.25rem;">🎓</span>
            <span style="font-size: 0.6875rem; font-weight: 700; color: #1E3A8A; letter-spacing: 0.08em; text-transform: uppercase;">
                LOVELY PROFESSIONAL UNIVERSITY
            </span>
        </div>
        <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; line-height: 1.35;">
            Academic Early-Warning & Intervention System
        </div>
    </div>
    <div style="font-size: 0.6875rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem;">
        Navigation Menu
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation",
        [
            "📊 Executive Overview",
            "👤 Student Risk & SHAP Profiler",
            "📋 Batch Prediction & CSV Upload",
            "🔬 Model Performance & EDA Lab"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    # Surveillance Cohort Dataset Selector
    st.markdown("""
    <div style="font-size: 0.6875rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1.25rem; margin-bottom: 0.45rem;">
        Surveillance Cohort Dataset
    </div>
    """, unsafe_allow_html=True)
    
    cohort_keys = list(COHORT_REGISTRY.keys())
    if st.session_state.get("is_custom_cohort", False):
        custom_key_label = f"📂 Custom Active Cohort ({len(df_students):,} Students)"
        cohort_options = [custom_key_label] + cohort_keys
        default_idx = 0
    else:
        current_k = st.session_state.get("selected_cohort_key", cohort_keys[0])
        cohort_options = cohort_keys
        default_idx = cohort_options.index(current_k) if current_k in cohort_options else 0

    chosen_cohort = st.selectbox(
        "Select Active Cohort",
        cohort_options,
        index=default_idx,
        label_visibility="collapsed",
        key="sidebar_cohort_selector"
    )

    if not chosen_cohort.startswith("📂 Custom Active") and chosen_cohort != st.session_state.get("selected_cohort_key"):
        c_path = COHORT_REGISTRY[chosen_cohort]["file"]
        st.session_state.active_students = load_cohort_dataset(c_path)
        st.session_state.selected_cohort_key = chosen_cohort
        st.session_state.is_custom_cohort = False
        st.rerun()

    if st.session_state.get("is_custom_cohort", False):
        st.markdown(f"""
        <div style="font-size: 0.72rem; color: #1E3A8A; background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 4px; padding: 4px 8px; margin-top: 0.35rem; margin-bottom: 0.35rem; text-align: center; font-weight: 500;">
            Active: Custom Cohort ({len(df_students):,} students)
        </div>
        """, unsafe_allow_html=True)
        if st.button("↺ Reset to Baseline (2,500)", use_container_width=True):
            st.session_state.active_students = load_cohort_dataset(COHORT_REGISTRY["🏛️ Institutional Baseline (2,500 Students)"]["file"])
            st.session_state.selected_cohort_key = "🏛️ Institutional Baseline (2,500 Students)"
            st.session_state.is_custom_cohort = False
            st.rerun()

    st.markdown("""
    <div style="font-size: 0.6875rem; font-weight: 700; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1.25rem; margin-bottom: 0.45rem;">
        Active ML Engine
    </div>
    <div class="flat-card" style="padding: 0.875rem; font-size: 0.8125rem; margin-bottom: 0.6rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; padding-bottom: 0.4rem; border-bottom: 1px solid #F1F5F9;">
            <span style="font-weight: 600; color: #0F172A;">Architecture</span>
            <span style="background-color: #EFF6FF; color: #1E3A8A; font-weight: 600; padding: 2px 6px; border-radius: 4px; font-size: 0.72rem;">Tuned XGBoost</span>
        </div>
        <div style="display: flex; flex-direction: column; gap: 0.35rem; color: #475569; font-size: 0.75rem;">
            <div style="display: flex; justify-content: space-between;">
                <span>Sampling:</span>
                <span style="color: #0F172A; font-weight: 500;">SMOTE (Train-only)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Explainability:</span>
                <span style="color: #0F172A; font-weight: 500;">TreeSHAP (Exact)</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>High-Risk Recall:</span>
                <span style="color: #16A34A; font-weight: 600;">78.6%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Test ROC-AUC:</span>
                <span style="color: #1E3A8A; font-weight: 600;">0.946</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    sidebar_monitored = len(df_students)
    sidebar_high_risk = int((df_students["Risk_Level"] == "High").sum()) if "Risk_Level" in df_students.columns else 0
    sidebar_risk_pct = (sidebar_high_risk / max(1, sidebar_monitored)) * 100
    sidebar_risk_str = f"{sidebar_risk_pct:.1f}%"
    sidebar_monitored_str = f"{sidebar_monitored:,}"

    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 0.6rem;">
        <div class="flat-card" style="padding: 0.6rem; margin-bottom: 0; text-align: center;">
            <div style="font-size: 0.68rem; color: #64748B;">Monitored</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #0F172A;">{sidebar_monitored_str}</div>
        </div>
        <div class="flat-card" style="padding: 0.6rem; margin-bottom: 0; text-align: center; border-left: 3px solid #DC2626;">
            <div style="font-size: 0.68rem; color: #DC2626;">At-Risk</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #DC2626;">{sidebar_risk_str}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # (Custom cohort reset is handled in the cohort selector above)
    


# =============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# =============================================================================
if "Executive Overview" in page:
    # Top KPI Metrics Cards
    total_students = len(df_students)
    high_risk_count = (df_students["Risk_Level"] == "High").sum() if "Risk_Level" in df_students.columns else 0
    high_risk_pct = (high_risk_count / max(1, total_students)) * 100
    avg_att = df_students["Attendance_Pct"].mean() if "Attendance_Pct" in df_students.columns else 75.0
    shortage_count = (df_students["Attendance_Pct"] < 75.0).sum() if "Attendance_Pct" in df_students.columns else 0
    avg_cgpa = df_students["Prev_Sem_CGPA"].mean() if "Prev_Sem_CGPA" in df_students.columns else 7.0

    cohort_display_label = st.session_state.get("selected_cohort_key", "🏛️ Institutional Baseline (2,500 Students)")
    if st.session_state.get("is_custom_cohort", False):
        cohort_display_label = f"Custom Ingested Cohort ({total_students:,} Students)"

    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Executive Academic Health Dashboard</h1>
        <p style="font-size: 0.9375rem; color: #475569; margin: 0;">
            Institutional early-warning surveillance across <b>{total_students:,} active enrolled students</b> [{cohort_display_label}]. Identifies attrition and failure risks before End-Term examinations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    tuned_recall = model_comparison.get("Tuned XGBoost (Winning Architecture)", {}).get("High_Risk_Recall", 0.786) * 100
    tuned_f1 = model_comparison.get("Tuned XGBoost (Winning Architecture)", {}).get("F1_Macro", 0.798) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="flat-card">
            <div class="flat-card-header">Total Monitored</div>
            <div class="flat-metric-value">{total_students:,}</div>
            <div class="flat-metric-subtext">{cohort_display_label}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="flat-card" style="border-left: 3px solid #DC2626;">
            <div class="flat-card-header" style="color: #DC2626;">Critical At-Risk</div>
            <div class="flat-metric-value" style="color: #DC2626;">{high_risk_count} <span style="font-size: 1rem; font-weight: 500;">({high_risk_pct:.1f}%)</span></div>
            <div class="flat-metric-subtext">Immediate triage required</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="flat-card">
            <div class="flat-card-header">Attendance Compliance</div>
            <div class="flat-metric-value">{avg_att:.1f}%</div>
            <div class="flat-metric-subtext">{shortage_count} below 75% cutoff</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="flat-card">
            <div class="flat-card-header">Mean Cohort CGPA</div>
            <div class="flat-metric-value">{avg_cgpa:.2f}</div>
            <div class="flat-metric-subtext">Out of 10.0 scale</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="flat-card" style="border-left: 3px solid #1E3A8A;">
            <div class="flat-card-header">Model High-Risk Recall</div>
            <div class="flat-metric-value" style="color: #1E3A8A;">{tuned_recall:.1f}%</div>
            <div class="flat-metric-subtext">Macro F1: {tuned_f1:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Visual Charts Row
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    c_chart1, c_chart2 = st.columns([1, 1.6])

    with c_chart1:
        # Risk Distribution Bar Chart (Solid flat colors)
        risk_counts = df_students["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig_risk = go.Figure(data=[
            go.Bar(
                x=["Low Risk", "Medium Risk", "High Risk"],
                y=risk_counts.values,
                marker=dict(
                    color=["#16A34A", "#D97706", "#DC2626"],
                    line=dict(color="#E2E8F0", width=1)
                ),
                text=[f"{v} ({v/total_students*100:.1f}%)" for v in risk_counts.values],
                textposition="auto"
            )
        ])
        fig_risk = apply_flat_plotly_layout(fig_risk, "Student Cohort Risk Level Distribution", height=340)
        fig_risk.update_yaxes(title="Student Headcount")
        st.plotly_chart(fig_risk, use_container_width=True)

    with c_chart2:
        # Department-wise Risk Breakdown
        dept_risk_df = pd.crosstab(df_students["Department"], df_students["Risk_Level"])
        # Ensure ordered columns
        for c in ["Low", "Medium", "High"]:
            if c not in dept_risk_df.columns:
                dept_risk_df[c] = 0
        dept_risk_df = dept_risk_df[["Low", "Medium", "High"]]

        fig_dept = go.Figure()
        fig_dept.add_trace(go.Bar(
            name="Low Risk", x=dept_risk_df.index, y=dept_risk_df["Low"],
            marker_color="#16A34A", marker_line=dict(color="#E2E8F0", width=1)
        ))
        fig_dept.add_trace(go.Bar(
            name="Medium Risk", x=dept_risk_df.index, y=dept_risk_df["Medium"],
            marker_color="#D97706", marker_line=dict(color="#E2E8F0", width=1)
        ))
        fig_dept.add_trace(go.Bar(
            name="High Risk", x=dept_risk_df.index, y=dept_risk_df["High"],
            marker_color="#DC2626", marker_line=dict(color="#E2E8F0", width=1)
        ))
        fig_dept.update_layout(barmode="stack")
        fig_dept = apply_flat_plotly_layout(fig_dept, "Departmental Academic Vulnerability Distribution", height=340)
        fig_dept.update_xaxes(tickangle=-15)
        fig_dept.update_yaxes(title="Enrolled Students")
        st.plotly_chart(fig_dept, use_container_width=True)

    # Roster Table of Immediate Attention Cases
    st.markdown("""
    <div style="margin-top: 1.5rem; margin-bottom: 0.75rem;">
        <h2 style="font-size: 1.125rem; font-weight: 600; color: #0F172A; margin: 0;">Critical High-Risk Action Roster</h2>
        <p style="font-size: 0.8125rem; color: #64748B; margin-top: 0.15rem;">
            Filtered view of students requiring urgent mentor consultations and official intimation letters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        sel_dept = st.selectbox(
            "Filter by Department",
            ["All Departments"] + list(df_students["Department"].unique()),
            key="overview_dept_filter"
        )
    with col_f2:
        sel_sem = st.selectbox(
            "Filter by Semester",
            ["All Semesters"] + list(range(1, 9)),
            key="overview_sem_filter"
        )

    roster_df = df_students[df_students["Risk_Level"] == "High"].copy()
    if sel_dept != "All Departments":
        roster_df = roster_df[roster_df["Department"] == sel_dept]
    if sel_sem != "All Semesters":
        roster_df = roster_df[roster_df["Semester"] == int(sel_sem)]

    display_cols = [
        "Student_ID", "Department", "Semester", "Attendance_Pct",
        "CA1_Marks", "CA2_Marks", "CA3_Marks", "MidTerm_Marks",
        "Prev_Sem_CGPA", "Active_Backlogs"
    ]
    
    avail_cols = [c for c in display_cols if c in roster_df.columns]
    roster_display = roster_df[avail_cols].copy()
    roster_display.index = range(1, len(roster_display) + 1)
    roster_display.index.name = "Sr. No."
    
    st.dataframe(
        roster_display,
        use_container_width=True,
        height=280
    )
    st.caption(f"Showing {len(roster_df)} High-Risk student records matching active filters.")


# =============================================================================
# PAGE 2: INDIVIDUAL STUDENT RISK & SHAP PROFILER
# =============================================================================
elif "Student Risk" in page:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Individual Student Risk & Explainable AI (XAI) Profiler</h1>
        <p style="font-size: 0.9375rem; color: #475569; margin: 0;">
            Deep-dive diagnostic profiler. Select an existing enrolled student or run interactive What-If simulations to inspect local SHAP attributions and tailored intervention protocols.
        </p>
    </div>
    """, unsafe_allow_html=True)

    profiler_mode = st.radio(
        "Input Mode",
        ["Select Enrolled Student Record", "Interactive 'What-If' Parameter Simulation"],
        horizontal=True,
        label_visibility="collapsed"
    )

    # Initialize student record dictionary
    student_record = {}

    if profiler_mode == "Select Enrolled Student Record":
        c_sel1, c_sel2 = st.columns([1.5, 3.5])
        with c_sel1:
            # Let user search or pick from student IDs
            all_ids = list(df_students["Student_ID"].values)
            # Default to a high-risk student for immediate impressive showcase
            high_ids = list(df_students[df_students["Risk_Level"] == "High"]["Student_ID"].values)
            default_index = all_ids.index(high_ids[0]) if high_ids else 0

            selected_id = st.selectbox(
                "Search / Select Student ID",
                all_ids,
                index=default_index,
                help=f"Type to search among {len(all_ids):,} student records in active cohort."
            )
        
        # Extract row
        matched_row = df_students[df_students["Student_ID"] == selected_id].iloc[0]
        student_record = matched_row.to_dict()

        with c_sel2:
            st.markdown(f"""
            <div class="flat-card" style="padding: 0.875rem 1.25rem; margin-bottom: 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 1.125rem; font-weight: 700; color: #0F172A;">{matched_row['Student_ID']}</div>
                        <div style="font-size: 0.8125rem; color: #475569;">
                            Department: <b>{matched_row['Department']}</b> | Semester: <b>{matched_row['Semester']}</b>
                        </div>
                    </div>
                    <div>
                        <span class="{'badge-high' if matched_row['Risk_Level']=='High' else ('badge-medium' if matched_row['Risk_Level']=='Medium' else 'badge-low')}">
                            Ground Truth: {matched_row['Risk_Level']} Risk
                        </span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else: # What-If Simulation
        st.markdown("""
        <div style="background-color: #F1F5F9; border: 1px solid #E2E8F0; border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.8125rem; color: #475569;">
            <b>What-If Simulator Active:</b> Modify any academic indicator using the sliders below to see live ML probability adjustments, SHAP attributions, and dynamic faculty action plans.
        </div>
        """, unsafe_allow_html=True)

        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            st.markdown("<b>Demographics & Attendance</b>", unsafe_allow_html=True)
            w_dept = st.selectbox("Department", df_students["Department"].unique(), index=0)
            w_sem = st.slider("Semester", min_value=1, max_value=8, value=4)
            w_att = st.slider("Attendance Percentage (%)", min_value=30.0, max_value=100.0, value=68.5, step=0.5)
            w_backlogs = st.number_input("Active Backlogs Count", min_value=0, max_value=8, value=2, step=1)

        with w_col2:
            st.markdown("<b>Assessment Marks (0 - 100)</b>", unsafe_allow_html=True)
            w_ca1 = st.slider("CA1 Marks", min_value=0.0, max_value=100.0, value=42.0, step=0.5)
            w_ca2 = st.slider("CA2 Marks", min_value=0.0, max_value=100.0, value=38.0, step=0.5)
            w_ca3 = st.slider("CA3 Marks", min_value=0.0, max_value=100.0, value=35.0, step=0.5)
            w_midterm = st.slider("Mid-Term Exam Marks", min_value=0.0, max_value=100.0, value=36.0, step=0.5)

        with w_col3:
            st.markdown("<b>Academic History & Engagement</b>", unsafe_allow_html=True)
            w_cgpa = st.slider("Previous Semester CGPA", min_value=0.0, max_value=10.0, value=5.80, step=0.05)
            w_asn = st.slider("Assignment Submission Rate (%)", min_value=0.0, max_value=100.0, value=52.0, step=1.0)
            w_study = st.slider("Study Hours / Week", min_value=0.0, max_value=45.0, value=7.5, step=0.5)
            w_part = st.slider("Class Participation Score (1-10)", min_value=1.0, max_value=10.0, value=4.0, step=0.5)

        student_record = {
            "Department": w_dept,
            "Semester": w_sem,
            "Attendance_Pct": w_att,
            "CA1_Marks": w_ca1,
            "CA2_Marks": w_ca2,
            "CA3_Marks": w_ca3,
            "MidTerm_Marks": w_midterm,
            "Prev_Sem_CGPA": w_cgpa,
            "Assignment_Submission_Rate": w_asn,
            "LMS_Logins_Per_Week": 14,
            "Active_Backlogs": w_backlogs,
            "Study_Hours_Per_Week": w_study,
            "Participation_Score": w_part
        }

    # Prepare DataFrame for Model Inference
    inf_df = pd.DataFrame([student_record])
    if "Student_ID" in inf_df.columns:
        inf_df = inf_df.drop(columns=["Student_ID"])
    if "Risk_Level" in inf_df.columns:
        inf_df = inf_df.drop(columns=["Risk_Level"])

    # Compute Inference & SHAP Values
    shap_df, probs, pred_label = explain_student_risk(inf_df, pipeline, explainer, feature_names, class_idx=2)
    p_low, p_med, p_high = probs[0], probs[1], probs[2]

    # Diagnostic Header Card
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    diag_c1, diag_c2 = st.columns([1.2, 2.8])

    with diag_c1:
        badge_class = "badge-high" if pred_label == "High" else ("badge-medium" if pred_label == "Medium" else "badge-low")
        st.markdown(f"""
        <div class="flat-card" style="height: 100%;">
            <div class="flat-card-header">Model Prediction Outcome</div>
            <div style="margin-top: 0.5rem; margin-bottom: 0.75rem;">
                <span class="{badge_class}" style="font-size: 1.125rem; padding: 0.4rem 0.85rem;">
                    {pred_label.upper()} RISK
                </span>
            </div>
            <div style="font-size: 0.8125rem; color: #475569; line-height: 1.6; margin-top: 0.75rem;">
                <div><b>P(High Risk):</b> <span style="color: #DC2626; font-weight: 600;">{p_high*100:.1f}%</span></div>
                <div><b>P(Medium Risk):</b> <span style="color: #D97706; font-weight: 600;">{p_med*100:.1f}%</span></div>
                <div><b>P(Low Risk):</b> <span style="color: #16A34A; font-weight: 600;">{p_low*100:.1f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with diag_c2:
        # Academic Indicators Quick Matrix with Safe NaN Handling
        is_att_na = pd.isna(student_record.get('Attendance_Pct'))
        att_val = float(safe_val(student_record, 'Attendance_Pct', 75.0))
        att_color = '#DC2626' if att_val < 75.0 else '#0F172A'
        att_display = f"{att_val:.1f}%" if not is_att_na else f"{att_val:.1f}% <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        att_sub = "Cutoff: 75%" if not is_att_na else "Raw: N/A (Cutoff: 75%)"

        is_ca1_na = pd.isna(student_record.get('CA1_Marks'))
        is_ca2_na = pd.isna(student_record.get('CA2_Marks'))
        is_ca3_na = pd.isna(student_record.get('CA3_Marks'))
        ca1_val = float(safe_val(student_record, 'CA1_Marks', 50.0))
        ca2_val = float(safe_val(student_record, 'CA2_Marks', 50.0))
        ca3_val = float(safe_val(student_record, 'CA3_Marks', 50.0))
        ca_avg_val = (ca1_val + ca2_val + ca3_val) / 3.0
        ca_trend_val = ca3_val - ca1_val
        has_ca_na = is_ca1_na or is_ca2_na or is_ca3_na
        ca_display = f"{ca_avg_val:.1f}/100" if not has_ca_na else f"{ca_avg_val:.1f}/100 <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        ca_sub = f"Trend: {ca_trend_val:+.1f}" if not has_ca_na else f"Trend: {ca_trend_val:+.1f} (Imputed)"

        is_mid_na = pd.isna(student_record.get('MidTerm_Marks'))
        mid_val = float(safe_val(student_record, 'MidTerm_Marks', 50.0))
        mid_display = f"{mid_val:.1f}/100" if not is_mid_na else f"{mid_val:.1f}/100 <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        mid_sub = "Weight: 25%" if not is_mid_na else "Raw: N/A (Imputed)"

        is_backlogs_na = pd.isna(student_record.get('Active_Backlogs'))
        backlogs_val = int(safe_val(student_record, 'Active_Backlogs', 0))
        backlogs_color = '#DC2626' if backlogs_val > 0 else '#16A34A'
        backlogs_display = f"{backlogs_val}" if not is_backlogs_na else f"{backlogs_val} <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        backlogs_sub = "Active pending" if not is_backlogs_na else "Raw: N/A (Imputed: 0)"

        is_cgpa_na = pd.isna(student_record.get('Prev_Sem_CGPA'))
        cgpa_val = float(safe_val(student_record, 'Prev_Sem_CGPA', 7.0))
        cgpa_display = f"{cgpa_val:.2f}" if not is_cgpa_na else f"{cgpa_val:.2f} <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        cgpa_sub = "Out of 10.0" if not is_cgpa_na else "Raw: N/A (Imputed)"

        is_asn_na = pd.isna(student_record.get('Assignment_Submission_Rate'))
        asn_val = float(safe_val(student_record, 'Assignment_Submission_Rate', 75.0))
        asn_display = f"{asn_val:.1f}%" if not is_asn_na else f"{asn_val:.1f}% <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        asn_sub = "Target: >80%" if not is_asn_na else "Raw: N/A (Imputed)"

        is_part_na = pd.isna(student_record.get('Participation_Score'))
        part_val = float(safe_val(student_record, 'Participation_Score', 6.0))
        part_display = f"{part_val:.1f}/10" if not is_part_na else f"{part_val:.1f}/10 <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        part_sub = "Lab & Tutorial" if not is_part_na else "Raw: N/A (Imputed)"

        is_study_na = pd.isna(student_record.get('Study_Hours_Per_Week'))
        study_val = float(safe_val(student_record, 'Study_Hours_Per_Week', 15.0))
        study_display = f"{study_val:.1f} hrs/wk" if not is_study_na else f"{study_val:.1f} hrs/wk <span style='font-size:0.6875rem; color:#D97706; font-weight:500;'>(Imputed)</span>"
        study_sub = "Independent Study" if not is_study_na else "Raw: N/A (Imputed)"

        st.markdown(f"""
        <div class="flat-card" style="height: 100%;">
            <div class="flat-card-header">Key Academic Telemetry</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-top: 0.5rem;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Attendance</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: {att_color};">{att_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{att_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">CA Average</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: #0F172A;">{ca_display}</div>
                    <div style="font-size: 0.6875rem; color: {'#DC2626' if ca_trend_val<0 else '#16A34A'};">{ca_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Mid-Term</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: #0F172A;">{mid_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{mid_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Backlogs</div>
                    <div style="font-size: 1.125rem; font-weight: 700; color: {backlogs_color};">{backlogs_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{backlogs_sub}</div>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-top: 0.75rem; padding-top: 0.5rem; border-top: 1px solid #F1F5F9;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Prev CGPA</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0F172A;">{cgpa_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{cgpa_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Assignment Sub.</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0F172A;">{asn_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{asn_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Class Part.</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0F172A;">{part_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{part_sub}</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748B;">Study Hours</div>
                    <div style="font-size: 1rem; font-weight: 600; color: #0F172A;">{study_display}</div>
                    <div style="font-size: 0.6875rem; color: #94A3B8;">{study_sub}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # SHAP Explanations and Faculty Action Plan
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    c_xai, c_actions = st.columns([1.5, 1.5])

    with c_xai:
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Local SHAP Risk Attribution (TreeExplainer)</h3>
            <p style="font-size: 0.8125rem; color: #64748B; margin-top: 0.15rem;">
                Features driving prediction toward High Risk (red) vs mitigating risk (blue).
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Plotly Horizontal Bar Chart of Local SHAP
        top_shap = shap_df.head(9).sort_values(by="SHAP_Value", ascending=True)
        bar_colors = ["#DC2626" if val > 0 else "#2563EB" for val in top_shap["SHAP_Value"]]

        fig_shap = go.Figure(go.Bar(
            x=top_shap["SHAP_Value"],
            y=top_shap["Display_Feature"],
            orientation="h",
            marker=dict(color=bar_colors, line=dict(color="#E2E8F0", width=1)),
            text=[f"{v:+.3f}" for v in top_shap["SHAP_Value"]],
            textposition="auto"
        ))
        fig_shap = apply_flat_plotly_layout(fig_shap, "", height=360)
        fig_shap.update_xaxes(title="SHAP Value (Impact on Log-Odds of High Risk)")
        st.plotly_chart(fig_shap, use_container_width=True)

        st.caption("Positive SHAP values push student toward High Risk. Negative values push student toward Academic Safety.")

    with c_actions:
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Actionable Faculty & Mentor Intervention Plan</h3>
            <p style="font-size: 0.8125rem; color: #64748B; margin-top: 0.15rem;">
                Rule-based institutional protocols triggered by student metrics and SHAP risk drivers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        interventions = generate_actionable_interventions(student_record)
        
        for item in interventions:
            p_class = (
                "intervention-item-critical" if item["priority"] == "CRITICAL"
                else ("intervention-item-medium" if item["priority"] == "MEDIUM"
                else ("intervention-item-low" if "COMMENDATION" in item["priority"]
                else "intervention-item"))
            )
            
            st.markdown(f"""
            <div class="{p_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="font-size: 0.6875rem; font-weight: 700; color: {item['priority_color']}; text-transform: uppercase; letter-spacing: 0.05em;">
                            {item['priority']} PRIORITY • {item['category']}
                        </span>
                        <div style="font-size: 0.9375rem; font-weight: 600; color: #0F172A; margin-top: 0.2rem;">
                            {item['title']}
                        </div>
                    </div>
                    <span style="font-size: 0.75rem; background-color: #F1F5F9; color: #475569; padding: 2px 8px; border-radius: 4px; font-weight: 500;">
                        {item['metric']}
                    </span>
                </div>
                <div style="font-size: 0.8125rem; color: #334155; margin-top: 0.5rem; line-height: 1.5;">
                    {item['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: BATCH PREDICTION & CSV UPLOAD
# =============================================================================
elif "Batch Prediction" in page:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Batch Student Evaluation & Remedial Triage</h1>
        <p style="font-size: 0.9375rem; color: #475569; margin: 0;">
            Upload departmental student spreadsheets to classify academic risk across an entire section. Automatically generates prioritized intervention strategies for academic mentoring committees.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚡ Quick Roster Actions: Enroll New Student or Withdraw Student", expanded=False):
        act_tab1, act_tab2 = st.tabs(["➕ Enroll Individual Student", "➖ Withdraw / Remove Student"])
        
        with act_tab1:
            with st.form("enroll_student_form", clear_on_submit=True):
                e_c1, e_c2, e_c3 = st.columns(3)
                with e_c1:
                    new_id = st.text_input("Student Registration ID", value=f"LPU12{len(df_students)+1:04d}")
                    new_dept = st.selectbox("Department", df_students["Department"].unique(), key="add_dept")
                    new_sem = st.slider("Semester", 1, 8, 1, key="add_sem")
                    new_att = st.slider("Attendance %", 0.0, 100.0, 80.0, 0.5, key="add_att")
                with e_c2:
                    new_ca1 = st.number_input("CA1 Marks (0-100)", 0.0, 100.0, 65.0, 1.0, key="add_ca1")
                    new_ca2 = st.number_input("CA2 Marks (0-100)", 0.0, 100.0, 65.0, 1.0, key="add_ca2")
                    new_ca3 = st.number_input("CA3 Marks (0-100)", 0.0, 100.0, 65.0, 1.0, key="add_ca3")
                    new_mid = st.number_input("Mid-Term Marks (0-100)", 0.0, 100.0, 60.0, 1.0, key="add_mid")
                with e_c3:
                    new_cgpa = st.number_input("Prev CGPA (0-10)", 0.0, 10.0, 7.5, 0.1, key="add_cgpa")
                    new_asn = st.slider("Assignment Rate %", 0.0, 100.0, 85.0, 1.0, key="add_asn")
                    new_backlogs = st.number_input("Active Backlogs", 0, 10, 0, 1, key="add_bklg")
                    new_study = st.number_input("Study Hours/Week", 0.0, 60.0, 15.0, 1.0, key="add_std")
                    new_part = st.number_input("Participation Score (1-10)", 1.0, 10.0, 7.0, 0.5, key="add_prt")
                
                submitted = st.form_submit_button("Enroll Student into Active Surveillance")
                if submitted:
                    if new_id in df_students["Student_ID"].values:
                        st.error(f"Student ID {new_id} already exists in active surveillance.")
                    else:
                        new_row = {
                            "Student_ID": new_id,
                            "Department": new_dept,
                            "Semester": new_sem,
                            "Attendance_Pct": new_att,
                            "CA1_Marks": new_ca1,
                            "CA2_Marks": new_ca2,
                            "CA3_Marks": new_ca3,
                            "MidTerm_Marks": new_mid,
                            "Prev_Sem_CGPA": new_cgpa,
                            "Assignment_Submission_Rate": new_asn,
                            "LMS_Logins_Per_Week": 14,
                            "Active_Backlogs": new_backlogs,
                            "Study_Hours_Per_Week": new_study,
                            "Participation_Score": new_part
                        }
                        temp_req = [
                            "Department", "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks",
                            "CA3_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
                            "LMS_Logins_Per_Week", "Active_Backlogs", "Study_Hours_Per_Week", "Participation_Score"
                        ]
                        temp_df = pd.DataFrame([new_row])[temp_req]
                        pred_code = pipeline.predict(temp_df)[0]
                        inv_map = {0: "Low", 1: "Medium", 2: "High"}
                        new_row["Risk_Level"] = inv_map[pred_code]
                        
                        updated_df = pd.concat([df_students, pd.DataFrame([new_row])], ignore_index=True)
                        st.session_state.active_students = updated_df
                        st.session_state.is_custom_cohort = True
                        st.success(f"Enrolled student {new_id} ({new_dept}) classified as {new_row['Risk_Level']} Risk! Total monitored is now {len(updated_df):,}.")
                        st.rerun()

        with act_tab2:
            r_c1, r_c2 = st.columns([3, 1])
            with r_c1:
                rem_id = st.selectbox("Select Student ID to Withdraw", df_students["Student_ID"].values, key="rem_id_sel")
            with r_c2:
                st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("Withdraw Student", type="secondary", use_container_width=True):
                    updated_df = df_students[df_students["Student_ID"] != rem_id].reset_index(drop=True)
                    st.session_state.active_students = updated_df
                    st.session_state.is_custom_cohort = True
                    st.warning(f"Student {rem_id} withdrawn from active surveillance. Total monitored is now {len(updated_df):,}.")
                    st.rerun()

    st.markdown("""
    <div style="font-size: 1rem; font-weight: 600; color: #0F172A; margin-top: 0.5rem; margin-bottom: 0.5rem;">
        Select Ingestion Source: Pre-Configured Departmental Cohort or External CSV
    </div>
    """, unsafe_allow_html=True)

    batch_tab1, batch_tab2 = st.tabs([
        "⚡ Select Pre-Packaged Departmental Cohort",
        "📂 Upload External Student Roster CSV"
    ])

    batch_df = None
    batch_source_name = ""

    with batch_tab1:
        st.markdown("""
        <div style="font-size: 0.8125rem; color: #475569; margin-bottom: 0.75rem;">
            Select any departmental or specialized cohort to immediately execute model inference, view prioritized interventions, or download the dataset.
        </div>
        """, unsafe_allow_html=True)
        
        b_c1, b_c2, b_c3 = st.columns([2.5, 1.2, 1.2])
        with b_c1:
            pre_cohort_choice = st.selectbox(
                "Choose Departmental Cohort Dataset",
                list(COHORT_REGISTRY.keys()),
                key="batch_pre_cohort_select"
            )
        
        target_cohort_file = COHORT_REGISTRY[pre_cohort_choice]["file"]
        
        with b_c2:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Run Batch Inference", type="primary", use_container_width=True):
                st.session_state.active_batch_source = "preset"
                st.session_state.preset_cohort_choice = pre_cohort_choice
        
        with b_c3:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            if os.path.exists(target_cohort_file):
                with open(target_cohort_file, "rb") as f_dl:
                    st.download_button(
                        label="📥 Download CSV",
                        data=f_dl.read(),
                        file_name=os.path.basename(target_cohort_file),
                        mime="text/csv",
                        use_container_width=True
                    )

    with batch_tab2:
        st.markdown("""
        <div style="font-size: 0.8125rem; color: #475569; margin-bottom: 0.75rem;">
            Upload any institution student roster (.csv) to evaluate academic risk for your custom section.
        </div>
        """, unsafe_allow_html=True)
        
        up_c1, up_c2 = st.columns([3, 1])
        with up_c1:
            uploaded_file = st.file_uploader(
                "Upload Student Roster CSV",
                type=["csv"],
                help="Upload a CSV with student academic features (Semester, Attendance_Pct, CA Marks, etc.).",
                key="custom_csv_uploader"
            )
        with up_c2:
            st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
            sample_template_path = os.path.join("data", "blank_template_student_roster.csv")
            if not os.path.exists(sample_template_path):
                sample_template_path = os.path.join("data", "sample_batch_upload.csv")
            if os.path.exists(sample_template_path):
                with open(sample_template_path, "rb") as f_tpl:
                    st.download_button(
                        label="📥 Sample CSV Template",
                        data=f_tpl.read(),
                        file_name="sample_student_roster_template.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_source_name = uploaded_file.name
            st.session_state.active_batch_source = "upload"
            st.success(f"Successfully loaded {len(batch_df)} student records from '{uploaded_file.name}'.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif st.session_state.get("active_batch_source") == "preset":
        chosen_p = st.session_state.get("preset_cohort_choice", pre_cohort_choice)
        c_path = COHORT_REGISTRY[chosen_p]["file"]
        if os.path.exists(c_path):
            batch_df = pd.read_csv(c_path)
            batch_source_name = chosen_p
            st.info(f"Displaying batch inference for pre-configured cohort: **{chosen_p}** ({len(batch_df):,} records).")

    if batch_df is not None:
        # Validate required columns (LMS logins completely excluded from requirements)
        req_cols = [
            "Department", "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks",
            "CA3_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
            "Active_Backlogs", "Study_Hours_Per_Week", "Participation_Score"
        ]
        missing_cols = [c for c in req_cols if c not in batch_df.columns]
        
        if missing_cols:
            st.error(f"Uploaded CSV is missing mandatory columns: {missing_cols}")
        else:
            with st.spinner("Processing batch through Scikit-Learn Pipeline & XGBoost..."):
                # Clean inference dataframe & ensure pipeline expected feature structure
                pipeline_features = [
                    "Department", "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks",
                    "CA3_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
                    "LMS_Logins_Per_Week", "Active_Backlogs", "Study_Hours_Per_Week", "Participation_Score"
                ]
                X_batch = batch_df[[c for c in req_cols if c in batch_df.columns]].copy()
                if "LMS_Logins_Per_Week" not in X_batch.columns:
                    X_batch["LMS_Logins_Per_Week"] = 14
                X_batch = X_batch[pipeline_features]
                
                # Apply model
                batch_probs = pipeline.predict_proba(X_batch)
                batch_preds = pipeline.predict(X_batch)
                inv_map = {0: "Low", 1: "Medium", 2: "High"}
                pred_labels = [inv_map[p] for p in batch_preds]

                # Annotate dataframe (clean display without any LMS column)
                res_df = batch_df.copy()
                if "LMS_Logins_Per_Week" in res_df.columns:
                    res_df = res_df.drop(columns=["LMS_Logins_Per_Week"])
                res_df["Predicted_Risk"] = pred_labels
                res_df["Prob_High_Risk"] = (batch_probs[:, 2] * 100).round(1)
                res_df["Prob_Medium_Risk"] = (batch_probs[:, 1] * 100).round(1)
                res_df["Prob_Low_Risk"] = (batch_probs[:, 0] * 100).round(1)

                # Generate primary action for each student
                primary_actions = []
                action_priorities = []
                for _, row in res_df.iterrows():
                    actions = generate_actionable_interventions(row.to_dict())
                    if actions:
                        primary_actions.append(actions[0]["title"] + ": " + actions[0]["action"])
                        action_priorities.append(actions[0]["priority"])
                    else:
                        primary_actions.append("Nominate for Honors & Research Fellowship")
                        action_priorities.append("LOW RISK")

                res_df["Primary_Intervention"] = primary_actions
                res_df["Intervention_Priority"] = action_priorities

            # Summary KPI Cards
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            b_col1, b_col2, b_col3, b_col4 = st.columns(4)

            total_b = len(res_df)
            high_b = (res_df["Predicted_Risk"] == "High").sum()
            med_b = (res_df["Predicted_Risk"] == "Medium").sum()
            low_b = (res_df["Predicted_Risk"] == "Low").sum()

            with b_col1:
                st.markdown(f"""
                <div class="flat-card">
                    <div class="flat-card-header">Batch Evaluated</div>
                    <div class="flat-metric-value">{total_b}</div>
                    <div class="flat-metric-subtext">Students analyzed</div>
                </div>
                """, unsafe_allow_html=True)

            with b_col2:
                st.markdown(f"""
                <div class="flat-card" style="border-left: 3px solid #DC2626;">
                    <div class="flat-card-header" style="color: #DC2626;">High Risk</div>
                    <div class="flat-metric-value" style="color: #DC2626;">{high_b} ({high_b/total_b*100:.1f}%)</div>
                    <div class="flat-metric-subtext">Immediate intervention needed</div>
                </div>
                """, unsafe_allow_html=True)

            with b_col3:
                st.markdown(f"""
                <div class="flat-card" style="border-left: 3px solid #D97706;">
                    <div class="flat-card-header" style="color: #D97706;">Medium Risk</div>
                    <div class="flat-metric-value" style="color: #D97706;">{med_b} ({med_b/total_b*100:.1f}%)</div>
                    <div class="flat-metric-subtext">Moderate attention</div>
                </div>
                """, unsafe_allow_html=True)

            with b_col4:
                st.markdown(f"""
                <div class="flat-card" style="border-left: 3px solid #16A34A;">
                    <div class="flat-card-header" style="color: #16A34A;">Low Risk</div>
                    <div class="flat-metric-value" style="color: #16A34A;">{low_b} ({low_b/total_b*100:.1f}%)</div>
                    <div class="flat-metric-subtext">Academic safety zone</div>
                </div>
                """, unsafe_allow_html=True)

            # COHORT SYNCHRONIZATION BAR
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
            sync_c1, sync_c2 = st.columns([2.5, 1.2])
            with sync_c1:
                st.markdown(f"""
                <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4px solid #1E3A8A; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.8125rem; color: #1E3A8A;">
                    <b>Synchronize System Surveillance:</b> Activate these {total_b:,} evaluated student records as the active university cohort. This immediately updates the Executive Overview, monitored totals, departmental distribution charts, and individual profiler.
                </div>
                """, unsafe_allow_html=True)
            with sync_c2:
                st.markdown("<div style='height: 0.25rem;'></div>", unsafe_allow_html=True)
                if st.button("🔄 Activate as Monitored Cohort", use_container_width=True, type="primary"):
                    sync_df = res_df.copy()
                    if "Student_ID" not in sync_df.columns:
                        sync_df["Student_ID"] = [f"LPU24_NEW_{i+1:04d}" for i in range(len(sync_df))]
                    sync_df["Risk_Level"] = sync_df["Predicted_Risk"]
                    st.session_state.active_students = sync_df
                    st.session_state.is_custom_cohort = True
                    st.session_state.selected_cohort_key = f"Custom Batch ({len(sync_df):,} students)"
                    st.success(f"System synchronized! Executive Overview now monitors all {len(sync_df):,} students.")
                    st.rerun()

            # Filter & Display Table
            st.markdown("""
            <div style="margin-top: 1rem; margin-bottom: 0.5rem;">
                <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Predicted Risk Roster & Interventions</h3>
            </div>
            """, unsafe_allow_html=True)

            col_bf1, col_bf2 = st.columns([1, 1])
            with col_bf1:
                filter_risk = st.multiselect(
                    "Filter by Predicted Risk",
                    ["High", "Medium", "Low"],
                    default=["High", "Medium", "Low"]
                )
            with col_bf2:
                filter_bdept = st.selectbox(
                    "Filter by Department",
                    ["All Departments"] + list(res_df["Department"].unique())
                )

            filtered_res = res_df[res_df["Predicted_Risk"].isin(filter_risk)]
            if filter_bdept != "All Departments":
                filtered_res = filtered_res[filtered_res["Department"] == filter_bdept]

            cols_to_show = [
                "Student_ID" if "Student_ID" in filtered_res.columns else "Semester",
                "Department", "Attendance_Pct", "MidTerm_Marks", "Prev_Sem_CGPA",
                "Active_Backlogs", "Predicted_Risk", "Prob_High_Risk", "Intervention_Priority",
                "Primary_Intervention"
            ]
            cols_to_show = [c for c in cols_to_show if c in filtered_res.columns]

            batch_display = filtered_res[cols_to_show].copy()
            batch_display.index = range(1, len(batch_display) + 1)
            batch_display.index.name = "Sr. No."

            st.dataframe(batch_display, use_container_width=True, height=320)

            # CSV Download Button
            csv_export = res_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Annotated Intervention Report (CSV)",
                data=csv_export,
                file_name="LPU_Academic_Intervention_Report.csv",
                mime="text/csv"
            )


# =============================================================================
# PAGE 4: MODEL PERFORMANCE & EDA LAB
# =============================================================================
elif "Model Performance" in page:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.75rem; font-weight: 700; color: #0F172A; margin-bottom: 0.25rem;">Supervised Learning Diagnostic & EDA Laboratory</h1>
        <p style="font-size: 0.9375rem; color: #475569; margin: 0;">
            Comprehensive academic modeling documentation, exploratory feature distributions, 5-algorithm benchmark comparison, confusion matrix, ROC curves, global SHAP importance, and error analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab_eda, tab_benchmark, tab_diag, tab_error = st.tabs([
        "1. Exploratory Data Analysis (EDA)",
        "2. 5-Candidate Benchmark & Tuned Winner",
        "3. ROC & Confusion Diagnostics",
        "4. Error Analysis & Limitations"
    ])

    # TAB 1: EDA
    with tab_eda:
        st.markdown("""
        <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Cohort Feature Distributions & Correlation Matrix</h3>
            <p style="font-size: 0.8125rem; color: #64748B; margin-top: 0.15rem;">
                Statistical exploration of 2,500 LPU student records demonstrating feature variance and class imbalance prior to modeling.
            </p>
        </div>
        """, unsafe_allow_html=True)

        e_c1, e_c2 = st.columns(2)

        with e_c1:
            # Correlation Matrix Heatmap
            corr_dict = eda_stats.get("correlation_matrix", {})
            if corr_dict:
                corr_df = pd.DataFrame(corr_dict)
                if "LMS_Logins_Per_Week" in corr_df.columns:
                    corr_df = corr_df.drop(columns=["LMS_Logins_Per_Week"], errors="ignore")
                if "LMS_Logins_Per_Week" in corr_df.index:
                    corr_df = corr_df.drop(index=["LMS_Logins_Per_Week"], errors="ignore")
                fig_corr = px.imshow(
                    corr_df,
                    text_auto=".2f",
                    color_continuous_scale=[[0, "#EFF6FF"], [0.5, "#93C5FD"], [1, "#1E3A8A"]],
                    aspect="auto"
                )
                fig_corr = apply_flat_plotly_layout(fig_corr, "Academic Indicator Correlation Matrix", height=420)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Correlation data not available.")

        with e_c2:
            # Attendance vs Mid-Term Marks Scatter colored by Risk
            fig_scatter = px.scatter(
                df_students,
                x="Attendance_Pct",
                y="MidTerm_Marks",
                color="Risk_Level",
                color_discrete_map={"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"},
                opacity=0.75,
                labels={"Attendance_Pct": "Attendance %", "MidTerm_Marks": "Mid-Term Marks (0-100)"}
            )
            # Add 75% attendance threshold line
            fig_scatter.add_vline(x=75.0, line_dash="dash", line_color="#0F172A", annotation_text="75% Cutoff")
            fig_scatter = apply_flat_plotly_layout(fig_scatter, "Attendance vs Mid-Term Marks (Colored by Risk)", height=420)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # SMOTE Balancing Comparison
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<b>Class Imbalance Mitigation: SMOTE (Synthetic Minority Over-sampling Technique)</b>", unsafe_allow_html=True)
        st.write("To prevent model bias toward majority 'Low Risk' students, SMOTE is applied strictly on the Training set (70%) within the scikit-learn cross-validation folds.")

        if smote_summary:
            before_smote = smote_summary.get("before_smote", {})
            after_smote = smote_summary.get("after_smote", {})
            
            classes = ["Low", "Medium", "High"]
            b_vals = [before_smote.get(c, 0) for c in classes]
            a_vals = [after_smote.get(c, 0) for c in classes]

            fig_smote = go.Figure(data=[
                go.Bar(name="Before SMOTE (Imbalanced Train)", x=classes, y=b_vals, marker_color="#94A3B8"),
                go.Bar(name="After SMOTE (Balanced Train)", x=classes, y=a_vals, marker_color="#1E3A8A")
            ])
            fig_smote.update_layout(barmode="group")
            fig_smote = apply_flat_plotly_layout(fig_smote, "Training Set Class Distribution Before & After SMOTE", height=320)
            fig_smote.update_yaxes(title="Sample Count")
            st.plotly_chart(fig_smote, use_container_width=True)

    # TAB 2: BENCHMARK
    with tab_benchmark:
        st.markdown("""
        <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Model Benchmark Comparison & Tuning (5 Core Paradigms + 1 Tuned Architecture)</h3>
            <p style="font-size: 0.8125rem; color: #475569; margin-top: 0.15rem;">
                Compares the 5 foundational candidate algorithms (Baseline Logistic Regression, Random Forest, SVM, XGBoost, and LightGBM) alongside the final <b>Tuned XGBoost</b> architecture optimized via <code>RandomizedSearchCV</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if model_comparison:
            comp_df = pd.DataFrame(model_comparison).T
            # Reorder columns for readability
            pref_cols = [
                "Accuracy", "High_Risk_Recall", "High_Risk_Precision", "High_Risk_F1",
                "F1_Macro", "Precision_Macro", "Recall_Macro", "ROC_AUC_Macro"
            ]
            valid_cols = [c for c in pref_cols if c in comp_df.columns]
            comp_df = comp_df[valid_cols]

            st.dataframe(comp_df.style.highlight_max(axis=0, color="#E0F2FE"), use_container_width=True)

            st.markdown("""
            <div class="flat-card" style="margin-top: 1rem;">
                <div class="flat-card-header">Why Prioritize High-Risk Recall?</div>
                <div style="font-size: 0.875rem; color: #334155; line-height: 1.6;">
                    In an academic early-warning system, the cost of a <b>False Negative</b> (failing to identify an at-risk student who subsequently fails or drops out) is catastrophic to the student's degree progression and career. 
                    Conversely, a <b>False Positive</b> (flagging a safe student for a routine mentor check-in) carries a negligible cost. 
                    Hence, hyperparameter tuning optimized for <b>High-Risk Recall</b> and Macro F1 ensures no vulnerable student falls through institutional cracks.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Global SHAP Feature Importance Plot
            st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
            st.markdown("<b>Global Feature Importance (SHAP TreeExplainer across entire Cohort)</b>", unsafe_allow_html=True)
            if global_shap_data:
                g_df = pd.DataFrame(global_shap_data)
                g_df = g_df[~g_df["Feature"].str.contains("LMS", case=False)].head(10).sort_values(by="High_Risk_Importance", ascending=True)
                fig_gshap = go.Figure(go.Bar(
                    x=g_df["High_Risk_Importance"],
                    y=g_df["Feature"],
                    orientation="h",
                    marker=dict(color="#1E3A8A", line=dict(color="#E2E8F0", width=1))
                ))
                fig_gshap = apply_flat_plotly_layout(fig_gshap, "Top 10 Global Drivers of High Academic Risk", height=350)
                fig_gshap.update_xaxes(title="Mean |SHAP Value| (Impact on High-Risk Classification)")
                st.plotly_chart(fig_gshap, use_container_width=True)

    # TAB 3: DIAGNOSTICS (Confusion Matrix & ROC)
    with tab_diag:
        st.markdown("""
        <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Confusion Matrix & One-vs-Rest ROC Analysis</h3>
            <p style="font-size: 0.8125rem; color: #475569; margin-top: 0.15rem;">
                Visual verification of classification boundaries on the held-out 375 test samples.
            </p>
        </div>
        """, unsafe_allow_html=True)

        d_c1, d_c2 = st.columns(2)

        with d_c1:
            # Confusion Matrix
            if confusion_matrix_data:
                cm_matrix = confusion_matrix_data.get("matrix_raw", [])
                labels = confusion_matrix_data.get("labels", ["Low", "Medium", "High"])
                
                # Annotate text
                cm_text = []
                for r in cm_matrix:
                    row_txt = [str(val) for val in r]
                    cm_text.append(row_txt)

                fig_cm = px.imshow(
                    cm_matrix,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    color_continuous_scale=[[0, "#FFFFFF"], [0.2, "#DBEAFE"], [1, "#1E3A8A"]],
                    labels=dict(x="Predicted Risk Level", y="Actual Risk Level")
                )
                fig_cm = apply_flat_plotly_layout(fig_cm, "Test Set Confusion Matrix (Tuned XGBoost)", height=380)
                st.plotly_chart(fig_cm, use_container_width=True)

        with d_c2:
            # ROC Curves
            if roc_data:
                fig_roc = go.Figure()
                colors = {"Low": "#16A34A", "Medium": "#D97706", "High": "#DC2626"}
                
                for c_name, d in roc_data.items():
                    fig_roc.add_trace(go.Scatter(
                        x=d["fpr"],
                        y=d["tpr"],
                        name=f"{c_name} Risk (AUC = {d['auc']:.3f})",
                        mode="lines",
                        line=dict(color=colors.get(c_name, "#1E3A8A"), width=2)
                    ))
                    
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name="Random Classifier (AUC = 0.50)",
                    mode="lines",
                    line=dict(color="#94A3B8", dash="dash")
                ))
                fig_roc = apply_flat_plotly_layout(fig_roc, "Multi-Class One-vs-Rest ROC Curves", height=380)
                fig_roc.update_xaxes(title="False Positive Rate")
                fig_roc.update_yaxes(title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

    # TAB 4: ERROR ANALYSIS & LIMITATIONS
    with tab_error:
        st.markdown("""
        <div style="margin-top: 0.5rem; margin-bottom: 1rem;">
            <h3 style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin: 0;">Error Analysis & Academic Safeguards</h3>
            <p style="font-size: 0.8125rem; color: #475569; margin-top: 0.15rem;">
                Breakdown of misclassifications, critical false negatives, ethical governance, and operational limitations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if error_analysis:
            err_c1, err_c2, err_c3 = st.columns(3)
            with err_c1:
                st.markdown(f"""
                <div class="flat-card">
                    <div class="flat-card-header">Test Misclassifications</div>
                    <div class="flat-metric-value">{error_analysis.get('total_misclassifications', 0)} / {error_analysis.get('total_test_samples', 0)}</div>
                    <div class="flat-metric-subtext">Overall error rate: {error_analysis.get('overall_error_rate', 0)*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with err_c2:
                st.markdown(f"""
                <div class="flat-card" style="border-left: 3px solid #DC2626;">
                    <div class="flat-card-header" style="color: #DC2626;">Critical False Negatives</div>
                    <div class="flat-metric-value" style="color: #DC2626;">{error_analysis.get('critical_false_negatives_count', 0)}</div>
                    <div class="flat-metric-subtext">High Risk predicted as Medium/Low</div>
                </div>
                """, unsafe_allow_html=True)

            with err_c3:
                st.markdown(f"""
                <div class="flat-card" style="border-left: 3px solid #D97706;">
                    <div class="flat-card-header" style="color: #D97706;">False Alarms (FP)</div>
                    <div class="flat-metric-value" style="color: #D97706;">{error_analysis.get('false_alarm_fp_count', 0)}</div>
                    <div class="flat-metric-subtext">Low Risk predicted as High</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="flat-card" style="margin-top: 1rem;">
                <div class="flat-card-header">Misclassification Diagnosis: False Negatives vs. False Positives</div>
                <div style="font-size: 0.875rem; color: #334155; line-height: 1.6;">
                    <ul>
                        <li><b>Critical False Negatives Analysis:</b> Students with borderline attendance (~76-78%) but severe drops in late CA assessments (e.g. CA3 &lt; 25) can occasionally be assigned to 'Medium Risk' rather than 'High Risk'. <i>Mitigation:</i> The threshold for mentor alert triggering is deliberately set to P(High) &gt; 35% in operational deployments.</li>
                        <li><b>False Alarms Analysis:</b> Safe students who suffer a single missed assignment or irregular study patterns are sometimes flagged as Medium. This is safe behavior since it triggers an advisory email without disciplinary sanctions.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Institutional Safeguards & Governance
        st.markdown("""
        <div class="flat-card">
            <div class="flat-card-header">Institutional Safeguards & Ethical AI Governance</div>
            <div style="font-size: 0.875rem; color: #334155; line-height: 1.6;">
                <ol>
                    <li><b>Strict Human-in-the-Loop Requirement:</b> Under university academic policy, the ML model serves solely as a decision-support advisory tool. No automated disciplinary sanctions, debarments, or grade penalties are ever executed without manual review by the Faculty Academic Advisory Committee.</li>
                    <li><b>Data Privacy & Access Control (FERPA & UGC Compliance):</b> Student marks, attendance records, and risk predictions are restricted to designated faculty mentors and heads of departments via role-based access control.</li>
                    <li><b>Demographic Parity & Bias Mitigation:</b> The dataset and model exclude socio-economic, gender, and demographic attributes. Only academic progress telemetry (attendance, CAs, assignment submissions) is utilized.</li>
                    <li><b>Continuous Assessment Model Drift Monitoring:</b> The pipeline must be retrained every academic semester following Mid-Term examination results to adapt to curriculum updates.</li>
                </ol>
            </div>
        </div>
        """, unsafe_allow_html=True)