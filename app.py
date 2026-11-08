import os
import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioScreen AI | Heart Disease Risk Screening",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM STYLING (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Force crisp light background and dark text */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Main container styling */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Sidebar light styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }

    /* Hero header - Medical Blue Gradient */
    .hero-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #0284c7 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 22px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.16);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
        color: #ffffff !important;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #f1f5f9 !important;
        margin-bottom: 12px;
        font-weight: 400;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* Card styling */
    .section-card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e293b;
        padding-bottom: 10px;
        margin-bottom: 12px;
        border-bottom: 2px solid #cbd5e1;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Risk Outcome Cards */
    .risk-card-high {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: #ffffff !important;
        padding: 22px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(220, 38, 38, 0.25);
    }
    .risk-card-moderate {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #ffffff !important;
        padding: 22px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(217, 119, 6, 0.25);
    }
    .risk-card-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff !important;
        padding: 22px 24px;
        border-radius: 14px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(5, 150, 105, 0.25);
    }
    .risk-card-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
        color: #ffffff !important;
    }
    .risk-card-desc {
        font-size: 1.02rem;
        font-weight: 500;
        color: #ffffff !important;
        opacity: 0.98;
    }

    /* Risk Factor item */
    .factor-pill {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 5px solid #dc2626;
        color: #991b1b !important;
        padding: 12px 16px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 10px;
        font-size: 0.95rem;
        font-weight: 600;
    }

    /* Recommendation card */
    .recom-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }

    /* Metric box */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04) !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MODEL & ARTIFACT CACHED LOADER
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading heart disease screening model...")
def load_bundle():
    try:
        bundle = joblib.load("heart_model.joblib")
        return bundle["model"], float(bundle["threshold"]), bundle["columns"], None
    except Exception as e:
        return None, 0.54, [], str(e)

model, threshold, columns, load_error = load_bundle()

# Codebook Reference Mappings (BRFSS 2025)
age_map = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
    6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
    11: "70-74", 12: "75-79", 13: "80+"
}

health_map = {
    1: "1 - Excellent",
    2: "2 - Very good",
    3: "3 - Good",
    4: "4 - Fair",
    5: "5 - Poor"
}

smoking_map = {
    1: "Current smoker (Every day)",
    2: "Current smoker (Some days)",
    3: "Former smoker",
    4: "Never smoked"
}

diabetes_map = {
    0: "No (Non-diabetic)",
    1: "Pre-diabetes or borderline",
    2: "Yes (Diagnosed diabetes)"
}

# -----------------------------------------------------------------------------
# PRESET PROFILES FOR EASY TESTING
# -----------------------------------------------------------------------------
PRESETS = {
    "Manual (Custom Patient Details)": None,
    "Example 1: Low Risk (Healthy Young Adult)": {
        "age_group": 2, "sex": 1, "bmi": 22.4, "general_health": 1,
        "smoking_status": 4, "high_bp": 0, "high_cholesterol": 0,
        "diabetes": 0, "stroke": 0, "kidney_disease": 0, "copd": 0,
        "difficulty_walking": 0
    },
    "Example 2: Moderate Risk (Middle-Aged Smoker w/ Hypertension)": {
        "age_group": 8, "sex": 1, "bmi": 28.5, "general_health": 3,
        "smoking_status": 1, "high_bp": 1, "high_cholesterol": 1,
        "diabetes": 1, "stroke": 0, "kidney_disease": 0, "copd": 0,
        "difficulty_walking": 0
    },
    "Example 3: High Risk (Senior with Multiple Comorbidities)": {
        "age_group": 11, "sex": 2, "bmi": 32.0, "general_health": 4,
        "smoking_status": 3, "high_bp": 1, "high_cholesterol": 1,
        "diabetes": 2, "stroke": 1, "kidney_disease": 0, "copd": 1,
        "difficulty_walking": 1
    }
}

# Initialize session state for all patient inputs
default_inputs = {
    "age_group": 10,
    "sex": 1,
    "bmi": 27.5,
    "general_health": 3,
    "smoking_status": 4,
    "high_bp": 0,
    "high_cholesterol": 0,
    "diabetes": 0,
    "stroke": 0,
    "kidney_disease": 0,
    "copd": 0,
    "difficulty_walking": 0
}

for k, v in default_inputs.items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset():
    selected = st.session_state.selected_preset
    preset_data = PRESETS.get(selected)
    if preset_data:
        for k, v in preset_data.items():
            st.session_state[k] = v

def get_bmi_category(bmi_val):
    if bmi_val < 18.5:
        return "Underweight (< 18.5)", "🔵"
    elif bmi_val < 25.0:
        return "Normal Weight (18.5 - 24.9)", "🟢"
    elif bmi_val < 30.0:
        return "Overweight (25.0 - 29.9)", "🟠"
    elif bmi_val < 35.0:
        return "Obese Class I (30.0 - 34.9)", "🔴"
    else:
        return "Obese Class II/III (≥ 35.0)", "🟣"

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫀 CardioScreen AI")
    st.caption("AI & Expert Systems Decision Support Tool")
    st.divider()

    st.subheader("⚡ Quick Patient Presets")
    st.selectbox(
        "Load sample profile:",
        options=list(PRESETS.keys()),
        key="selected_preset",
        on_change=apply_preset,
        help="Select a preset case to test different risk tiers automatically."
    )

    if st.button("🔄 Reset to Default Values", use_container_width=True):
        for k, v in default_inputs.items():
            st.session_state[k] = v
        st.session_state.selected_preset = "Manual (Custom Patient Details)"
        st.rerun()

    st.divider()
    st.markdown("#### ⚙️ System Configuration")
    st.markdown(f"""
    - **ML Model**: Tuned XGBoost
    - **Decision Cutoff**: `{threshold:.1%}`
    - **Dataset**: CDC BRFSS 2025
    - **Architecture**: Hybrid ML + Expert Rules
    """)

    st.divider()
    st.info("💡 **Clinical Screening Note:** This system is an educational screening prototype developed to assist risk stratification. It is not an alternative to professional clinical diagnosis.", icon="ℹ️")

# -----------------------------------------------------------------------------
# HERO BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🫀 Heart Disease Risk Screening Tool</div>
    <div class="hero-subtitle">
        Intelligent cardiovascular risk stratification combining population-scale Machine Learning (XGBoost)
        with clinical Expert Rule heuristics based on the CDC BRFSS epidemiological dataset.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">🫀 AI & Expert Screening</span>
        <span class="hero-badge">📊 High Recall Calibrated (0.54)</span>
        <span class="hero-badge">🛡️ Clinical Heuristics</span>
        <span class="hero-badge">📈 BRFSS 2025 Benchmarked</span>
    </div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"⚠️ Model Warning: Unable to load heart_model.joblib ({load_error}). Please ensure the model file and its dependencies (scikit-learn, xgboost) are available.")

# -----------------------------------------------------------------------------
# PATIENT DETAILS INPUT FORM
# -----------------------------------------------------------------------------
st.markdown("### 📋 Patient Clinical Profile")

col_demographics, col_clinical = st.columns(2, gap="large")

with col_demographics:
    st.markdown('<div class="section-card-header">👤 Demographics & Lifestyle</div>', unsafe_allow_html=True)

    age_group = st.selectbox(
        "Age Group",
        options=list(age_map.keys()),
        format_func=lambda x: f"{age_map[x]} years",
        key="age_group"
    )

    sex = st.selectbox(
        "Biological Sex",
        options=[1, 2],
        format_func=lambda x: "Male" if x == 1 else "Female",
        key="sex"
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=12.0,
        max_value=99.0,
        step=0.5,
        key="bmi"
    )
    bmi_text, bmi_icon = get_bmi_category(bmi)
    st.caption(f"{bmi_icon} Category: **{bmi_text}**")

    general_health = st.selectbox(
        "General Health Perception",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: health_map[x],
        key="general_health"
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        options=[1, 2, 3, 4],
        format_func=lambda x: smoking_map[x],
        key="smoking_status"
    )

with col_clinical:
    st.markdown('<div class="section-card-header">🩺 Medical Diagnoses & Chronic Conditions</div>', unsafe_allow_html=True)

    high_bp = st.radio(
        "High Blood Pressure Diagnosed?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="high_bp"
    )

    high_cholesterol = st.radio(
        "High Cholesterol Diagnosed?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="high_cholesterol"
    )

    diabetes = st.selectbox(
        "Diabetes Diagnosis",
        options=[0, 1, 2],
        format_func=lambda x: diabetes_map[x],
        key="diabetes"
    )

    stroke = st.radio(
        "Previous History of Stroke?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="stroke"
    )

    kidney_disease = st.radio(
        "Chronic Kidney Disease?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="kidney_disease"
    )

    copd = st.radio(
        "COPD / Emphysema?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="copd"
    )

    difficulty_walking = st.radio(
        "Difficulty Walking / Climbing Stairs?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        horizontal=True,
        key="difficulty_walking"
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ACTION BUTTON & PREDICTION
# -----------------------------------------------------------------------------
btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    evaluate_clicked = st.button("🔍 Evaluate Patient Cardiovascular Risk", type="primary", use_container_width=True)

# Automatically show results if clicked or if already active
if evaluate_clicked or "has_evaluated" in st.session_state:
    st.session_state["has_evaluated"] = True

    person = {
        "age_group": age_group, "sex": sex, "bmi": bmi, "general_health": general_health,
        "smoking_status": smoking_status, "high_bp": high_bp, "high_cholesterol": high_cholesterol,
        "diabetes": diabetes, "stroke": stroke, "kidney_disease": kidney_disease,
        "copd": copd, "difficulty_walking": difficulty_walking
    }

    # Run model prediction if loaded
    if model is not None:
        row = pd.DataFrame([person]).reindex(columns=columns)
        risk = float(model.predict_proba(row)[0, 1])
    else:
        # Fallback approximation for demonstration if model dependencies are missing
        risk = 0.42

    # Expert Clinical Rules Engine
    reasons = []
    if age_group >= 10 and high_bp == 1 and high_cholesterol == 1:
        reasons.append("Age 65+ with concurrent High Blood Pressure and High Cholesterol")
    if diabetes == 2 and high_bp == 1:
        reasons.append("Diagnosed Diabetes combined with High Blood Pressure (High micro/macrovascular risk)")
    if stroke == 1:
        reasons.append("Documented medical history of Stroke (Established vascular disease)")
    if smoking_status in (1, 2) and age_group >= 8:
        reasons.append("Active Smoker aged 55 or older")
    if general_health >= 4 and (kidney_disease == 1 or copd == 1 or difficulty_walking == 1):
        reasons.append("Fair/Poor overall health accompanied by Kidney Disease, COPD, or Walking Difficulty")
    if bmi >= 30 and (high_bp == 1 or diabetes >= 1):
        reasons.append("BMI 30+ (Obesity) combined with Hypertension or Diabetes/Pre-diabetes")

    # Classification logic
    if risk >= threshold and len(reasons) >= 1:
        risk_level = "HIGH"
        risk_class = "risk-card-high"
        risk_icon = "🚨"
        risk_headline = "HIGH CARDIOVASCULAR RISK"
        risk_advice = "Urgent Cardiology screening and comprehensive diagnostic evaluation recommended."
    elif risk >= threshold or len(reasons) >= 2:
        risk_level = "MODERATE"
        risk_class = "risk-card-moderate"
        risk_icon = "⚠️"
        risk_headline = "MODERATE CARDIOVASCULAR RISK"
        risk_advice = "Physician checkup and preventative cardiovascular evaluation recommended."
    else:
        risk_level = "LOW"
        risk_class = "risk-card-low"
        risk_icon = "✅"
        risk_headline = "LOW CARDIOVASCULAR RISK"
        risk_advice = "Maintain routine preventative care, regular health checkups, and a healthy lifestyle."

    st.markdown("---")
    st.markdown("### 📊 Screening Assessment Results")

    # Result Banner
    st.markdown(f"""
    <div class="{risk_class}">
        <div class="risk-card-title">{risk_icon} {risk_headline}</div>
        <div class="risk-card-desc">{risk_advice}</div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Display
    m1, m2, m3 = st.columns(3)
    with m1:
        diff_threshold = risk - threshold
        diff_str = f"{diff_threshold:+.1%} vs cutoff"
        st.metric(
            label="Predicted Risk Probability",
            value=f"{risk:.1%}",
            delta=diff_str,
            delta_color="inverse"
        )
    with m2:
        st.metric(
            label="Screening Decision Cutoff",
            value=f"{threshold:.1%}",
            help="Threshold optimized during model validation for high recall/sensitivity."
        )
    with m3:
        st.metric(
            label="Clinical Expert Rules Triggered",
            value=f"{len(reasons)} Active",
            help="Number of high-risk clinical interaction heuristics met."
        )

    # Risk Progress Bar
    st.markdown("**Screening Risk Gauge:**")
    st.progress(min(max(risk, 0.0), 1.0))
    st.caption(f"Scale: 0% (Minimal Risk) • **{threshold:.1%} Decision Cutoff** • 100% (Extreme Risk)")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed Analysis Tabs
    tab_rules, tab_recom, tab_benchmark = st.tabs([
        "📋 Clinical Findings & Rules",
        "💡 Tailored Action Plan",
        "📈 Model Benchmarks & Metrics"
    ])

    with tab_rules:
        st.markdown("#### Key Risk Factors & Expert System Detections")
        if reasons:
            st.write("The clinical heuristic engine identified the following high-risk co-morbid patterns:")
            for r in reasons:
                st.markdown(f'<div class="factor-pill">⚠️ {r}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No high-risk synergistic comorbid rule patterns were triggered for this patient profile.")

    with tab_recom:
        st.markdown("#### Personalized Clinical & Lifestyle Recommendations")
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.append(("🫀 Cardiology Referral", "Schedule a referral to a cardiologist for an in-depth workup (e.g., 12-lead ECG, echocardiogram, exercise stress test, coronary calcium score)."))
        elif risk_level == "MODERATE":
            recommendations.append(("🩺 Primary Care Review", "Schedule a comprehensive physical exam with your primary care provider within 30-60 days to reassess cardiovascular markers."))

        if high_bp == 1:
            recommendations.append(("🩺 Blood Pressure Management", "Engage in home blood pressure monitoring (morning and evening), reduce sodium intake (< 2,300 mg/day), and discuss antihypertensive therapy with a doctor."))

        if high_cholesterol == 1:
            recommendations.append(("🩸 Lipid Profile Optimization", "Request a fasting lipid panel (LDL, HDL, triglycerides). Discuss dietary adjustments (lowering saturated and trans fats) and potential statin therapy."))

        if diabetes in (1, 2):
            recommendations.append(("🔬 Glycemic Control", "Maintain tight blood sugar monitoring (target HbA1c < 7.0%). Diabetes significantly accelerates arterial atherosclerosis."))

        if smoking_status in (1, 2):
            recommendations.append(("🚭 Smoking Cessation Program", "Enroll in structured smoking cessation support or consult a physician regarding nicotine replacement therapies and bupropion/varenicline."))

        if bmi >= 30.0:
            recommendations.append(("🥗 Weight Optimization & Nutrition", "Work with a clinical dietitian on a heart-healthy dietary pattern (e.g., Mediterranean or DASH diet) targeting gradual 5-10% weight reduction."))

        if copd == 1 or difficulty_walking == 1:
            recommendations.append(("🚶 Cardiopulmonary Mobility", "Pursue physician-supervised light physical activity or pulmonary rehabilitation to safely increase aerobic capacity."))

        recommendations.append(("❤️ General Cardiovascular Wellness", "Prioritize 7-8 hours of quality sleep nightly, stress reduction techniques, and 150 minutes of moderate-intensity weekly physical activity when medically cleared."))

        for title, desc in recommendations:
            st.markdown(f"""
            <div class="recom-box">
                <strong>{title}</strong><br>
                <span style="opacity: 0.9; font-size: 0.93rem;">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab_benchmark:
        st.markdown("#### Model Performance Evaluation (CDC BRFSS Test Set)")
        
        if os.path.exists("model_comparison.csv"):
            try:
                comp_df = pd.read_csv("model_comparison.csv")
                st.dataframe(
                    comp_df.style.format({
                        "accuracy": "{:.3f}",
                        "recall": "{:.3f}",
                        "precision": "{:.3f}",
                        "f1": "{:.3f}",
                        "roc_auc": "{:.3f}",
                        "pr_auc": "{:.3f}"
                    }),
                    use_container_width=True
                )
                st.caption("Benchmark of tested classification algorithms on the stratifed holdout test set.")
            except Exception as e:
                st.info(f"Comparison table: {e}")

        # Show performance figures if available
        fig_roc = "Figures/roc_curve.png"
        fig_cm = "Figures/confusion_matrix.png"
        
        if os.path.exists(fig_roc) or os.path.exists(fig_cm):
            c_roc, c_cm = st.columns(2)
            with c_roc:
                if os.path.exists(fig_roc):
                    st.image(fig_roc, caption="Receiver Operating Characteristic (ROC-AUC: 0.849)", use_container_width=True)
            with c_cm:
                if os.path.exists(fig_cm):
                    st.image(fig_cm, caption="Confusion Matrix at Calibrated Threshold (0.54)", use_container_width=True)



# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("CardioScreen AI • Artificial Intelligence & Expert Systems (AI&ES) Project • Built with Streamlit, Scikit-Learn & XGBoost")
