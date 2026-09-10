import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Card container styling */
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Primary Headers */
    .main-title {
        color: #1E293B;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Prediction Outcome Cards */
    .approved-card {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 20px;
        border-radius: 8px;
        color: #065F46;
    }

    .rejected-card {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 20px;
        border-radius: 8px;
        color: #991B1B;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loading
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("logistic_regression_model.pkl")


try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Could not load model (`logistic_regression_model.pkl`). Details: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar Header & Info
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/isometric-folders/100/money-transfer.png",
        width=80,
    )
    st.title("Loan Decision Portal")
    st.markdown(
        "Use this intelligent portal to assess loan application risks in real time."
    )
    st.divider()
    st.caption("⚙️ **Model Type:** Logistic Regression")
    st.caption("📊 **Dataset Features:** Income, Credit Score, Employment, Debt Ratio")

# ---------------------------------------------------------
# Main UI Elements
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🏦 Automated Loan Approval Risk Estimator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Adjust the parameters on the left or below to evaluate candidate eligibility instantly.</div>',
    unsafe_allow_html=True,
)

# Input Layout using Columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 👤 Applicant Details")
    income = st.number_input(
        "Annual Income ($)",
        min_value=0,
        max_value=2_000_000,
        value=500_000,
        step=25_000,
        help="Applicant's total annual gross income.",
    )

    employment_years = st.slider(
        "Employment Duration (Years)",
        min_value=0,
        max_value=40,
        value=5,
        help="Number of consecutive years in current field or employment.",
    )

with col2:
    st.markdown("### 📊 Financial Indicators")
    credit_score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=680,
        help="FICO or equivalent credit score rating.",
    )

    debt_ratio = st.slider(
        "Debt-to-Income (DTI) Ratio",
        min_value=0.00,
        max_value=1.00,
        value=0.35,
        step=0.01,
        help="Monthly debt payments divided by monthly gross income.",
    )

st.divider()

# ---------------------------------------------------------
# Prediction Trigger & Logic
# ---------------------------------------------------------
predict_btn = st.button("🚀 Evaluate Application", type="primary", use_container_width=True)

if predict_btn:
    # Prepare DataFrame matching exact column order/names
    input_df = pd.DataFrame(
        [[income, credit_score, employment_years, debt_ratio]],
        columns=["income", "credit_score", "employment_years", "debt_ratio"],
    )

    # Perform Prediction
    prediction = model.predict(input_df)[0]

    # Calculate Probabilities if supported by model
    has_proba = hasattr(model, "predict_proba")
    if has_proba:
        probabilities = model.predict_proba(input_df)[0]
        prob_approved = float(probabilities[1])
    else:
        prob_approved = 1.0 if prediction == 1 else 0.0

    st.markdown("### 🎯 Decision Results")

    res_col1, res_col2 = st.columns([1, 1], gap="medium")

    with res_col1:
        if prediction == 1:
            st.markdown(
                f"""
                <div class="approved-card">
                    <h2 style="margin:0; font-size: 1.6rem;">🎉 Application Approved</h2>
                    <p style="margin-top: 8px; font-size: 1rem;">The applicant meets the required score thresholds for funding.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="rejected-card">
                    <h2 style="margin:0; font-size: 1.6rem;">❌ Application Declined</h2>
                    <p style="margin-top: 8px; font-size: 1rem;">The applicant does not satisfy the default safety threshold.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with res_col2:
        st.markdown("**Approval Probability Confidence**")
        st.progress(prob_approved)
        st.metric(
            label="Approval Confidence Score",
            value=f"{prob_approved * 100:.1f}%",
            delta="High Eligibility" if prob_approved >= 0.5 else "High Risk",
            delta_color="normal" if prob_approved >= 0.5 else "inverse",
        )