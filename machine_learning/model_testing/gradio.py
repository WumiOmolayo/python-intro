"""
Loan Approval Predictor - Gradio App
Loads the Logistic Regression model saved by the notebook as
'logistic_regression_model.pkl' and predicts approve/reject + probability.
"""

import pickle
import pandas as pd
import gradio as gr

# Load the trained Logistic Regression model (saved by the notebook)
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_loan_approval(income, credit_score, employment_years, debt_ratio):
    """Run the applicant's details through the trained model and format the result."""

    # Must match the FEATURES order the model was trained on
    applicant = pd.DataFrame({
        'income':           [income],
        'credit_score':     [credit_score],
        'employment_years': [employment_years],
        'debt_ratio':       [debt_ratio],
    })

    probability = model.predict_proba(applicant)[:, 1][0]   # probability of "approved"
    decision = model.predict(applicant)[0]

    if decision == 1:
        status, color, bg = "✅ APPROVED", "#16a34a", "#f0fdf4"
    else:
        status, color, bg = "❌ REJECTED", "#dc2626", "#fef2f2"

    if probability >= 0.7:
        note = "High confidence approval."
    elif probability >= 0.5:
        note = "Approved, but on the borderline - consider a manual review."
    elif probability >= 0.3:
        note = "Rejected, but not by a wide margin - consider a manual review."
    else:
        note = "Low approval probability."

    bar_width = round(probability * 100)

    return f"""
    <div style="background:{bg}; border:2px solid {color}; border-radius:14px;
                padding:24px; font-family:sans-serif;">
        <h2 style="color:{color}; margin:0 0 10px 0;">{status}</h2>
        <p style="margin:4px 0 6px 0; color:#374151;">Probability of approval</p>
        <div style="background:#e5e7eb; border-radius:8px; height:24px; width:100%;
                    overflow:hidden;">
            <div style="background:{color}; height:100%; width:{bar_width}%;
                        text-align:right; color:white; font-size:12px;
                        line-height:24px; padding-right:8px; transition:width 0.4s;">
                {probability:.0%}
            </div>
        </div>
        <p style="margin-top:16px; color:#374151;">{note}</p>
    </div>
    """


with gr.Blocks(title="Loan Approval Predictor", theme=gr.themes.Soft(primary_hue="blue")) as demo:

    gr.Markdown("# 🏦 Loan Approval Predictor")
    gr.Markdown("Enter applicant details to predict loan approval, using a **Logistic Regression** model.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Applicant Details")

            income = gr.Number(
                label="Monthly Income (NGN)",
                value=400000,
                minimum=0
            )
            credit_score = gr.Slider(
                label="Credit Score",
                minimum=300, maximum=850, value=650, step=1
            )
            employment_years = gr.Slider(
                label="Years of Employment",
                minimum=0, maximum=40, value=3, step=1
            )
            debt_ratio = gr.Slider(
                label="Debt-to-Income Ratio",
                minimum=0.0, maximum=1.0, value=0.35, step=0.01
            )

            predict_btn = gr.Button("Predict Approval", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### Result")
            output = gr.HTML()

    predict_btn.click(
        fn=predict_loan_approval,
        inputs=[income, credit_score, employment_years, debt_ratio],
        outputs=output
    )

    gr.Markdown("""
    ---
    **Model:** Logistic Regression, trained on historical loan approval data.
    **Features used:** income, credit_score, employment_years, debt_ratio.
    """)

if __name__ == "__main__":
    demo.launch(share= True)