
import streamlit as st
import pandas as pd
import pickle
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sales Predictor Dashboard",
    page_icon="💰",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_model(model_path):
    """Loads the saved regression model from a .pkl file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None

MODEL_FILENAME = 'model-reg-67130701915.pkl'
loaded_model = load_model(MODEL_FILENAME)

# --- APP HEADER ---
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💰 Sales Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict estimated sales based on your social media advertising budgets.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- MODEL VALIDATION ---
if loaded_model is None:
    st.error(
        f"⚠️ Model file `{MODEL_FILENAME}` not found. "
        "Please place it in the same directory and reload the page."
    )
else:
    # --- USER INPUT FORM ---
    with st.form("prediction_form"):
        st.subheader("📊 Input Advertising Budgets (USD)")

        col1, col2, col3 = st.columns(3)

        with col1:
            youtube_budget = st.number_input("🎬 YouTube", min_value=0, value=150, step=10)
        with col2:
            tiktok_budget = st.number_input("🎵 TikTok", min_value=0, value=40, step=5)
        with col3:
            instagram_budget = st.number_input("📸 Instagram", min_value=0, value=60, step=5)

        submitted = st.form_submit_button("🔮 Predict Sales")

    # --- PREDICTION ---
    if submitted:
        new_data = pd.DataFrame({
            'youtube': [youtube_budget],
            'tiktok': [tiktok_budget],
            'instagram': [instagram_budget]
        })

        st.markdown("---")
        st.subheader("📈 Prediction Result")

        try:
            prediction = loaded_model.predict(new_data)
            estimated_sales = prediction[0]

            st.metric(
                label="Estimated Total Sales",
                value=f"${estimated_sales:,.2f} K",
                delta=None,
                help="Predicted sales value based on the entered advertising budgets."
            )

            st.success("✅ Prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# --- SIDEBAR ---
st.sidebar.header("ℹ️ About This App")
st.sidebar.info(
    """
    This dashboard uses a **Linear Regression model**  
    trained from historical advertising data to estimate total sales.  
    Adjust the budgets for YouTube, TikTok, and Instagram  
    to see how they impact the predicted result.
    """
)
