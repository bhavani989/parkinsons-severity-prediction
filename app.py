import streamlit as st
import numpy as np
import joblib

# ============================================
# Load model files
# ============================================
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
model = joblib.load("svm_model.joblib")
feature_names = joblib.load("feature_names.joblib")

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="Parkinson's Severity Predictor",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# Custom CSS Styling
# ============================================
st.markdown("""
<style>
body {
    background-color: #F7F9F9;
}
.big-title {
    font-size: 46px;
    font-weight: 900;
    color: #1A5276;
    text-align: center;
}
.sub-title {
    font-size: 20px;
    color: #5D6D7E;
    text-align: center;
    margin-bottom: 20px;
}
.card {
    background-color: #FFFFFF;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
}
.footer {
    font-size: 14px;
    text-align: center;
    color: #85929E;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# Header
# ============================================
st.markdown("<div class='big-title'>🧠 Parkinson's Disease Severity Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Elegant, User-Friendly Prediction • PCA + SVM Model • UCI Telemonitoring Dataset</div>", unsafe_allow_html=True)

# ============================================
# About Section
# ============================================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📘 About This Prediction Tool")
    st.write("""
    This web application predicts the **UPDRS severity score**, a clinical measure of Parkinson's Disease progression.  
    Instead of asking for complex speech biomarker values (Jitter, Shimmer, NHR, RPDE), we provide **simple sliders** that map to **medically realistic values** internally.
    
    This makes the tool easy and friendly to use while keeping the backend scientifically accurate.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ============================================
# Input Section
# ============================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Patient Information")

col1, col2 = st.columns(2)

age = col1.number_input(
    "👤 Age",
    min_value=20,
    max_value=90,
    value=60,
    help="Patient's age. Older age can correlate with higher severity."
)

sex = col2.selectbox(
    "⚧ Sex",
    options=[("Male", 1), ("Female", 0)],
    format_func=lambda x: x[0],
    help="Biological sex (numeric encoding used for the model)."
)[1]

st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# Speech + Tremor Sliders
# ============================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎤 Speech & Tremor Characteristics")

exp = st.expander("ℹ️ What do these sliders mean?")
exp.write("""
**Tremor Severity:** Higher values represent increased hand/body tremors.  
**Voice Clarity:** Low clarity (0–3) indicates hoarseness/roughness; higher clarity (7–10) suggests clear speech.  
**Speech Stability:** Measures voice stability during vowel sounds.  
**Distortion Level (Jitter/Shimmer):** Represents vibration inconsistencies in speech. Higher = worse distortion.
""")

col3, col4 = st.columns(2)

tremor = col3.slider("🤲 Tremor Severity", 0, 10, 5)
voice_clarity = col3.slider("🎙️ Voice Clarity", 0, 10, 5)
speech_stability = col4.slider("🗣️ Speech Stability", 0, 10, 5)
distortion = col4.slider("📉 Distortion Level (Jitter/Shimmer)", 0, 10, 5)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# Feature Mapping (Medically Realistic Conversion)
# ============================================
def create_medical_feature_vector():
    """
    Convert simple sliders into realistic scientific biomarker values
    in the ranges used in the UCI dataset.
    """
    features = np.zeros(len(feature_names))

    for i, name in enumerate(feature_names):

        # Map simple values → realistic scientific ranges
        if "age" in name.lower():
            features[i] = age

        elif "sex" in name.lower():
            features[i] = sex

        # --- JITTER RANGE: 0.0001 – 0.01 ---
        elif "jitter" in name.lower():
            features[i] = np.interp(distortion, [0, 10], [0.0001, 0.010])

        # --- SHIMMER RANGE: 0.01 – 0.2 ---
        elif "shimmer" in name.lower():
            features[i] = np.interp(distortion, [0, 10], [0.01, 0.2])

        # --- NHR RANGE: 0 – 0.3 ---
        elif "NHR" in name:
            features[i] = np.interp(10 - voice_clarity, [0, 10], [0.01, 0.30])

        # --- HNR RANGE: 5 – 35 ---
        elif "HNR" in name:
            features[i] = np.interp(voice_clarity, [0, 10], [5, 35])

        # --- RPDE RANGE: 0.3 – 0.65 ---
        elif "RPDE" in name:
            features[i] = np.interp(10 - speech_stability, [0, 10], [0.30, 0.65])

        # --- DFA RANGE: 0.5 – 1.3 ---
        elif "DFA" in name:
            features[i] = np.interp(10 - tremor, [0, 10], [0.50, 1.30])

        # --- PPE RANGE: 0.02 – 0.3 ---
        elif "PPE" in name:
            features[i] = np.interp(distortion + (10 - speech_stability), [0, 20], [0.02, 0.30])

        # DEFAULT (safe small variation)
        else:
            features[i] = np.interp(distortion, [0, 10], [0.1, 1.0])

    return features.reshape(1, -1)


# ============================================
# Prediction Section
# ============================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Predict Parkinson's Severity")

if st.button("🧠 Get Prediction", use_container_width=True):
    features = create_medical_feature_vector()
    scaled = scaler.transform(features)
    transformed = pca.transform(scaled)
    prediction = model.predict(transformed)[0]

    st.success(f"### 🟢 Predicted UPDRS Severity: **{prediction:.2f}**")
    st.write("Higher values indicate more severe Parkinson's symptoms.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# Footer
# ============================================
st.markdown("<div class='footer'>Built with ❤️ for Parkinson's Awareness • PCA + SVM Model</div>", unsafe_allow_html=True)
