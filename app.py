import streamlit as st
import numpy as np
import joblib

# ============================================
# Load Model Files
# ============================================
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca.joblib")
model = joblib.load("svm_model.joblib")
feature_names = joblib.load("feature_names.joblib")

# ============================================
# PAGE CONFIG & STYLING
# ============================================
st.set_page_config(page_title="Parkinson's Severity Predictor", layout="wide", page_icon="🧠")

st.markdown("""
<style>
body { background-color: #F7F9F9; }
.big-title { font-size: 42px; font-weight: 900; color: #1A5276; text-align:center; }
.sub-title { font-size: 18px; color:#5D6D7E; text-align:center; margin-bottom:20px; }
.card {
    background-color:white;
    padding:20px; border-radius:16px;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
}
.footer { text-align:center; font-size:14px; color:#85929E; margin-top:40px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🧠 Parkinson's Disease Severity Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Medical-Realistic Synthetic Feature Engine • PCA + SVM Model</div>", unsafe_allow_html=True)

# ============================================
# USER INPUTS
# ============================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Patient Details")

c1, c2 = st.columns(2)

age = c1.number_input("👤 Age", 20, 90, 60)
sex = c2.selectbox("⚧ Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🎤 Speech & Tremor Characteristics")

exp = st.expander("ℹ️ What do these sliders represent?")
exp.write("""
### Tremor Severity
Indicates hand/limb tremors.

### Voice Clarity
Higher clarity = clearer speech.

### Speech Stability
Measures smoothness of vocal vibrations.

### Distortion (Jitter/Shimmer Level)
Represents how irregular the speech signal is.
""")

col3, col4 = st.columns(2)
tremor = col3.slider("🤲 Tremor Severity", 0, 10, 5)
clarity = col3.slider("🎙️ Voice Clarity", 0, 10, 5)
stability = col4.slider("🗣️ Speech Stability", 0, 10, 5)
distortion = col4.slider("📉 Distortion Level (Jitter/Shimmer)", 0, 10, 5)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================
# REALISTIC FEATURE GENERATOR
# ============================================

def realistic_value(mean, std, min_val, max_val):
    """Generate normally distributed values clipped to real medical ranges."""
    val = np.random.normal(mean, std)
    return float(np.clip(val, min_val, max_val))


def create_medical_feature_vector():
    """
    Convert sliders → MEDICAL REALISTIC acoustic features
    using UCI dataset ranges + natural correlations.
    """
    features = np.zeros(len(feature_names))

    for i, f in enumerate(feature_names):

        # Demographic
        if "age" in f.lower():
            features[i] = age
        elif "sex" in f.lower():
            features[i] = sex

        # Jitter — correlated with distortion
        elif "jitter" in f.lower():
            base = realistic_value(0.003, 0.002, 0.0001, 0.010)
            features[i] = base + distortion * 0.0005

        # Shimmer — more sensitive to distortion
        elif "shimmer" in f.lower():
            base = realistic_value(0.04, 0.015, 0.01, 0.2)
            features[i] = base + distortion * 0.003

        # NHR — inverse of clarity
        elif "NHR" in f:
            features[i] = realistic_value(0.10, 0.05, 0.0, 0.30) + (10 - clarity) * 0.01

        # HNR — proportional to clarity
        elif "HNR" in f:
            features[i] = realistic_value(20, 7, 5, 35) + clarity * 0.5

        # RPDE — non-linear: instability increases entropy
        elif "RPDE" in f:
            features[i] = realistic_value(0.45, 0.07, 0.30, 0.65) + (10 - stability) * 0.005

        # DFA — tremor introduces nonlinearity
        elif "DFA" in f:
            features[i] = realistic_value(0.9, 0.15, 0.50, 1.30) - tremor * 0.01

        # PPE — strongly related to distortion + stability
        elif "PPE" in f:
            features[i] = realistic_value(0.1, 0.05, 0.02, 0.30) + (
                distortion * 0.02 + (10 - stability) * 0.02
            )

        # Safe default for unused features
        else:
            features[i] = realistic_value(0.2, 0.1, 0.05, 1.0)

    return features.reshape(1, -1)

# ============================================
# PREDICTION BUTTON
# ============================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Predict Parkinson's Severity")

if st.button("🧠 Predict UPDRS Severity", use_container_width=True):
    x = create_medical_feature_vector()
    x_scaled = scaler.transform(x)
    x_pca = pca.transform(x_scaled)
    pred = model.predict(x_pca)[0]

    st.success(f"### 🟢 Predicted UPDRS Severity: **{pred:.2f}**")
    st.write("Higher scores indicate more severe Parkinson’s Disease.")

st.markdown("</div>", unsafe_allow_html=True)

