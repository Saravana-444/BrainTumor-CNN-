import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(120deg, #0f0c29, #302b63, #24243e);
}

.glass {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.4);
    color: white;
}

.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 25px;
}

.metric-box {
    background: rgba(0,0,0,0.4);
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #aaaaaa;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 NeuroScan AI")
st.sidebar.markdown("**CNN-Powered Brain Tumor Detection System**")
st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Tech Stack**")
st.sidebar.markdown("- TensorFlow / Keras\n- Streamlit\n- Google Colab\n- GitHub")
st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developer**")
st.sidebar.markdown("Your Name Here")
st.sidebar.markdown("🎯 Final Year AI Project")

# -----------------------------
# Google Drive model download
# -----------------------------
MODEL_PATH = "BRAINTUMOR.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1ALXsWSNXUsrBDXA5v24McLRPu0ZHLuQm"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... Please wait ⏳"):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 NeuroScan AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced CNN-Based Brain Tumor Detection Dashboard</div>", unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["🧪 Detection", "📊 Model Info", "ℹ️ About"])

# ---------------- DETECTION TAB ----------------
with tab1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Uploaded MRI Scan", use_container_width=True)

        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        with st.spinner("🔍 Running CNN Inference..."):
            prediction = model.predict(img_array)[0][0]

        confidence = float(prediction)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-box'>📊 Confidence<br><h2>{:.2f}%</h2></div>".format(confidence * 100), unsafe_allow_html=True)
        with col2:
            status = "Tumor Detected" if confidence >= 0.5 else "No Tumor"
            st.markdown(f"<div class='metric-box'>🧠 Status<br><h2>{status}</h2></div>", unsafe_allow_html=True)
        with col3:
            risk = "High Risk" if confidence >= 0.75 else "Low Risk"
            st.markdown(f"<div class='metric-box'>⚠️ Risk Level<br><h2>{risk}</h2></div>", unsafe_allow_html=True)

        st.progress(min(max(confidence, 0.0), 1.0))

        if confidence >= 0.5:
            st.error("🚨 **Brain Tumor Detected — Please consult a medical professional**")
        else:
            st.success("✅ **No Tumor Detected — MRI appears normal**")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- MODEL INFO TAB ----------------
with tab2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("""
    ### 📊 CNN Model Information
    - Architecture: Convolutional Neural Network (CNN)
    - Input Size: 224 x 224 RGB
    - Output: Binary Classification (Tumor / No Tumor)
    - Training Platform: Google Colab
    - Dataset: Public MRI Brain Tumor Dataset
    - Deployment: Streamlit Web App
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT TAB ----------------
with tab3:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown("""
    ### ℹ️ About This Project
    **NeuroScan AI** is a deep learning-based medical imaging system designed to assist
    in detecting brain tumors from MRI scans using a trained CNN model.

    ⚠️ This tool is for educational and research purposes only. It is not a medical diagnosis system.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("<div class='footer'>© 2026 NeuroScan AI • Powered by Deep Learning</div>", unsafe_allow_html=True)
