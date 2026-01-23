import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.25);
    margin-top: 20px;
}

.title {
    text-align: center;
    color: #ffffff;
    font-size: 36px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #dddddd;
    margin-bottom: 20px;
}

.footer {
    text-align: center;
    color: #cccccc;
    font-size: 13px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 Brain Tumor Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a brain MRI image and let CNN analyze it</div>", unsafe_allow_html=True)

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

    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# ---------------- CARD START ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🖼️ Uploaded MRI Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    with st.spinner("🔍 Analyzing MRI scan..."):
        prediction = model.predict(img_array)[0][0]

    confidence = float(prediction)
    st.markdown(f"### 📊 Confidence Score")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.write(f"**{confidence * 100:.2f}%**")

    if confidence >= 0.5:
        st.error("⚠️ **Brain Tumor Detected**")
    else:
        st.success("✅ **No Brain Tumor Detected**")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>Powered by CNN • TensorFlow • Streamlit • Google Colab • GitHub</div>",
    unsafe_allow_html=True
)
