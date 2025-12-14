import streamlit as st
import os
import gdown
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain MRI image to detect tumor presence.")

# -----------------------------
# Google Drive model download
# -----------------------------
MODEL_PATH = "BRAINTUMOR.h5"
DRIVE_URL = "https://drive.google.com/uc?id=1ALXsWSNXUsrBDXA5v24McLRPu0ZHLuQ"

@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]

    st.write(f"### Confidence: `{prediction:.2f}`")

    if prediction >= 0.5:
        st.error("⚠️ Brain Tumor Detected")
    else:
        st.success("✅ No Brain Tumor Detected")
