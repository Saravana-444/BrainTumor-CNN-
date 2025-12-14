import streamlit as st
import os
import urllib.request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload a brain MRI image to detect tumor presence.")

# -----------------------------
# Model download settings
# -----------------------------
MODEL_URL = "https://drive.google.com/file/d/1ALXsWSNXUsrBDXA5v24McLRPu0ZHLuQm/view?usp=drive_link"
MODEL_PATH = "BRAINTUMOR.h5"

# -----------------------------
# Download model if not exists
# -----------------------------
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.write(f"### Prediction Confidence: `{prediction:.2f}`")

    if prediction >= 0.5:
        st.error("⚠️ **Brain Tumor Detected**")
    else:
        st.success("✅ **No Brain Tumor Detected**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("CNN Model | TensorFlow | Streamlit")
