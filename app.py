import streamlit as st
import os
import urllib.request
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

st.title("🧠 Brain Tumor Detection")

MODEL_URL = "https://github.com/Saravana-444/BrainTumor-CNN-/releases/download/v1.0/BRAINTUMOR.h5"
MODEL_PATH = "BRAINTUMOR.h5"

@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return load_model(MODEL_PATH)

model = load_model_safe()

file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img = img.resize((224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0

    pred = model.predict(arr)[0][0]

    if pred >= 0.5:
        st.error("Brain Tumor Detected")
    else:
        st.success("No Brain Tumor Detected")
