import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from PIL import Image

# Load trained objects
dt_model = joblib.load("decision_tree_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Decision Tree Image Classifier")

st.title("🌿 Rice Image Classification using Decision Tree")
st.write("HOG feature extraction + Decision Tree model")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 🔴 MUST MATCH TRAINING PREPROCESSING 🔴
    img = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # HOG feature extraction (same as training)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # Scale features
    features = scaler.transform([features])

    # Prediction
    prediction = dt_model.predict(features)
    class_name = le.inverse_transform(prediction)

    st.success(f"✅ Predicted Class: **{class_name[0]}**")
