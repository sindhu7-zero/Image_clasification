"""
Streamlit app for multiclass fish image classification.
Loads model and class names saved by train.py.
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Cache model and class names to avoid reloading on every interaction
@st.cache_resource
def load_model_and_classes():
    """Load trained model and class labels from disk."""
    try:
        model = tf.keras.models.load_model("models/fish_classifier.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    try:
        with open("models/class_names.txt", "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("Class names file not found. Please run train.py first.")
        st.stop()

    return model, class_names

# Load resources once
model, class_names = load_model_and_classes()

# Page config
st.set_page_config(
    page_title="🐟 Fish Classifier",
    page_icon="🐟",
    layout="centered"
)

# App header
st.title("🐟 Multiclass Fish Image Classifier")
st.write("Upload a fish image to classify its species.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    try:
        # Open and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image to model input size
        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        with st.spinner("Classifying..."):
            predictions = model.predict(img_array)
        
        # Get top prediction
        confidence = float(np.max(predictions))
        predicted_class = class_names[int(np.argmax(predictions))]

        # Display result
        st.success(f"**Prediction**: {predicted_class}")
        st.info(f"**Confidence**: {confidence:.2%}")

    except Exception as e:
        st.error(f"Error processing image: {e}")
else:
    st.info("👆 Please upload a fish image to begin.")