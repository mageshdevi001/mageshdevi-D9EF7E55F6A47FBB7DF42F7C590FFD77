import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
from tensorflow.keras.models import load_model

# Load model and class map
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_my_model():
    return load_model('fake_product_detection_model.h5', compile=False)

model = load_my_model()

# Load class labels
with open('class_map.json', 'r') as f:
    class_map = json.load(f)
class_labels = {v: k for k, v in class_map.items()}  # Reverse mapping

# Preprocess function
def preprocess_image(image, image_size=(224, 224)):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure correct color format
        img = cv2.resize(img, image_size)
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit UI
st.title("🛍️ Fake Product Detection")
st.write("Upload an image to classify it as **real** or **fake**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Convert file to OpenCV format
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process and predict
        processed_image = preprocess_image(image)
        if processed_image is not None:
            prediction = model.predict(processed_image)
            predicted_class = np.argmax(prediction)
            class_name = class_labels.get(predicted_class, "Unknown")

            st.success(f"**Prediction: {class_name}**")
    else:
        st.error("Could not process the image. Try uploading a different file.")
