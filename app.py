import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --------------------------------------------------
# App Title
# --------------------------------------------------
st.set_page_config(page_title="Biryani Detector", page_icon="🥘")
st.title("Biryani vs Pulao Detector 🥘")
st.write("Debug Status: App Initialized.")

# --------------------------------------------------
# Cache model loading
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("biryani_pulao_model.keras")

# --------------------------------------------------
# File uploader
# --------------------------------------------------
file = st.file_uploader(
    "UPLOAD IMAGE HERE",
    type=["jpg", "jpeg", "png"]
)

if file is not None:
    # 1. Show the image
    image = Image.open(file).convert("RGB") # Good fix! Handles PNG transparency
    st.image(image, caption="Uploaded Image", width=300)
    
    st.write("Analyzing...")

    # 2. Check model existence
    model_name = "biryani_pulao_model.keras"
    if not os.path.exists(model_name):
        st.error(f"CRITICAL ERROR: '{model_name}' not found.")
        st.stop()

    try:
        model = load_model()
        
        # 3. Preprocessing
        img = image.resize((160, 160))
        img_array = tf.keras.utils.img_to_array(img)
        
        # 🛑 DELETE THIS LINE! The model does it internally.
        # img_array = img_array / 255.0  <-- BAD! 
        
        img_array = tf.expand_dims(img_array, axis=0)

        # 4. Prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Ensure this order matches your folders (Alphabetical)
        class_names = ["Biryani", "Pulao"] 
        winner = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        st.divider()
        if winner == "Biryani":
            st.success(f"## 🥘 It's BIRYANI! ({confidence:.1f}%)")
            st.balloons()
        else:
            st.info(f"## 🥣 It's PULAO. ({confidence:.1f}%)")

    except Exception as e:
        st.error("Model crashed during prediction.")
        st.exception(e)
