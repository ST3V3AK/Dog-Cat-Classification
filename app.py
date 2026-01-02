import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from utils import preprocess_image

img_size = (180, 180)
st.set_page_config(page_title="Cat and Dog Classification", layout="centered")
st.title("🐱🐶 Cat vs Dog Classification (Custom TFLite Model)")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="model/dog_cat_model.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload an Image of a cat or dog", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded image", use_container_width=True)

    input = preprocess_image(img_np)

    interpreter.set_tensor(
        input_details[0]['index'],
        input
    )
    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )

     # Handle sigmoid or softmax output
    if output.shape[-1] == 1:
        prob_dog = output[0][0]
        prob_cat = 1 - prob_dog
    else:
        prob_cat, prob_dog = output[0]

    label = "Dog 🐶" if prob_dog > prob_cat else "Cat 🐱"
    confidence = max(prob_cat, prob_dog)

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    st.bar_chart({
        "Cat": float(prob_cat),
        "Dog": float(prob_dog)
    })
