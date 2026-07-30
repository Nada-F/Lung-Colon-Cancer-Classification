import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

model = tf.keras.models.load_model("lung_colon_cancer_model.keras")

with open("classes.json", "r") as f:
    class_names = json.load(f)

st.title("Lung & Colon Cancer Classification")

uploaded_file = st.file_uploader(
    "Upload Histopathology Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, 1)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(image, width=300)

    img = cv2.resize(image, (224,224))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    st.success(
        f"Prediction : {class_names[index]}"
    )

    st.write(
        f"Confidence : {prediction[0][index]*100:.2f}%"
    )