import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

from tensorflow.keras.applications.efficientnet import preprocess_input


model = tf.keras.models.load_model(
    "lung_colon_efficientnet_final.keras"
)

with open("classes.json", "r") as f:
    class_names = json.load(f)


st.set_page_config(
    page_title="Cancer Classification",
    layout="centered"
)

st.title(" Lung & Colon Cancer Classification")

st.write(
    "Upload a Histopathology Image"
)

uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = cv2.resize(
        image,
        (224,224)
    )

    img = img.astype(np.float32)

    img = preprocess_input(img)

    img = np.expand_dims(
        img,
        axis=0
    )

    prediction = model.predict(
        img,
        verbose=0
    )

    index = np.argmax(prediction)

    confidence = float(
        prediction[0][index]
    )

    st.subheader("Prediction")

    st.success(
        class_names[index]
    )

    st.subheader("Confidence")

    st.write(
        f"{confidence*100:.2f}%"
    )

    
    st.subheader("All Classes")

    for i, cls in enumerate(class_names):

        st.write(
            f"{cls} : {prediction[0][i]*100:.2f}%"
        )

    
    if confidence < 0.80:

        st.warning(
            " Low confidence. The uploaded image may be different from the training dataset."
        )