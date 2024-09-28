import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Assuming the model is saved in 'Model/Eye_Cataract_VGG16_model.h5'
model = load_model("Model/Eye_Cataract_VGG16_model.h5")


def welcome():
    return "Welcome All"

def predict_cataract(image_path):
    image = image_path
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.reshape(1, 224, 224, 3)
    result = model.predict(image)
    return result


def main():
    st.title("Cataract Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cataract Detection</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    image_file = st.file_uploader("Upload Cataract Image", type=['jpg', 'png'])
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, use_column_width=True)
        result = predict_cataract(image)
        if result[0][0] == 1:
            st.success('Cataract Not Detected')
        else:
            st.success('Cataract Detected')


if __name__ == '__main__':
    main()