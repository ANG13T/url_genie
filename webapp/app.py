import streamlit as st
import tensorflow as tf
from tensorflow import keras

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('Malicious_URL_Prediction.h5')
    return model
with st.spinner("Loading Model...."):
    model=load_model()

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("https://github.com/ANG13T/url_genie/blob/main/webapp/assets/url_genie_logo.png?raw=true")

with col3:
    st.write(' ')

st.markdown("<h1 style='text-align: center; color: #14559E'>URL Genie</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #494848;'>Malicious URL Detection Model made using Python</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #494848;'>This program utilizes a Multilayer Perceptron Neural Network model with optimized hyper-parameters using genetic algorithms to perform malicous URL detection</p>", unsafe_allow_html=True)

def predict(val):
    st.write("Predicting Class...")
    with st.spinner("Classifying..."):
        pred_test = model.predict(val)
        st.write(pred_test)


value = st.text_input("Enter URL to scan", "https://google.com")
submit = st.button("Classify URL", on_click=predict(value))
