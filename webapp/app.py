import streamlit as st
from PIL import Image

image = "/assets/url_genie_logo.png"

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image)

with col3:
    st.write(' ')
st.title("Malicious Domain Name Detection Model made using Python")
st.text_input("Enter URL to scan", "https://google.com")