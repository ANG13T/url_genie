import streamlit as st
from PIL import Image

image = "https://github.com/ANG13T/url_genie/blob/main/webapp/assets/url_genie_logo.png?raw=true"

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image(image)

with col3:
    st.write(' ')

st.markdown("<h1 style='text-align: center; color: #14559E'>URL Genie</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #494848;'>Malicious URL Detection Model made using Python</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #494848;'>Malicious URL Detection Model made using Python</p>", unsafe_allow_html=True)
st.text_input("Enter URL to scan", "https://google.com")
