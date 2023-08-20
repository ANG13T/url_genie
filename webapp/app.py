import streamlit as st
from tensorflow import keras
from urllib.parse import urlparse
import numpy as np
import re

def load_model():
    model=keras.models.load_model('Malicious_URL_Prediction.h5')
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

def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
    
def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')

def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1

def extract_features(url):
    # 'hostname_length', 'path_length', 'fd_length', 'count-', 'count@', 'count?', 'count%', 'count.', 'count=', 'count-http','count-https', 'count-www', 'count-digits','count-letters', 'count_dir', 'use_of_ip'
    hostname_length = len(urlparse(url).netloc)
    path_length = len(urlparse(url).path)
    f_length = fd_length(url)
    count_1 = url.count('-')
    count_2 = url.count('@')
    count_3 = url.count('?')
    count_4 = url.count('%')
    count_5 = url.count('.')
    count_6 = url.count('=')
    count_7 = url.count('http')
    count_8 = url.count('https')
    count_9 = url.count('www')
    count_10 = digit_count(url)
    count_11 = letter_count(url)
    count_12 = no_of_dir(url)
    count_13 = having_ip_address(url)
    output = [hostname_length, path_length, f_length, count_1, count_2, count_3, count_4, count_5, count_6, count_7, count_8, count_9, count_10, count_11, count_12, count_13]
    print(output)
    features = np.array([output]) 
    return features

def predict(val):
    st.write(f'Classifying URL: {val}')
    with st.spinner("Classifying..."):
        input = extract_features(val)
        print(input.shape)
        for item in input:
            print(type(item))
        pred_test = model.predict(input)
        percentage_value = pred_test[0][0] * 100
        if (pred_test[0] < 0.5):
            st.write(f'<span style="color:green;">✅ **SAFE with {percentage_value:.2f}% malicious confidence**</span>', unsafe_allow_html=True)
        else: 
            st.write(f'<span style="color:red;">⛔️ **MALICOUS with {percentage_value:.2f}% malicious confidence**</span>', unsafe_allow_html=True)
        print(input, pred_test)


value = st.text_input("Enter URL to scan", "https://www.google.com")
submit = st.button("Classify URL")

if submit:
    predict(value)
