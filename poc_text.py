# -*- coding: utf-8 -*-
"""Poc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TpCBT-dIPl7pTRG-zzMoY_9NqsuW0GUS
"""


import re

def preprocess_text(text):
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text



import streamlit as st
import pandas as pd
import pickle

# Load your trained model
classifier, vectorizer = pickle.load(open('https://drive.google.com/file/d/1tQNqnMIohzeUFFFn8QfpqyVhx_M4AedZ/view?usp=drive_link', 'rb'))

st.title('Text Classification Tool')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
if uploaded_file is not None:
    df_test = pd.read_excel(uploaded_file)
    X_test = df_test.v2
    Y_test = df_test.v1
    le = LabelEncoder()
    Y_test = le.fit_transform(Y_test)
    Y_test = Y_test.reshape(-1,1)
    X_text=[preprocess_text(text) for text in X_test]
    X_test_vect = vectorizer.transform(X_test)
    predictions = classifier.predict(X_test_vect)  # adjust the column name
    df_test['predictions'] = predictions
    st.write(df_test)







