# -*- coding: utf-8 -*-
"""Poc.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TpCBT-dIPl7pTRG-zzMoY_9NqsuW0GUS
"""
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import LabelEncoder



import re

def preprocess_text(text):
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'\d+', '', text)

import streamlit as st
import pandas as pd
import pickle

# Load your trained model
classifier, vectorizer = pickle.load(open('text_classifier.pkl', 'rb'))

st.title('Text Classification Tool')

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
if uploaded_file is not None:
    df_test = pd.read_excel(uploaded_file)
    X_test = df_test.text
    # Sample text data
    text_data = X_test
    X_test=X_test.apply(clean_text)
    #X_text=[clean_text(text) for text in X_test]
    X_test_vect = vectorizer.transform(X_test)
    predictions = classifier.predict(X_test_vect)  # adjust the column name
    # Custom labels dictionary
    label_map = {0: 'Recharge carte prépayée non aboutie',
                 1: 'Retard d exécution d un ordre virement (normal, cih on line)',
                 2: 'Code PIN non reçu', 3: 'Manque solde sur le compte',
                 4: 'Non réception OTP (Activation, transfert ou recharge)',
                 5: 'Autre'} 
    transformed_labels = np.vectorize(label_map.get)(predictions)
    df_test['nature_prediction'] = transformed_labels
    st.write(df_test)







