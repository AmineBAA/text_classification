import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

# Load the model (adjust the path to your model file)
model = load_model('text_classification_model.h5')

# Load the label encoder (adjust the path to your label encoder file)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the TF-IDF vectorizer (adjust the path to your vectorizer file)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Function to preprocess and predict text
def predict(texts):
    # Transform text to TF-IDF features
    features = tfidf_vectorizer.transform(texts)
    # Predict classes
    predictions = model.predict(features)
    # Convert predictions to class labels
    predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    return predicted_classes

# Streamlit app
st.title("Text Classification with Uploaded Excel File")

# Upload Excel file
uploaded_file = st.file_uploader("Choose an Excel file...", type=["xlsx"])

if uploaded_file is not None:
    # Load the uploaded file
    df = pd.read_excel(uploaded_file)
    
    # Ensure the column containing text is named 'text'
    if 'text' not in df.columns:
        st.error("Excel file must contain a column named 'text'")
    else:
        # Extract text data
        texts = df['text'].astype(str)
        
        # Display the first few rows
        st.write("First few rows of the text data:")
        st.write(texts.head())
        
        # Predict classes
        predictions = predict(texts)
        
        # Add predictions to the DataFrame
        df['predicted_class'] = predictions
        
        # Display results
        st.write("Predictions:")
        st.write(df)
        
        # Option to download the result as a new Excel file
        result_file = 'predictions.xlsx'
        df.to_excel(result_file, index=False)
        st.download_button(label="Download Predictions", data=open(result_file, 'rb').read(), file_name=result_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
