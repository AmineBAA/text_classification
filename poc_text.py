import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
import re

# Load the classifier and vectorizer from .pkl files
with open('text_classifier.pkl', 'rb') as f:
    classifier, vectorizer  = pickle.load(f)


# Function to preprocess and predict text
def predict(texts):
    # Transform text to TF-IDF features using the vectorizer
    features = vectorizer.transform(texts)
    # Predict classes using the classifier
    predictions = classifier.predict(features)
    return predictions
import re

def preprocess_text(text):
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
    
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
        X = df['text'].astype(str)
        X = [str(text) for text in X]
        X=[text.replace("cart ","carte") for text in X]
        X=[preprocess_text(text) for text in X]

        
        # Predict classes
        predictions = predict(X)
        labels=['Recharge carte prépayée non aboutie','Retard d'exécution d'un ordre virement (normal, cih on line)	Code PIN non reçu','Non réception OTP (Activation, transfert ou recharge)',
        'Autre']
        
        predicted_label=labels[prediction] 
        
        # Add predictions to the DataFrame
        df['predicted_class'] = predicted_label
        
        # Display results
        st.write("Predictions:")
        st.write(df)
        
        # Option to download the result as a new Excel file
        result_file = 'predictions.xlsx'
        df.to_excel(result_file, index=False)
        st.download_button(label="Download Predictions", data=open(result_file, 'rb').read(), file_name=result_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
