import streamlit as st
import pickle
import re

# Load the trained classifier and vectorizer from .pkl files
with open('text_classifier.pkl', 'rb') as f:
    classifier, vectorizer = pickle.load(f)

# Function to preprocess and predict the class of text
def preprocess_text(text):
    # Remove numbers from the text
    text = re.sub(r'\d+', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def predict(text):
    # Preprocess the input text
    text = preprocess_text(text)
    # Transform text to TF-IDF features using the vectorizer
    features = vectorizer.transform([text])
    # Predict the class using the classifier
    prediction = classifier.predict(features)[0]
    return prediction

# Class label mapping
labels = {
    0: "Recharge carte prépayée non aboutie",
    1: "Retard d'exécution de virement",
    2: "Code PIN non reçu",
    3: "Retrait non servi",
    4: "Non réception OTP",
    5: "Autre"
}

# Initialize conversation history in session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit app setup
st.title("Text Classification Chatbot")
st.write("Enter a message and get a predicted class for your query.")

# User text input with a unique key
user_input = st.text_input("You:", key="user_input")

# Predict and display the response if input is provided
if user_input:
    # Generate a prediction and response
    predicted_class = predict(user_input)
    response = labels[predicted_class]

    # Add user input and bot response to the conversation history
    st.session_state.history.append(f"You: {user_input}")
    st.session_state.history.append(f"Bot: {response}")

    # Clear the input field after submission
    st.session_state.user_input = ""  # Reset input field value

# Display the conversation history
st.write("### Conversation History")
for message in st.session_state.history:
    st.write(message)
