import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np

# Load model, tokenizer, and label encoder
@st.cache_resource  # Caches the model loading to improve performance
def load_resources():
    model = load_model('sentiment_model.keras')
    tokenizer = joblib.load('tokenizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_resources()

def predict_sentiment(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)

    # Predict sentiment
    prediction = model.predict(padded_sequence)
    sentiment_index = np.argmax(prediction, axis=-1)[0]
    return label_encoder.inverse_transform([sentiment_index])[0]

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment:")

# Input text box
user_input = st.text_area("Input Text", "")

if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")