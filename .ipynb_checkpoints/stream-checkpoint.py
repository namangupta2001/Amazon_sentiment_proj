import streamlit as st
from joblib import load

# Load the saved model and vectorizer
model = load("model.joblib")
vectorizer = load("vectorizer.joblib")

# Define the web application
st.title("Amazon Review Sentiment Analysis")
review_text = st.text_input("Enter a review:", "")

if review_text:
    # Convert the text input into a TF-IDF representation
    review_vector = vectorizer.transform([review_text])

    # Make a prediction using the trained model
    sentiment = model.predict(review_vector)[0]

    # Display the prediction
    if sentiment == 1:
        st.write("This review is positive.")
    else:
        st.write("This review is negative.")
