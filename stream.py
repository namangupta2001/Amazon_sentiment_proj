import streamlit as st
from joblib import load
from PIL import Image
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
        pos = '<p style="font-family:Courier; color:Green; font-size: 20px;">This review is positive</p>'
        st.markdown(pos, unsafe_allow_html=True)
        st.write("This review is positive.")
    else:
        neg = '<p style="font-family:Courier; color:Red; font-size: 20px;">This review is negative</p>'
        st.markdown(neg, unsafe_allow_html=True)
        st.write("This review is negative.")

st.markdown("- I have tested the model on Naive Bayes , XGBoost ,KNN classifier and Logistic Regression .")
st.markdown("- Logistic Regression gave the highest accuracy of  83% .")
st.markdown("- [Github](https://github.com/namangupta2001/Amazon_sentiment_proj)")


# Load the image from a file
image = Image.open("linear regression.png")

# Display the image in Streamlit
st.image(image, caption="Confusion matrix of Logistic Regression ", use_column_width=True)        