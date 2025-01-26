import streamlit as st
import pickle
import os

# Load the pre-trained model and vectorizer
def load_model_and_tfidf():
    with open(r"X:\Github\Sentiment-Analysis-Amazon-TFIDF\models\model.pkl", 'rb') as f:
        svc_model = pickle.load(f)
    
    with open(r"X:\Github\Sentiment-Analysis-Amazon-TFIDF\models\tfidf_vectorizer.pkl", 'rb') as f:
        tfidf = pickle.load(f)
        
    return svc_model, tfidf


svc_model, tfidf = load_model_and_tfidf()

# Title and description
st.title("Sentiment Analysis Application")
st.write("""
This application predicts the sentiment of a given review (Positive or Negative). 
Enter your review in the text box below and click 'Predict Sentiment' to see the result.
""")

# Define a reset function
def reset_input():
    st.session_state['review_input'] = ""

# Initialize session state for input field
if "review_input" not in st.session_state:
    reset_input()

# Input text box
review_input = st.text_area("Enter your review here:", key="review_input")

# Buttons for predict and clear
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("Predict Sentiment")
with col2:
    clear_btn = st.button("Clear", on_click=reset_input)

# Prediction logic
if predict_btn:
    if review_input.strip() != "":
        # Transform input using the loaded vectorizer
        review_tfidf = vectorizer.transform([review_input])
        
        # Make prediction
        prediction = model.predict(review_tfidf)[0]

        # Map prediction to sentiment label
        sentiment = "Positive" if prediction == 1 else "Negative"

        # Display the result
        if sentiment == "Positive":
            st.success("The sentiment is: Positive")
        else:
            st.error("The sentiment is: Negative")
    else:
        st.error("Please enter a valid review.")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("""
This sentiment analysis application uses a pre-trained Decision Tree Classifier 
and a TF-IDF vectorizer to predict whether a review is positive or negative.
""")

st.sidebar.header("How It Works")
st.sidebar.write("""
1. Enter a review in the text box.
2. Click the 'Predict Sentiment' button to see the result.
3. Use the 'Clear' button to reset the input.
""")

st.sidebar.header("Developer Notes")
st.sidebar.write("""
- This application is built with Python and Streamlit.
- The machine learning model used is a Decision Tree Classifier trained on TF-IDF features.
""")

st.sidebar.header("Contact")
st.sidebar.write("""
For questions or suggestions, reach out to:
- **Email**: ahmed.hamdii.kamal@gmail.com
- **GitHub**: https://github.com/hamdii10
""")
