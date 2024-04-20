import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt


# Function to perform sentiment analysis
def analyze_sentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Main Menu
def main_menu():
    st.title('Movie Review Sentiment Analysis')
    option = st.sidebar.selectbox(
        'Menu',
        ['Sentiment Analysis', 'About the Project', 'Outcomes']
    )

    if option == 'Sentiment Analysis':
        st.write('Sentiment Analysis')
        sentiment_analysis()    

    elif option == 'About the Project':
        about()
        
    elif option == 'Outcomes':
        st.markdown("""
        The key outcomes of this project are:

        - Successful preprocessing and preparation of the IMDB movie review dataset for sentiment analysis.
        - Training of a Logistic Regression model on the preprocessed data using a custom script.
        - Deployment of the trained model to a SageMaker instance for hosting and inference.
        - Evaluation of the model's performance on the test dataset, with the test accuracy printed.        
        
        """)

# Function to perform sentiment analysis
def about():
    st.markdown("""    
    This is an end-to-end project for sentiment analysis on IMDB movie reviews using a Logistic Regression model. The key aspects of the project are:

    1. **Data Preprocessing**
    - The dataset used is the IMDB movie review dataset, which contains 50,000 movie reviews labeled as either "positive" or "negative".
    - The data is loaded into a Pandas DataFrame, and the shape of the dataset is checked (50,000 rows, 2 columns).
    - There are no missing values in the dataset.

    2. **Train-Test Split**
    - The dataset is split into training (80%) and testing (20%) sets using the `train_test_split` function from scikit-learn.
    - The training and testing sets are saved as CSV files and uploaded to an S3 bucket for use in the model training.

    3. **Model Training**
    - A custom script `script.py` is created to train a Logistic Regression model on the movie review data.
    - The script reads the training and testing data from the S3 bucket, preprocesses the text data using a CountVectorizer, and trains the Logistic Regression model.
    - The trained model is then saved to the local model directory.

    4. **Model Deployment**
    - The trained model is deployed to a SageMaker instance using the `SKLearn` estimator.
    - The deployment process includes setting up the necessary configurations, such as the instance type, framework version, and resource limits.
    - The trained model is then hosted on the SageMaker instance, and the test accuracy is printed.

    #### About the Developers
    This project was developed by Vraj Patel and Ronak Makwana as part of their term project for the subject AMOD 5410H. 
    """)

def sentiment_analysis():
    # Text input for review
    review_text = st.text_input("Enter your movie review here:")

    if st.button("Submit"):
        if review_text:
            # Analyze sentiment of the input review
            sentiment = analyze_sentiment(review_text)

            # Display the sentiment
            st.write(f"Sentiment: {sentiment}")


# Run the main menu
main_menu()
