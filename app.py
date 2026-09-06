import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# Page configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="RNN Sentiment Analyzer",
    page_icon="🎬"
)

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

@st.cache_resource
def load_sentiment_model():
    return tf.keras.models.load_model("simple_rnn_imdb.keras")


model = load_sentiment_model()

# ---------------------------------------------------
# IMDB vocabulary
# ---------------------------------------------------

word_index = imdb.get_word_index()

MAX_LENGTH = 500


# ---------------------------------------------------
# Preprocessing
# ---------------------------------------------------

def preprocess_review(review):

    words = review.lower().split()

    encoded_review = []

    for word in words:
        if word in word_index:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH
    )

    return padded_review


# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
    """
    This application uses a Simple Recurrent Neural Network (RNN)
    trained on the IMDB movie review dataset to classify a review
    as positive or negative.
    """
)

review = st.text_area(
    "Enter a movie review:",
    placeholder="Example: This movie was absolutely fantastic..."
)


if st.button("Analyze Sentiment"):

    if review.strip():

        processed_review = preprocess_review(review)

        prediction = model.predict(
            processed_review,
            verbose=0
        )[0][0]

        if prediction >= 0.5:

            st.success("Positive Review 😊")

            st.write(
                f"Positive sentiment score: **{prediction:.2%}**"
            )

        else:

            st.error("Negative Review 😞")

            st.write(
                f"Positive sentiment score: **{prediction:.2%}**"
            )

    else:

        st.warning("Please enter a movie review.")
