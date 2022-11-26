"""

@author: onkar
"""

# Importing required libraries
import streamlit as st
import pickle
import re
import string
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
from afinn import Afinn
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


# Body of the application
st.header("Hotel Review Prediction Application.")
st.markdown("This application is trained on machine learning model.")
st.markdown("This application can predict if the given **review** is **Positive, Negative or Neutral**")


# User input
text = st.text_input("Type your review here", """""")


# Loading model
model = pickle.load(open("C:/Users/onkar/Downloads/decsn_tree_trained_aug.pickle", "rb"))


# Loading vectorizer
vectorizer = pickle.load(open("C:/Users/onkar/Downloads/vectorizer_aug_tf.pickle", "rb"))


# Text preprocessing functions
def convert_to_lower(text):
    return text.lower()


def remove_numbers(text):
    number_pattern = r'\d+'
    without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
    return without_number


def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)


def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc


# Output of influencing words
def list_convert(text):
    input_list = list(text.split(" "))
    print(input_list)
    return input_list


# Button of working
if st.button("Click to make prediction..."):

    # Cleaning the text input
    final_text = convert_to_lower(text)
    final_text = remove_numbers(final_text)
    final_text = remove_punctuation(final_text)
    final_text = remove_stopwords(final_text)
    final_text = remove_extra_white_spaces(final_text)
    final_text = lemmatizing(final_text)

    # # Looking at the final text
    # st.write(final_text)

    # Vectorizing the text
    final_text = [final_text]
    input_vectorized = vectorizer.transform(final_text).toarray()

    # Making prediction on vectorized input
    prediction = model.predict(input_vectorized)

    # Getting influencing words from text input
    # Converting text input into list of words.
    final_input_list = list_convert(text)

    # Using Affin to check sentiment scores
    afinn = Afinn()
    afn_score = [afinn.score(word) for word in final_input_list]

    # Making a dictionary of words and sentiment scores
    dict = {
        k: v for (k, v) in zip(final_input_list, afn_score)
    }

    # Prediction output
    if prediction == -1:
        word_list = []
        for key, value in dict.items():
            if value < 0:
                word_list.append(key)
            else:
                pass
        st.write("The review is Negative.")
        st.write(word_list)

    elif prediction == 1:
        word_list = []
        for key, value in dict.items():
            if value > 0:
                word_list.append(key)
            else:
                pass
        st.write("The review is Positive.")
        st.write(word_list)

    else:
        word_list = []
        for key, value in dict.items():
            if value == 0:
                word_list.append(key)
            else:
                pass
        st.write("The review is Neutral.")
        st.write(word_list)
else:
    pass
