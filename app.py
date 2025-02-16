import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model 


word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('RNN_imdb.h5')

#function to decode reviews 

def decode_review(encoded_review): 
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text) : 
  words = text.lower().split()
  encoded_review =[word_index.get(word , 2) +3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review],maxlen = 500)
  return padded_review

### prediction function 

def predict_sentiment(review) :
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)
  sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
  return sentiment, prediction[0][0]


### streamlit app 

import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

#user input 

user_input = st.text_area('Movie Review')

if st.button('Classify') : 
    
    sentiment , score = predict_sentiment(user_input)
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score}')
else:
    st.write('Please enter a movie review')