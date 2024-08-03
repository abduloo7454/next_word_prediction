import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the LSTM model
model = load_model('next_word_lstm.h5')  # Ensure your model is saved as an H5 file

##load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Create a function to predict the net words

def predict_next_word(model,tokenizer,text,max_sequnce_len):
  tokel_list = tokenizer.texts_to_sequences([text])[0]
  if len(tokel_list) >= max_sequence_len:
    tokel_list = tokel_list[-(max_sequence_len-1):] #Ensure the sequence lenght matches max_sequnce_len-1
  token_list = pad_sequences([tokel_list],maxlen=max_sequence_len-1,padding='pre')
  predicted = model.predict(token_list, verbose=0)
  predicted_word_index = np.argmax(predicted, axis=1)
  for word, index in tokenizer.word_index.items():
    if index == predicted_word_index:
      return word
  return None

  #streamlit app
  st.title("Next word Prediction with LSTM and Early Stopping")
  input_text = st.text_input("Enter some text:")
  if st.button("Predict next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word(model,tokenizer,input_text,max_sequence_len)
    st.write(f"Predicted next word: {next_word}")
