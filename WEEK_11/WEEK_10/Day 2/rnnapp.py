import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pickle

model = load_model("lstm_model.h5")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('max_len.pkl', 'rb') as f:
    max_len = pickle.load(f)

st.title("Next Word Prediction")

text = st.text_input("Enter text")

if st.button("Predict"):
    if text.strip():
        seq = tokenizer.texts_to_sequences([text])[0]
        if len(seq) > max_len - 1:
            seq = seq[-(max_len - 1):]
        elif len(seq) < max_len - 1:
            seq = [0] * ((max_len - 1) - len(seq)) + seq
        seq = np.array([seq])
        pred = model.predict(seq, verbose=0)
        predicted_index = np.argmax(pred)
        predicted_word = tokenizer.index_word.get(predicted_index, "Unknown")
        st.write(f"Predicted next word: {predicted_word}")
    else:
        st.write("Please enter some text.")