import streamlit as st
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


max_features = 10000
maxlen = 500


RNN_model = load_model('RNN_ml2.keras')  


word_index = imdb.get_word_index()


def encode_text(text, word_index, maxlen=500):
    text = text.lower().replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("'", "").split()
    encoded = []
    for word in text:
        index = word_index.get(word)
        if index is not None and index < max_features:
            encoded.append(index + 3)
        else:
            encoded.append(2)  
    return pad_sequences([encoded], maxlen=maxlen)


st.title("Sentiment Analysis")
st.write("Entrez une critique de film pour prédire si elle est **Positive** ou **Négative**.")


user_input = st.text_area("Votre critique de film :", height=150)

if st.button("Prédire"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer une critique avant de lancer la prédiction.")
    else:
        
        encoded_sample = encode_text(user_input, word_index, maxlen)
        prediction = RNN_model.predict(encoded_sample)
        proba = prediction[0][0]
      
        st.subheader("Résultat de la prédiction :")
        if proba >= 0.5:
            st.success(f"**Positive** ({proba*100:.2f}%) 👍")
        else:
            st.error(f"**Négative** ({(1 - proba)*100:.2f}%) 👎")
