import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentimentAnalysis import sentimentAnalysis

st.header("Sentiment Analysis Of Tweets")

tweet = (st.text_input(label = "Enter a tweet"))
click = st.button("Submit")

model = joblib.load("model.pkl")

if click:

    # Preprocessing
    tweet = sentimentAnalysis.clean_tags(tweet)
    tweet = sentimentAnalysis.clean_punctuations(tweet)
    tweet = sentimentAnalysis.removeHashtags(tweet)

    count_vect = CountVectorizer(stop_words = 'english')
    count_vect = count_vect.fit_transform(tweet)
    count_vect = count_vect.astype(np.float64)

    st.write(count_vect)
    
    ans = model.predict(count_vect)
    st.write("The sentiment of the given tweet is: ", ans[0])
    st.write(type(ans))