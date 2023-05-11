import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentimentAnalysis import sentimentAnalysis

st.header("Sentiment Analysis Of Tweets")

tweets = ['agdfhfbnfvri'] * 16033
tweet = (st.text_input(label = "Enter a tweet"))
tweets[0] = (tweet)
click = st.button("Submit")

model = joblib.load("model.pkl")

if click:

    # Preprocessing
    tweets[0] = sentimentAnalysis.clean_tags(tweets[0])
    tweets[0] = sentimentAnalysis.clean_punctuations(tweets[0])
    tweets[0] = sentimentAnalysis.removeHashtags(tweets[0])

    count_vect = CountVectorizer(stop_words = 'english')
    count_vect = count_vect.fit_transform(tweets)
    count_vect = count_vect.astype(np.float64)

    st.write(count_vect)
    
    ans = model.predict(count_vect)
    st.write("The sentiment of the given tweet is: ", ans[0])
    st.write(type(ans))