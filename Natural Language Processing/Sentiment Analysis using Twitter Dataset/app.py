import streamlit as st
import pickle
from nltk.stem import WordNetLemmatizer
import re
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import wordcloud
import warnings

warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

length = pickle.load(open("Length.pkl", "rb"))
wc_pos = pickle.load(open("Positive Wordcloud.pkl", "rb"))
wc_neg = pickle.load(open("Negative Wordcloud.pkl", "rb"))
d = pickle.load(open("Sentiment Data.pkl", "rb"))

with st.sidebar:
    selected = option_menu("Main Menu", ["Data Visualisation", 'Predictions'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)

if (selected == 'Predictions'):
    urlPattern = r'((https://)[^ ]*|(http://)[^ ]*|( www.)[^ ]*)'
    userPattern = r'@[^\s]+'
    alphaPattern = r'[^A-Za-z0-9]'
    wnl = WordNetLemmatizer()

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
            ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
            ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
            ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
            '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
            '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
            ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


    def processText(text):
        processedText = []
        for tweet in text:
            tweet = str.lower(tweet)
            
            tweet = re.sub(urlPattern, 'URL', tweet)
            
            tweet = re.sub(userPattern, 'USER', tweet)
            
            tweet = re.sub(alphaPattern, ' ', tweet)
            
            for emoji in emojis.keys():
                tweet = tweet.replace(emoji, 'EMOJI' + emojis[emoji])
            
            newTweet = ''
            for word in tweet.split():
                if len(word) > 1:
                    newTweet += wnl.lemmatize(word) + ' '
            newTweet.rstrip()
            processedText.append(newTweet)
        return processedText

    st.header("Sentiment Analysis using Twitter Dataset")

    # Taking Input
    tweets = []
    tweet = st.text_input("Enter tweet: ")
    tweets.append(tweet)
    tweets = processText(tweets)
    button = st.button("Predict")
    # print(tweets)

    model = pickle.load(open('SA LR-Model.pkl', 'rb'))
    vectoriser = pickle.load(open("Vectoriser.pkl", 'rb'))

    tweets = vectoriser.transform(tweets)
    prediction = model.predict(tweets)

    if button and prediction == 1:
        st.write("Positive")
    elif button:
        st.write("Negative")
        
elif selected == 'Data Visualisation':
    
    st.write("Distribution of tweets according to sentiment (0 = Negative, 1 = Positive)")
    st.bar_chart(d)
    
    st.write("Distribution of tweets according to length")
    fig = plt.figure(figsize=(12, 8))
    sns.kdeplot(length)
    plt.xlabel("Length of Tweets")
    st.pyplot(fig)

    st.write("Most common words in Positive Sentiment Tweets")
    plt.imshow(wc_pos)
    st.pyplot()
    
    st.write("Most common words in Negative Sentiment Tweets")
    plt.imshow(wc_neg)
    st.pyplot()