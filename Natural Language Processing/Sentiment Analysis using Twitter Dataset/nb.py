# %% [markdown]
# # Imports

# %%
import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import pickle
from wordcloud import wordcloud
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# # Loading the dataset

# %%
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', names = 'Sentiment IDs Date Flag User Text'.split())

# %%
data.head()

# %%
data = data[['Sentiment', 'Text']]

# %%
data

# %%
data['Sentiment'].replace(4, 1,inplace=True)

# %%
# Negative Sentiment tweet count
data[data['Sentiment'] == 0].count()

# %%
text = list(data['Text'])

# %%
sentiment = list(data['Sentiment'])

# %%
# Positive Sentiment tweet count
data[data['Sentiment'] == 1].count()

# %% [markdown]
# # EDA

# %%
data.groupby("Sentiment").count().plot(kind = 'bar')

# %% [markdown]
# # Preprocessing Data

# %% [markdown]
# ### Steps:

# %% [markdown]
# ##### 1. Lowering case

# %% [markdown]
# ##### 2. Removing Stopwords

# %% [markdown]
# ##### 4. Removing punctuations

# %% [markdown]
# ##### 5. Removing hashtags
# 

# %% [markdown]
# ##### 2. Removing hyperlinks
# 

# %% [markdown]
# ##### 7. Removing user tags

# %% [markdown]
# ##### 3. Lemmatization

# %% [markdown]
# ##### 8. Removing emojis

# %%
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

# %%
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %%
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

# %%
newTweets_neg = processText(text[:800000])

# %%
newTweets_pos = processText(text[800000:])

# %% [markdown]
# # Visualisation

# %%
data_neg = " ".join(newTweets_neg)
wc = wordcloud.WordCloud(max_words=100, height=1280, width=720).generate(data_neg)
plt.imshow(wc)

# %%
data_pos = " ".join(newTweets_pos)
wc = wordcloud.WordCloud(max_words=100, height=1280, width=720).generate(data_pos)
plt.imshow(wc)

# %%
for i in newTweets_pos:
    newTweets_neg.append(i)
newTweets = newTweets_neg[:]

# %%
len(newTweets)

# %%
newTweets

# %% [markdown]
# # Splitting the data

# %%
x_train, x_test, y_train, y_test = train_test_split(newTweets, sentiment, test_size=0.30, random_state=42)

# %% [markdown]
# # TF-IDF Vectoriser

# %%
tfidf_vect = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=50000)

# %%
tfidf_vect.fit(x_train)

# %%
print("No. of feature words: ", len(tfidf_vect.get_feature_names_out()))

# %%
x_train = tfidf_vect.transform(x_train)

# %%
x_test = tfidf_vect.transform(x_test)

# %%
print(x_train)

# %% [markdown]
# # Creating a Model

# %% [markdown]
# ##### 1. LightGBM Classifier

# %%
model = LGBMClassifier()

# %%
model.fit(x_train, y_train)

# %%
y_pred = model.predict(x_test)

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# ##### 2. Logistic Regression

# %%
model2 = LogisticRegression()

# %%
model2.fit(x_train, y_train)

# %%
y_pred2 = model2.predict(x_test)
print(classification_report(y_test, y_pred2))

# %% [markdown]
# # Saving the models

# %%
pickle.dump(model2, open("SA LR-Model.pkl", "wb"))
pickle.dump(tfidf_vect, open("Vectoriser.pkl", "wb"))
pickle.dump(text, open("Text.pkl", 'wb'))
pickle.dump(data, open("Data.pkl", 'wb'))

# %% [markdown]
# # Making Predictions

# %% [markdown]
# ##### Loading the saved model

# %%
saved_model = pickle.load(open('SA LR-Model.pkl', 'rb'))
vectoriser = pickle.load(open("Vectoriser.pkl", 'rb'))

# %%
text = ["What a mad day i've had", 
       'Dont be sarcastic, look at the brighter side',
       'Love is a waste of time', 
       'My heart goes out to him',
       'He has a heart condition']
text = processText(text)

# %%
text

# %%
text = vectoriser.transform(text)

# %%
print(text)

# %%
predictions = saved_model.predict(text)
# print(len(predictions))
# print(predictions[0])
# print(type(predictions))
predictions = list(predictions)
for i in range(len(predictions)):
    if predictions[i] == 1:
        predictions[i] = 'Positive'
    else:
        predictions[i] = 'Negative'
print(predictions)

# %%



