#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


import numpy as np
import nltk.corpus
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import lightgbm as lgb
import pickle
import joblib


# # Reading the database

# In[2]:


data = pd.read_csv('train.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


# No. of positive tweets
data[data['label'] == 0].count()


# In[7]:


# No of negative tweets
data[data['label'] == 1].count()


# # Preprocessing the data

# In[8]:


data.dropna(inplace=True)


# ##### Removing the user tags

# In[9]:


def clean_tags(tweet):
    tags = re.findall("@[\w]*", tweet)
#     print(tags)
    for i in tags:
        tweet = str.replace(tweet, i, "")
    return tweet


# In[10]:


data['cleaned_tweets'] = data['tweet'].apply(clean_tags)
data.head()


# ##### Removing the punctuations

# In[11]:


string.punctuation


# In[12]:


def clean_punctuations(tweet):
    punctuations = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
    tweet = re.sub(punctuations, "", tweet)
    return tweet
# string.punctuation


# In[13]:


# print(type(string.punctuation))
data['cleaned_tweets'] = data['cleaned_tweets'].str.replace("[^A-Za-z#]", " ")
data['cleaned_tweets'].head(-5)
# data['cleaned_tweets'].iloc[31957]


# ##### Removing words less than a given size

# In[14]:


data['cleaned_tweets'] = data['cleaned_tweets'].apply(lambda x : ["".join(word) for word in x.split() if len(word) > 4])


# In[15]:


data['cleaned_tweets']


# In[16]:


def joinWords(tweet):
    ans = ""
    for word in tweet:
        ans = " ".join(tweet)
    return ans


# In[17]:


data['cleaned_tweets'] = data['cleaned_tweets'].apply(joinWords)


# In[18]:


data['cleaned_tweets']


# ###### Lemmatization

# In[19]:


from nltk.stem import WordNetLemmatizer


# In[20]:


lemmer = WordNetLemmatizer()


# In[21]:


data['cleaned_tweets'] = data['cleaned_tweets'].apply(lambda x : [lemmer.lemmatize(word) for word in x.split()])
data['cleaned_tweets'] = data['cleaned_tweets'].apply(joinWords)


# In[22]:


data['cleaned_tweets'].head(-5)


# ##### Removing Hashtags

# In[23]:


def removeHashtags(tweet):
    tags = re.findall("#[\w]*", tweet)
#     print(tags)
    for i in tags:
        tweet = str.replace(tweet, i, "")
    return tweet


# In[24]:


data['cleaned_tweets'] = data['cleaned_tweets'].apply(removeHashtags)
# count = 0
# for i in data['cleaned_tweets'].iloc[31953].split():
#     count += 1
# print(count)


# # Exploratory Data Analysis

# ##### Calculating Length of each tweet

# In[25]:


data['length'] = data['cleaned_tweets'].apply(lambda x : len(x))
data.head(-5)


# ##### Checking to see if length dictates the sentiment of tweet

# In[26]:


sns.boxplot(data = data, x = 'label', y = 'length')


# ##### Count of positive and negative tweets

# In[27]:


sns.countplot(data = data, x = 'label')


# # Creating and training the model

# ##### Creating a bag of words

# In[28]:


# def joinWords(tweet):
#     ans = ""
#     for word in tweet:
#         ans = " ".join(tweet)
#     return ans


# In[29]:


allWords = ""
for tweet in data['cleaned_tweets']:
    allWords = allWords + tweet + " "

allWords = allWords.rstrip()
allWords


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer


# In[31]:


count_vect = CountVectorizer(stop_words = 'english')


# In[32]:


count_vect = count_vect.fit_transform(data['cleaned_tweets'])


# In[33]:


count_vect = count_vect.astype(np.float64)


# ##### Splitting the data

# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[35]:


x = count_vect;
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


# ##### Creating a model (LightGBM)

# In[36]:


model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
model.fit(x_train,y_train,eval_set=[(x_test,y_test),(x_train,y_train)],
          verbose=20,eval_metric='logloss')


# ##### Creating a model (Logistic Regression)

# In[37]:


model2 = LogisticRegression(random_state = 42)
model2.fit(x_train, y_train)


# ##### Creating a model (Random Forrest Classifier)

# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


model3 = RandomForestClassifier(random_state = 42, verbose = 2, n_estimators=10)


# In[40]:


model3.fit(x_train, y_train)


# ##### Creating a model (Decision Tree)

# In[41]:


from sklearn.tree import DecisionTreeClassifier


# In[42]:


model4 = DecisionTreeClassifier(random_state = 42)


# In[43]:


model4.fit(x_train, y_train)


# # Making Predictions

# ##### Predictions from LightGBM classifier

# In[44]:


pred = model.predict(x_test)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


print(accuracy_score(pred, y_test))


# ##### Predictions from Logistic Regression model

# In[47]:


pred2 = model2.predict(x_test)


# In[48]:


print(accuracy_score(pred2, y_test))


# ##### Predictions from RandomForrest classifier

# In[49]:


pred3 = model3.predict(x_test)


# In[50]:


print(accuracy_score(pred3, y_test))


# ##### Predictions from Decision Tree classifier

# In[51]:


pred4 = model4.predict(x_test)


# In[52]:


print(accuracy_score(pred4, y_test))


# # Saving the best algorithm

# In[54]:


from joblib import Parallel, delayed
joblib.dump(model, "model.pkl")


# In[ ]:




