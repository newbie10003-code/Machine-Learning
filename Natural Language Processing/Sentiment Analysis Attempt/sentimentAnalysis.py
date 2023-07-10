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
from sklearn.feature_extraction.text import CountVectorizer

class sentimentAnalysis:
    def clean_tags(tweet):
        tags = re.findall("@[\w]*", tweet)
        # print(tags)
        for i in tags:
            tweet = str.replace(tweet, i, "")
        return tweet
    
    def clean_punctuations(tweet):
        punctuations = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'
        tweet = re.sub(punctuations, "", tweet)
        return tweet

    def removeHashtags(tweet):
        tags = re.findall("#[\w]*", tweet)
        # print(tags)
        for i in tags:
            tweet = str.replace(tweet, i, "")
        return tweet
    

