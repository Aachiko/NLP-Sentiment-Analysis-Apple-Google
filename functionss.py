#import libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,precision_score, recall_score, f1_score, auc,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import joblib
from sklearn.ensemble import RandomForestClassifier
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
# from mlxtend.plotting import plot_confusion_matrix
import matplotlib.cm as cm
from matplotlib import rcParams
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

# Function for deployment:
def predict_sentiment(tweet):
    # Stem the tweet
    stemmer = PorterStemmer()
    stemmed_tweet = [stemmer.stem(word) for word in tweet.split() if word not in stopwords_list]

    # Vectorize using loaded TF-IDF vectorizer
    tweet_tfidf = loaded_vectorizer.transform([' '.join(stemmed_tweet)])

    # Predict sentiment
    sentiment = loaded_model.predict(tweet_tfidf)[0]
    return sentiment

def extract_product(tweet):
    # Dictionary maps keywords to products
    tweet_lower = str(tweet).lower()
    product_keywords = {
        'Apple': ['apple', 'iphone', 'ipad', 'macbook', 'apple watch',
                  'airpods', 'ios', 'macos', 'app store', 'icloud', 'itunes'],
        'Google': ['google', 'pixel', 'pixelbook', 'chromebook', 'google home',
                   'nest', 'android', 'google play store',
                   'google maps', 'gmail', 'goog']
    }

    # Check if tweet is a string
    if isinstance(tweet, str):
        # Iterate through the dictionary to find a match
        for product, keywords in product_keywords.items():
            if any(keyword in tweet_lower for keyword in keywords):
                return product
    return None
# function to handle @), HTML tags and URLs in tweets
def preprocess_text(text):
    # convert text lower case
    text = text.lower()  
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove Twitter usernames (mentions)
    text = re.sub(r'@\w+\s*', '', text)

    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    return text

#Preprocess and tokenize
def preprocess_and_tokenize(text):
    text = preprocess_text(text)
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords_list]
    return ' '.join(filtered_tokens)

# Function to stem the text
def stem_text(text):
    stemmer = PorterStemmer()
    tokenized_text = word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in tokenized_text]
    return " ".join(stemmed_text)