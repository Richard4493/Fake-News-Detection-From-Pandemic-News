from sklearn.base import BaseEstimator, TransformerMixin
import nltk
import re
import pandas
import logging
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class WordLematization(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.word_lemmatizer = WordNetLemmatizer()

    def lemmatizer(self,data):
        stop_words = stopwords.words('english')
        train = []
        for row in data:
            filter_sentence = ''
            sentence = row
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for word in words:
                filter_sentence = filter_sentence + ' ' + str(self.word_lemmatizer.lemmatize(word))
            train.append(str(filter_sentence))
        logging.info("preprocessing done")
        return train

    def transform(self, df, y=None):

        return self.lemmatizer(df)

    def fit(self, df, y=None):
        return self    