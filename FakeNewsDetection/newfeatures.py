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
    def lemmatizerNew(self,data_final):
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        train = []
        for entry in data_final:
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
            train.append(str(Final_words))
        logging.info("preprocessing done")
        return train
    def transform(self, df, y=None):
        df = self.tokenize(df)
        return  self.lemmatizerNew(df)
        #return self.lemmatizer(df)

    def fit(self, df, y=None):
        return self    