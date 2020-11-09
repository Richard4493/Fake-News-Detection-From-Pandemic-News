import logging
import pandas as pd
import pickle as pk
from .feature import Feature
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
class Trainig:
    def train(self,filename):
        train = self.load_file(filename)
        train = train.fillna(' ')
        train['total'] = train['title'] + '()' + train['text']
        ft = Feature()
        data  = ft.feature(train)
        X_train = data['total']
        Y_train = train['label']
        tfidf = TfidfVectorizer(strip_accents=None,
                                lowercase=False,
                                preprocessor=None,
                                use_idf=True,
                                norm='l2',
                                smooth_idf=True)
        tf_idf_matrix = tfidf.fit_transform(X_train)
        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix,Y_train, random_state=0)
        Lr = LogisticRegression(C=10.0,random_state=0)
        Lr.fit(X_train, y_train)
        self.save_file(Lr ,"fake_news_model.sav")
        print("Accuracy : " + str(round(Lr.score(X_test,y_test),3)*100) + "%")
    def load_file(self,filename):
        try:                       #File input exception handling
            train = pd.read_csv(filename)
            return train
        except IOError:
            logging.error("Error: can\'t find the csv file or read data")
            exit()
    def save_file(self,lr,filename):
            pk.dump(lr,open(filename , 'wb'))