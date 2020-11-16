import pandas as pd
import pickle as pk
from .feature import Feature
from .classifier import Train
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import confusion_matrix,classification_report

class Detect:
    def __init__(self, filename):
        train = self.load_file(filename)
        train = train.fillna(' ')
        train['total'] = train['title'] + '()' + train['text']
        ft = Feature()
        data  = ft.feature(train)
        X_train = data['total']
        Y_train = train['label']
        tfidf = TfidfVectorizer(use_idf=True,
                                norm='l2',
                                smooth_idf=True,ngram_range=(2,2))
        tf_idf_matrix = tfidf.fit_transform(X_train)
        X_train, self.X_test, y_train, self.y_test = train_test_split(tf_idf_matrix,Y_train, random_state=0)
        self.t = Train(X_train , y_train)
    def train(self,model = "LR"):
        if(model == "LR"):
            m= self.t.logisticalRegression()
        elif model == "SVM":
            m =  self.t.svm()
        elif model == "RF":
            m= self.t.rf()
        else:
            logging.error("invalid model")        
        self.save_file(m ,"fake_news_model.sav")
        return m
    def compare(self):
        m1= self.t.logisticalRegression()
        m2= self.t.svm()
        m3= self.t.rf()
        return m1,m2,m3
    def getAccuracy(self,m):
        return round(m.score(self.X_test, self.y_test), 3) * 100
    def getConfmatrix(self,m):
        pred = m.predict(self.X_test)
        return confusion_matrix(self.y_test,pred)
    def getReport(self,m):
        pred = m.predict(self.X_test)
        return classification_report(self.y_test,pred)
    def load_file(self,filename):
        try:                       #File input exception handling
            train = pd.read_csv(filename)
            return train
        except IOError:
            logging.error("Error: can\'t find the csv file or read data")
            exit()
    def save_file(self,lr,filename):
            pk.dump(lr,open(filename , 'wb'))