import pandas as pd
import pickle as pk
from sklearn.model_selection import train_test_split
import logging
from sklearn.metrics import confusion_matrix,classification_report
from .classifier import *
class Detect:
    def __init__(self, data):
        self.data =data
        self.X_train, self.X_test, self.y_train, self.y_test  = self.get_train_data(self.data)
    def train(self,model):
        pipe = get_models()[model]
        a = pipe.fit(self.X_train , self.y_train)
        self.save_file(pipe ,"fake_news_model.sav")
        return pipe
    def get_train_data(self,dataset):
        dataset.dropna(inplace=True)
        X= dataset
        y = dataset['label']

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.10, random_state=0) 
        return X_train,X_test,y_train,y_test              
    def getAccuracy(self,m):
        return round(m.score(self.X_test, self.y_test), 3) * 100
    def save_file(self,lr,filename):
            pk.dump(lr,open(filename , 'wb'))