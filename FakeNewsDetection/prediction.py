import pickle
from .detect import Detect
import pandas
class Prediction:
    def __init__(self):
       pass
    def predict(self,text , title):
        df = pandas.DataFrame({
                                            'text' :[text],
                                            'title' : [title]})
        log = self.load_file('fake_news_model.sav')
        result=log.predict(df)
        return result[0]
    def load_file(self,filename):
        fake_news_model = open(filename, 'rb')
        p = pickle.load(fake_news_model)
        return p
