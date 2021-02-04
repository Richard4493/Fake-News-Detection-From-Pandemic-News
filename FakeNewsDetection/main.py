import pickle as pk
import pandas
from .detect import Detect
from .prediction import Prediction
import logging
class fake_news_detection:
    def __init__(self,filename):
        logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')
        self.filename = filename
        self.data = self.loadModel()
        self.tr = Detect(self.data)

    def loadModel(self):
        try : 
            data = pandas.read_csv(self.filename)
            logging.info("csv file found")
        except :
            logging.error("file not found")
            exit(0)           
        return  data
    def train(self, model = "LR"):      
        self.tr.train(model)
    def predict(self,text,title):
        return Prediction().predict(text , title)
