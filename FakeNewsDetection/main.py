from .feature import Feature
from .train import Trainig
from .prediction import Prediction
class fake_news_detection:
    def __init__(self):
        pass

    def train(self,filename):
        tr = Trainig()
        tr.train(filename)
    def predict(self,x):
            Prediction.predict(x)