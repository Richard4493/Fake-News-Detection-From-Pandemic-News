from .detect import Detect
from .prediction import Prediction
class fake_news_detection:
    def __init__(self,filename):
        self.tr = Detect(filename)
    def train(self):
        self.tr.train()
    def predict(self,x):
        Prediction.predict(x)
    def compare(self):
        m1,m2,m3= self.tr.compare()
        data= {'accuracy':(self.tr.getAccuracy(m1), self.tr.getAccuracy(m2), self.tr.getAccuracy(m3)),
               'cmatrix':(self.tr.getConfmatrix(m1),self.tr.getConfmatrix(m2),self.tr.getConfmatrix(m3)),
               'creport':(self.tr.getReport(m1),self.tr.getReport(m2),self.tr.getReport(m3))}
        return data