from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
class Train:
    def __init__(self , x_train , y_train):
        self.x_train = x_train
        self.y_train = y_train
    def svm(self):
        SVM=LinearSVC(random_state=0)
        SVM.fit(self.x_train,self.y_train)
        return SVM
    def logisticalRegression(self):
        Lr = LogisticRegression(C=10.0,random_state=0)
        Lr.fit(self.x_train,self.y_train)
        return Lr  
    def rf(self):
        RF=RandomForestClassifier(random_state=0)
        RF.fit(self.x_train,self.y_train)
        return RF          