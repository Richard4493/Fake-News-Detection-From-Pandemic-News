import pickle
class Prediction:
    def predict(self,x_predict):
        log = self.load_file('fake_news_model.sav')
        result=log.predict(x_predict)
        print(result)
    def load_file(self,filename):
        fake_news_model = open(filename, 'wb')
        p = pickle.load(fake_news_model)
        return p