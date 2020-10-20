import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TestData():
    def __init__(self):
        news = 'Trained_Model.sav'
        log = pickle.load(open(news, 'rb'))
        title1 = input("Enter title :")
        text1 = input("Enter text : ")
        total1 = title1 + ' ' + text1
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        News = ' '
        sentence = total1
        words = nltk.word_tokenize(sentence)
        words = [w for w in words if not w in stop_words]
        for words in words:
            News = News + ' ' + str(lemmatizer.lemmatize(words)).lower()
        x_predict = [News]
        result = log.predict(x_predict)
        print(result)
if __name__ == '__main__':
    TestData()
