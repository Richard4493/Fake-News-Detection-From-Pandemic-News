import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
class Main:
    def __init__(self):
        train = pd.read_csv('corona_fake1.csv')
        train = train.fillna(' ')
        train['total'] = train['title'] + '()' + train['text']
        data  = self.feature(train)
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
        print("Accuracy : " + str(round(Lr.score(X_test,y_test),3)*100) + "%")
    def feature(self,train):
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for index, row in train.iterrows():
            filter_sentence = ''
            sentence = row['total']
            title = row['title']
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            l = 0
            for letter in title.split("(?!^)"):
                if letter == ' ' :
                    break
                if(letter.isupper()):
                    l = l + 1
            for words in words:
                filter_sentence = filter_sentence  + ' ' +str(lemmatizer.lemmatize(words)).lower()
            train.loc[index, 'total'] = filter_sentence + '()' + str(l) + '()' + str(len(row['title'])) + '()' +str(len(row['text']))
        return train
if __name__ == '__main__':
    Main()