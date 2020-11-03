import re
import nltk
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
class Main:
    def _prepro(self, train):
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for index, row in train.iterrows():
            filter_sentence = ''
            sentence = row['total']
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for words in words:
                filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()
            train.loc[index, 'total'] = filter_sentence
        return train
    def __init__(self):
        try:                       #File input exception handling
            train = pd.read_csv('corona_fake_news.csv')
        except IOError:
            logging.warning("Error: can\'t find the csv file or read data")
            exit()
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
        print("Accuracy Of LR : " + str(round(Lr.score(X_test,y_test),3)*100) + "%")
        lr_pred = Lr.predict(X_test)
        print("\nConfusion Matrix of Logistic Regression Classifier:\n")
        print(confusion_matrix(y_test, lr_pred))
        print("\nCLassification Report of Logistic Regression Classifier:\n")
        print(classification_report(y_test, lr_pred))
        SVM=LinearSVC(random_state=0)
        SVM.fit(X_train,y_train)
        SVM_pred= SVM.predict(X_test)
        print("Accuracy Of SVM : " + str(round(SVM.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of SVM Classifier:\n")
        print(confusion_matrix(y_test, SVM_pred))
        print("\nCLassification Report of SVM Classifier:\n")
        print(classification_report(y_test, SVM_pred))
        RF=RandomForestClassifier(random_state=0)
        RF.fit(X_train,y_train)
        RF_pred= RF.predict(X_test)
        print("Accuracy Of RF : " + str(round(RF.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of RF Classifier:\n")
        print(confusion_matrix(y_test, RF_pred))
        print("\nCLassification Report of RF Classifier:\n")
        print(classification_report(y_test, RF_pred))

    def feature(self,train):
        train = self._prepro(train)
        for index, row in train.iterrows():
            title = row['title']
            l = 0
            for letter in title.split("(?!^)"):
                if letter == ' ' :
                    break
                if(letter.isupper()):
                    l = l + 1
            train.loc[index, 'total'] =train.loc[index, 'total']  + '()' + str(l)+ '()' + str(len(row['title'])) + '()' +str(len(row['text']))
        return train
if __name__ == '__main__':
    Main()