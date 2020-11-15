import re
import nltk
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
class Compare:
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
        logging.info('preprocessing done')
        return train
    def __init__(self):
        try:                       #File input exception handling
            data = pd.read_csv('corona_fake_news.csv')
        except IOError:
            logging.warning("Error: can\'t find the csv file or read data")
            exit()
        data = data.fillna(' ')
        data['total'] = data['title'] + '()' + data['text']
        data  = self.feature(data)
        X_train = data['total']
        Y_train = data['label']
        tfidf = TfidfVectorizer(preprocessor=None,
                                use_idf=True,
                                norm='l2',
                                smooth_idf=True)
        tf_idf_matrix = tfidf.fit_transform(X_train)
        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix,Y_train, random_state=0)
        lin_svm=LinearSVC(random_state=0)
        lin_svm.fit(X_train,y_train)
        lin_pred= lin_svm.predict(X_test)
        print("Accuracy Of lin_svm : " + str(round(lin_svm.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of lin_svm Classifier:\n")
        print(confusion_matrix(y_test, lin_pred))
        print("\nCLassification Report of lin_svm Classifier:\n")
        print(classification_report(y_test, lin_pred))
        poly_svm = SVC(kernel='poly', degree=8)
        poly_svm.fit(X_train,y_train)
        poly_pred = poly_svm.predict(X_test)
        print("Accuracy Of poly_svm : " + str(round(poly_svm.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of poly_svm Classifier:\n")
        print(confusion_matrix(y_test, poly_pred))
        print("\nCLassification Report of poly_svm Classifier:\n")
        print(classification_report(y_test, poly_pred))
        rbf_svm = SVC(kernel='rbf', degree=8)
        rbf_svm.fit(X_train, y_train)
        rbf_pred = rbf_svm.predict(X_test)
        print("Accuracy Of rbf_svm : " + str(round(rbf_svm.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of rbf_svm Classifier:\n")
        print(confusion_matrix(y_test, rbf_pred))
        print("\nCLassification Report of rbf_svm Classifier:\n")
        print(classification_report(y_test, rbf_pred))
        sig_svm = SVC(kernel='sigmoid', degree=8)
        sig_svm.fit(X_train, y_train)
        sig_pred = sig_svm.predict(X_test)
        print("Accuracy Of sig_svm : " + str(round(sig_svm.score(X_test, y_test), 3) * 100) + "%")
        print("\nConfusion Matrix of sig_svm Classifier:\n")
        print(confusion_matrix(y_test, sig_pred))
        print("\nCLassification Report of sig_svm Classifier:\n")
        print(classification_report(y_test, sig_pred))

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
        logging.info('feature extraction done')
if __name__ == '__main__':
    Compare()