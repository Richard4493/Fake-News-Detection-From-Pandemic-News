from sklearn.pipeline import Pipeline
import re
import nltk
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
class Comparison():
    def _prepro(self,data):  #modified the code to create a function for preprocessing the news data(text data)
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for index, row in data.iterrows():
            filter_sentence = ''
            sentence = row['total']
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for words in words:
                filter_sentence = filter_sentence  + ' ' +str(lemmatizer.lemmatize(words)).lower()
            data.loc[index, 'total'] = filter_sentence
        return data
    def __init__(self):
        try:                       #File input exception handling
            data = pd.read_csv('corona_fake.csv')
        except IOError:
            print("Error: can\'t find file or read data")
            exit()
        data = data.fillna(' ')
        logging.info('Taken CSV file as input')
        data['total'] = data['title'] + ' ' + ' ' + data['text']
        data = data[['total', 'label']]
        data = self._prepro(data)      #calling preprocessing function-cleaning the text input.
        logging.info('Preprocessing done')
        data = data[['total', 'label']]
        X_train = data['total']
        Y_train = data['label']
        x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=0)
        #Logistic Regression Model Using pipelline
        pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression(C=5.0,random_state=0))])
        logreg = pipe1.fit(x_train, y_train)
        logging.info('Logistic Regression Implemented')
        print("Accuracy of Logistic Regression : " + str(round(logreg.score(x_test, y_test), 3) * 100) + "%")
        #Decision tree model using pipeline
        pipe2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', DecisionTreeClassifier())])
        dt = pipe2.fit(x_train,y_train)
        logging.info('Decision Tree Implemented')
        print("Accuracy of decision tree : " + str(round(dt.score(x_test, y_test), 3) * 100) + "%")
        #PAC model training using pipeline
        pipe3 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', PassiveAggressiveClassifier(max_iter=50))])
        pac = pipe3.fit(x_train, y_train)
        logging.info('PAC Implemented')
        print("Accuracy of PAC : " + str(round(pac.score(x_test, y_test), 3) * 100) + "%")

if __name__ == '__main__':
    Comparison()