import re
import nltk
import pandas as pd
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



class Main:
    def __init__(self):
        train = pd.read_csv('corona_fake1.csv')
        logging.info("Read dataset")
        train = train.fillna(' ')
        train['total'] = train['title'] + ' ' + train['text']
        train = train[['total', 'label']]
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
        logging.info("Lemetizing")
        train = train[['total', 'label']]
        X_train = train['total']
        Y_train = train['label']
        tfidf = TfidfVectorizer(strip_accents=None,
                                lowercase=False,
                                preprocessor=None,
                                use_idf=True,
                                norm='l2',
                                smooth_idf=True)
        tf_idf_matrix = tfidf.fit_transform(X_train)
        X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, Y_train, random_state=0)
        Lr = LogisticRegression(C=10.0, random_state=0)
        Lr.fit(X_train, y_train)
        logging.info("Training data")

        print("Accuracy : " + str(round(Lr.score(X_test, y_test), 3) * 100) + "%")


if __name__ == '__main__':
    Main()
