import re
from sklearn.pipeline import Pipeline
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
class TrainModel():
    def __init__(self):
        train = pd.read_csv('corona_fake.csv')
        train = train.fillna(' ')
        train['total'] = train['title'] + ' ' + train['text']
        train = train[['total', 'label']]
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        train_data = []
        for index, row in train.iterrows():
            filter_sentence = ''
            sentence = row['total']
            sentence = re.sub(r'[^\w\s]', '', sentence)
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for words in words:
                filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()
                train.loc[index,'total'] =filter_sentence

        train = train[['total', 'label']]
        Y_train = train['label']
        X_train  = train['total']
        x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
        pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression())])
        model = pipe1.fit(x_train,y_train)
        print("Accuracy : " + str(round(model.score(x_test,y_test),3)*100) + "%")
        fake_news_model = open('fake_news_model.sav', 'wb')
        pickle.dump(model, fake_news_model)
        fake_news_model.close()
if __name__ == '__main__':
    TrainModel()