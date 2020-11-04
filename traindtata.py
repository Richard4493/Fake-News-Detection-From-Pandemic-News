from sklearn.pipeline import Pipeline
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
class TrainData():
    def __init__(self):
        train_data = pd.read_csv('corona_fake1.csv')
        train_data = train_data.fillna(' ')
        train_data['total'] = train_data['title'] + ' ' + train_data['text']
        train_data = train_data[['total', 'label']]
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for index, row in train_data.iterrows():
            sentence = ''
            sentence = row['total']
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for words in words:
                sentence = sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()
            train_data.loc[index,'total'] =sentence

        train_data = train_data[['total', 'label']]
        Y_train = train_data['label']
        X_train  = train_data['total']
        x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train,random_state=1)
        piping = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', DecisionTreeClassifier())])
        model = piping.fit(x_train,y_train)
        print("Training Completed")
        fake_news_model = open('Trained_Model.sav', 'wb')
        pickle.dump(model, fake_news_model)
        fake_news_model.close()
if __name__ == '__main__':
    TrainData()