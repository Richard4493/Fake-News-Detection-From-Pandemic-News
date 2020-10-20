from sklearn.pipeline import Pipeline
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
class PassiveModel():
    def __init__(self):
        learn = pd.read_csv('corona_fake.csv')
        learn = learn.fillna(' ')
        learn['total'] = learn['title'] + ' ' + learn['text']
        learn = learn[['total', 'label']]
        stop_words = stopwords.words('english')
        lemmatizer = WordNetLemmatizer()
        for index, row in learn.iterrows():
            filter_sentence = ''
            sentence = row['total']
            words = nltk.word_tokenize(sentence)
            words = [w for w in words if not w in stop_words]
            for words in words:
                filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()
            learn.loc[index,'total'] =filter_sentence

        learn = learn[['total', 'label']]
        Y_train = learn['label']
        X_train  = learn['total']
        x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train,random_state=1)
        pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', PassiveAggressiveClassifier())])
        model = pipe1.fit(x_train,y_train)
        accuracy=model.score(x_test,y_test)
        print("Accuracy : ")
        print(accuracy)
        Passiveaggressive_model = open('PAC_model.sav', 'wb')
        pickle.dump(model, Passiveaggressive_model)
        Passiveaggressive_model.close()
if __name__ == '__main__':
    PassiveModel()