import nltk
import re
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords


class Feature:
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
            train.loc[index, 'total'] =train.loc[index, 'total']  + '()' + str(l) + '()' + str(len(row['title'])) + '()' +str(len(row['text']))
        return train
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