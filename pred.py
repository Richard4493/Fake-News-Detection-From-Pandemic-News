import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
news = 'fake_news_model.sav'
log = pickle.load(open(news, 'rb'))
title1 = input ("Enter title :")
text1 = input("Enter text : ")
total1= title1 + ' ' + text1
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
filter_sentence = ' '
sentence = total1
sentence = re.sub(r'[^\w\s]', '', sentence)
# Tokenization
words = nltk.word_tokenize(sentence)
words = [w for w in words if not w in stop_words]
# Lemmatization
for words in words:
    filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(words)).lower()
x_predict = [filter_sentence]
result=log.predict(x_predict)
print(result)