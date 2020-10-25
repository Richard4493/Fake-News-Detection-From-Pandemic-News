import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
data = pd.read_csv('corona_fake.csv')
data = data.fillna(' ')
data['total'] = data['title'] + ' ' + ' ' + data['text']
data = data[['total', 'label']]
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

data = data[['total', 'label']]
X_train = data['total']
Y_train = data['label']
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix,Y_train, random_state=0)
logreg = LogisticRegression(C=5.0,random_state=0)
logreg.fit(X_train, y_train)
Accuracy1 = logreg.score(X_test, y_test)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
Accuracy2 = clf.score(X_test, y_test)
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train,y_train)
Accuracy3 = pac.score(X_test, y_test)
print("Accuracy of Logistic Regression Method is :")
print(Accuracy1)
print("Accuracy of Decision Tree Method is :")
print(Accuracy2)
print("Accuracy of PassiveAggressiveClassifier is :")
print(Accuracy3)

