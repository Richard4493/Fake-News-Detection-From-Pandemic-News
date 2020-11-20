import nltk
nltk.download('averaged_perceptron_tagger')
import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv('corona_fake_news.csv')
data = data.drop(['source'], axis = 1)
data.dropna(inplace=True)
data['total'] = data['title'] + ' ' + data['text']
data['title_word_count'] = data['title'].str.split().str.len()
data['text_word_count'] = data['text'].str.split().str.len()
for index, row in data.iterrows():
    title = row['title']
    count = 0
    for letter in title.split(" "):
        if(letter.isupper()):
            count+=1
    data.loc[index, 'capital_word_title_count'] = int(count)

data_final = data.drop(['title', 'text'], axis = 1)
from nltk.tokenize import word_tokenize
data_final['total'] = [entry.lower() for entry in data_final['total']]
data_final['total']= [word_tokenize(entry) for entry in data_final['total']]
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(data_final['total']):
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)

    data_final.loc[index,'total'] = str(Final_words)

data_final.total = data_final.total.astype(str)

Tfidf_vect = TfidfVectorizer(ngram_range=(2,2),max_features=10000)
Tfidf_vect.fit(data_final['total'])

pickle.dump(Tfidf_vect, open('tfidf.sav', 'wb'))
data_final.to_csv('file1.csv')
