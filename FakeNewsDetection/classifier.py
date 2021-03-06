from sklearn.linear_model import LogisticRegression
from .newfeatures import *
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
def text(X):
     return X.title
def title(X):
    return X.text
def get_models():
    models = {"LR" : ('LR',LogisticRegression()) ,"PAC" : ('PAC',PassiveAggressiveClassifier()),"SVM" : ('SVM',LinearSVC()) ,"PSVM"  : ('PSVM' , SVC(kernel='poly', degree=8)),"RBFSVM"  : ('RBFSVM' , SVC(kernel='rbf', degree=8)),"RF"  : ('RF' , RandomForestClassifier()),"DT"  : ('DT' , DecisionTreeClassifier())}
    m = {}
    pipe_text = Pipeline([('col_text', FunctionTransformer(text, validate=False))])
    pipe_title = Pipeline([('col_title', FunctionTransformer(title, validate=False))])
    feature_text =  FeatureUnion([
                                                       ("vectorizer" , Pipeline([("word_lemetization", WordLematization()),('tfidf', TfidfVectorizer(max_features=2000))])),
                                                        ('ave', WordCountExtractor())


                                                    ])
    feature_title = FeatureUnion([
                                                    ("vectorizer" , Pipeline([("word_lemetization", WordLematization()),('tfidf', TfidfVectorizer(max_features=2000))])),
                                                    ('ave', WordCountExtractor()) ,
                                                    ("capital_word_count" , CapitalWordCountExtractor())

                                                ])
    pipe1 = Pipeline(pipe_text.steps + [("newfeat1"  , feature_text)])
    pipe2 = Pipeline(pipe_title.steps + [("newfeat2"  , feature_title)])
    vectorizer = FeatureUnion([('text', pipe1), ('title', pipe2)])
    for name , model in models.items() :
        m[name] = Pipeline([('feats' , vectorizer)
                                        , model
                                    ])
    return m