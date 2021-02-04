import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
class main:
    html_temp = """
               	<div style="background-color:black;padding:10px">
               	<h1 style="color:white;text-align:left;">FAKE NEWS DETECTION </h1>
               	</div>
               	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    # st.subheader("ML App with Streamlit")
    html_temp = """
       	<div style="background-color:white;padding:5px">
       	<h2 style="color:black;text-align:center;">Streamlit ML App </h2>
       	</div>
       	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    data = pd.read_csv('file1.csv')
    Tfidf_vect = pickle.load(open('tfidf.sav', 'rb'))

    X = Tfidf_vect.transform(data['total'])

    ls = []
    for i in X.toarray():
        ls.append(list(i))
    df = pd.DataFrame(ls)
    dataset = data.join(df)

    dataset.dropna(inplace=True)
    dataset = dataset.drop(dataset.columns[0], axis=1)
    dataset = dataset.drop(['total'], axis=1)

    X = dataset.drop(['label'], axis=1)
    y = dataset['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=0)
    st.markdown("""
    <style>
    title.title {
        color: #fff;
        background-color: #111;
    }
    </style>
        """, unsafe_allow_html=True)
    st.title("Fake Data Detection")
    st.sidebar.write("Dataset used : covid_news Dataset")
    claasifier_name = st.sidebar.selectbox("Select The classifier", ("logistic regression", "lin svm","poly svm","random forest","decision tree"))
    def get_classifier(clf_name):
        if clf_name=="logistic regression":
            clf = LogisticRegression(C=5.0, random_state=0)
        elif clf_name == "lin svm":
            clf = LinearSVC(random_state=0)
        elif clf_name=="poly svm":
            clf = SVC(kernel='poly', degree=8)
        elif clf_name == "random forest":
            clf = RandomForestClassifier(random_state=0)
        else :
            clf= DecisionTreeClassifier(random_state=0)
        return clf
    opt=st.sidebar.checkbox("CLick to view the accuracy of the classifier")
    if opt==True:
        clf = get_classifier(claasifier_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc=accuracy_score(y_test, y_pred)
        per=acc*100
        st.write(f"classifier={claasifier_name}")
        st.write(f"accuracy = {per}")
    option=st.sidebar.checkbox("Do you want to see the accuracy graph")
    # st.write("You have selected:",option)
    if option==True:
        #logistic regression
        Lr = LogisticRegression(C=5.0, random_state=0)
        Lr.fit(X_train, y_train)
        lr_pred = Lr.predict(X_test)
        lr_acc=accuracy_score(y_test,lr_pred)
        lr_per=lr_acc*100
        # linear svm
        lin_svm = LinearSVC(random_state=0)
        lin_svm.fit(X_train, y_train)
        lin_pred = lin_svm.predict(X_test)
        lin_acc=accuracy_score(y_test,lin_pred)
        lin_per=lin_acc*100
        #poly svm
        poly_svm = SVC(kernel='poly', degree=8)
        poly_svm.fit(X_train, y_train)
        poly_pred = poly_svm.predict(X_test)
        poly_acc=accuracy_score(y_test,poly_pred)
        poly_per=poly_acc*100
        #random forest
        RF = RandomForestClassifier(random_state=0)
        RF.fit(X_train, y_train)
        RF_pred = RF.predict(X_test)
        RF_acc=accuracy_score(y_test,RF_pred)
        RF_per=RF_acc*100
        # decision tree
        DT = DecisionTreeClassifier(random_state=0)
        DT.fit(X_train, y_train)
        DT_pred = DT.predict(X_test)
        DT_acc=accuracy_score(y_test,DT_pred)
        DT_per=DT_acc*100

        x=["logistic regression", "lin svm","poly svm","random forest","decision tree"]
        h=[lr_per,lin_per,poly_per,RF_per,DT_per]
        plt.figure()
        plt.bar(x,h,.5)
        plt.xlabel("Classifier")
        plt.ylabel("Accuracy")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
if __name__ == '__main__':
    main()