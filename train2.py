import pandas as pd
import pickle
data = pd.read_csv('file1.csv')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
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

Lr = LogisticRegression(C=5.0, random_state=0)
Lr.fit(X_train, y_train)
lr_pred = Lr.predict(X_test)
print("Accuracy Of LR : " + str(accuracy_score(y_test, lr_pred)) + "%")
print("\nConfusion Matrix of Logistic Regression Classifier:\n")
print(confusion_matrix(y_test, lr_pred))
print("\nCLassification Report of Logistic Regression Classifier:\n")
print(classification_report(y_test, lr_pred))


SVM = LinearSVC(random_state=0)
SVM.fit(X_train, y_train)
SVM_pred = SVM.predict(X_test)

print("Accuracy Of SVM : " + str(accuracy_score(y_test, SVM_pred)) + "%")
print("\nConfusion Matrix of SVM Classifier:\n")
print(confusion_matrix(y_test, SVM_pred))
print("\nCLassification Report of SVM Classifier:\n")
print(classification_report(y_test, SVM_pred))



RF = RandomForestClassifier(random_state=0)
RF.fit(X_train, y_train)
RF_pred = RF.predict(X_test)

print("Accuracy Of RF : " + str(accuracy_score(y_test, RF_pred)) + "%")
print("\nConfusion Matrix of RF Classifier:\n")
print(confusion_matrix(y_test, RF_pred))
print("\nCLassification Report of RF Classifier:\n")
print(classification_report(y_test, RF_pred))