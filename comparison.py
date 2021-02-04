from FakeNewsDetection import fake_news_detection
if __name__ == '__main__':
    fk = fake_news_detection("file1.csv")
    data = fk.compare()
    a=data['accuracy']
    b = data['cmatrix']
    c =data['creport']
    print("Accuracy of LR :",a[0])
    print("Confusion Matrix of LR :\n",b[0])
    print("classification report LR :\n",c[0])
    print("Accuracy of PAC :", a[1])
    print("Confusion Matrix of PAC :\n", b[1])
    print("classification report PAC :\n", c[1])
    print("Accuracy of SVM :", a[2])
    print("Confusion Matrix of SVM :\n", b[2])
    print("classification report SVM :\n", c[2])
    print("Accuracy of Poly_SVM :", a[3])
    print("Confusion Matrix of Poly_SVM :\n", b[3])
    print("classification report Poly_SVM :\n", c[3])
    print("Accuracy of RBF_SVM :", a[4])
    print("Confusion Matrix of RBF_SVM :\n", b[4])
    print("classification report RBF_SVM :\n", c[4])
    print("Accuracy of RF :", a[5])
    print("Confusion Matrix of RF :\n", b[5])
    print("classification report RF :\n", c[5])
    print("Accuracy of Decision Tree :", a[6])
    print("Confusion Matrix of Decision Tree :\n", b[6])
    print("classification report Decision Tree :\n", c[6])


