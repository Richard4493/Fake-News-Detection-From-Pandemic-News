from FakeNewsDetection import fake_news_detection
if __name__ == '__main__':
    fk = fake_news_detection("corona_fake_news.csv")
    fk.train()
    data = fk.compare()
    a1,a2,a3=data['accuracy']
    b1,b2,b3=data['cmatrix']
    c1,c2,c3=data['creport']
    print("Accuracy of LR :",a1)
    print("Confusion Matrix of LR :\n",b1)
    print("classification report LR :\n",c1)
    print("Accuracy of SVM :", a2)
    print("Confusion Matrix of SVM :\n", b2)
    print("classification report SVM :\n", c2)
    print("Accuracy of RF :", a3)
    print("Confusion Matrix of RF :\n", b3)
    print("classification report RF :\n", c3)

