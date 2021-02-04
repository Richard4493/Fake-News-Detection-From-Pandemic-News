from FakeNewsDetection import fake_news_detection
if __name__ == '__main__':
    fk = fake_news_detection("corona_fake_news.csv")
    title = input("Enter title : ")
    text = input("Enter text : ")
    print(fk.predict(text, title))