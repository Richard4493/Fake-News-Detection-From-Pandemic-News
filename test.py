from FakeNewsDetection import fake_news_detection
if __name__ == '__main__':
    fk = fake_news_detection()
    fk.train("corona_fake1.csv")