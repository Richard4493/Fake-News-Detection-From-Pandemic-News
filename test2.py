import unittest
from  FakeNewsDetection import fake_news_detection
class TestLib(unittest.TestCase):

    def test_AccuracyCheck(self):
        fnd = fake_news_detection("corona_fake_news.csv")
        data = fnd.compare()
        a1, a2, a3 = data['accuracy']
        self.assertGreaterEqual(a1 , 95.1)
        self.assertGreaterEqual(a2 , 94.0)
        self.assertGreaterEqual(a3 , 90.5)
if __name__ == '__main__':
    unittest.main()