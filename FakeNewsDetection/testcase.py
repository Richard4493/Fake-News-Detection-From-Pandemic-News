import unittest
from  FakeNewsDetection import fake_news_detection
from .newfeatures import *
class TestMethods(unittest.TestCase):
    def test_prediction(self):
        fk = fake_news_detection("corona_fake_news.csv")
        title ="Due to the recent outbreak for the Coronavirus (COVID-19) the World Health Organization is giving away vaccine kits. Just pay $4.95 for shipping"
        text ="You just need to add water, and the drugs and vaccines are ready to be administered. There are two parts to the kit: one holds pellets containing the chemical machinery that synthesises the end product, and the other holds pellets containing instructions that telll the drug which compound to create. Mix two parts together in a chosen combination, add water, and the treatment is ready."
        self.assertEqual(fk.predict(text, title),"Fake")