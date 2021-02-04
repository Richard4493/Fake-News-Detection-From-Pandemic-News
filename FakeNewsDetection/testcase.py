import unittest
from  FakeNewsDetection import fake_news_detection
from .newfeatures import *
class TestMethods(unittest.TestCase):
    def test_prepro(self):
        X = pandas.DataFrame({"text" : ["From the evidence so far, the COVID-19 virus can be transmitted in ALL AREAS, including areas with hot and humid weather. Regardless of climate, adopt protective measures if you live in, or travel to an area reporting COVID-19. The best way to protect yourself against COVID-19 is by frequently cleaning your hands. By doing this you eliminate viruses that may be on your hands and avoid infection that could occur by then touching your eyes, mouth, and nose."] })
        data = WordLematization.tokenize(self,X.text)
        self.assertEqual(data ,[['from', 'the', 'evidence', 'so', 'far', ',', 'the', 'covid-19', 'virus', 'can', 'be', 'transmitted', 'in', 'all', 'areas', ',', 'including', 'areas', 'with', 'hot', 'and', 'humid', 'weather', '.', 'regardless', 'of', 'climate', ',', 'adopt', 'protective', 'measures', 'if', 'you', 'live', 'in', ',', 'or', 'travel', 'to', 'an', 'area', 'reporting', 'covid-19', '.', 'the', 'best', 'way', 'to', 'protect', 'yourself', 'against', 'covid-19', 'is', 'by', 'frequently', 'cleaning', 'your', 'hands', '.', 'by', 'doing', 'this', 'you', 'eliminate', 'viruses', 'that', 'may', 'be', 'on', 'your', 'hands', 'and', 'avoid', 'infection', 'that', 'could', 'occur', 'by', 'then', 'touching', 'your', 'eyes', ',', 'mouth', ',', 'and', 'nose', '.']])
        self.assertEqual(WordLematization.lemmatizerNew(self ,data) ,["['evidence', 'far', 'virus', 'transmit', 'area', 'include', 'area', 'hot', 'humid', 'weather', 'regardless', 'climate', 'adopt', 'protective', 'measure', 'live', 'travel', 'area', 'report', 'best', 'way', 'protect', 'frequently', 'clean', 'hand', 'eliminate', 'virus', 'may', 'hand', 'avoid', 'infection', 'could', 'occur', 'touch', 'eye', 'mouth', 'nose']"])
    def test_prediction(self):
        fk = fake_news_detection("corona_fake_news.csv")
        title ="Due to the recent outbreak for the Coronavirus (COVID-19) the World Health Organization is giving away vaccine kits. Just pay $4.95 for shipping"
        text ="You just need to add water, and the drugs and vaccines are ready to be administered. There are two parts to the kit: one holds pellets containing the chemical machinery that synthesises the end product, and the other holds pellets containing instructions that telll the drug which compound to create. Mix two parts together in a chosen combination, add water, and the treatment is ready."
        self.assertEqual(fk.predict(text, title),"Fake")

    def test_newFeatures(self):
        X = pandas.DataFrame({"text": [
            "From the evidence so far, the COVID-19 virus can be transmitted in ALL AREAS, including areas with hot and humid weather. Regardless of climate, adopt protective measures if you live in, or travel to an area reporting COVID-19. The best way to protect yourself against COVID-19 is by frequently cleaning your hands. By doing this you eliminate viruses that may be on your hands and avoid infection that could occur by then touching your eyes, mouth, and nose."]})
        data = WordCountExtractor.word_count(self, X.text)
        self.assertEqual(data.values, [78])
        self.assertEqual(CapitalWordCountExtractor.title_capital_word_count(self, X.text), [5])