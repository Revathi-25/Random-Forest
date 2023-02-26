import spacy
import string
import re
import pandas as pd
import numpy as np
from typing import List
from model import load_model, load_vec

class Model(object):

    def __init__(self, model, vec, nlp, stop_words, punc):
        self.model = model
        self.vec = vec
        self.nlp = nlp
        self.stop_words  = stop_words
        self.punc = punc
        self.observation = None
        self.labels = {'avg': 0, 'excellent': 1, 'high': 2, 'low': 3}

        

    def __repr__(self):
        print(f'Data passed : {self.observation}\n Model: {self.model}')

    def dataPreProcessing(self, x)-> List[List]:
        try:
            assert type(x) == str

            x = x.lower()[12:]
            x = ''.join([i for i in x if i not in self.punc])
            x = ' '.join([i for i in x.split()])
            x = ' '.join([i.lemma_ for i in self.nlp(x) if i.text not in self.stop_words])
            x = re.sub(r'\d+', '', x)
            print(self.vec)
            tfidf_vec = self.vec.transform(np.array([x]))
            print(tfidf_vec.shape)
            return tfidf_vec

        except AssertionError:
            print('Text feature of mail is not a string. Make sure first two arguments are strings.')
    
    def dataTransformation(self):
        print(self.observation[0])
        tfidf_text = self.dataPreProcessing(self.observation[0])
        tfidf_subject = self.dataPreProcessing(self.observation[1])
        self.observation = self.observation[2:]
        data = self.observation + tfidf_text.squeeze().tolist() + tfidf_subject.squeeze().tolist()
        return data
    
    def model_predict(self, observation):
        self.observation = observation
        data = self.dataTransformation()
        prediction = self.model.predict(data)
        return prediction
    
if __name__ == '__main__':
    model = load_model('backend/models/random_forest_model.pkl')
    vec = load_vec('backend/models/tfidf.pkl')
    nlp = spacy.load('de_core_news_sm')
    punc = string.punctuation
    stop_words = nlp.Defaults.stop_words
    observation = ['[TAF-AREA /]------------------------------- Reflact AG https://www.reflact.com/ Browseransicht # ********************************* » Software & Services http://www.reflact.com/software-services/ » Consulting &amp; Training http://www.reflact.com/consulting-training-2/ » Adobe Solutions http://www.reflact.com/consulting-training/ » Über uns http://www.reflact.com/ueber-uns/ » Kontakt http://www.reflact.com/kontakt/ ================================ Die 5 wichtigsten Erfolgsgeheimnisse der Besten im Social Learning Viele Unternehmen kennen das Problem: Es ist und bleibt trotz aller Bemühungen schwierig, Just-in-Time hilfreiche Lerninhalte in ausreichender Anzahl und Geschwindigkeit zur Verfügung zu stellen, da die Überführung des Wissens in ein geeignetes Online Format oft schwer fällt. Wie schön wäre es, wenn die Experten im Unternehmen ihre Erfahrungen und ihr wertvolles Wissen einfach selbst mit den Lernern teilen könnten. Die Lösung: Social Learning mit User Generated Content. Adobe Captivate Prime, das Lernmanagement System von Adobe, unterstützt zukünftig genau diesen sozialen Lernansatz. Seien Sie gespannt auf die nächsten Produkt-Updates, zu denen wir Sie natürlich auf dem Laufenden halten! Vorab haben wir für Sie die 5 wichtigsten Erfolgsfaktoren zusammengestellt, die die besten Unternehmen im Social Learning vom Rest unterscheiden. » mehr erfahren: http://www.reflact.com/adb-erfolgsgeheimnisse-social-learning-2/ ------------------------------- Adobes neue Tool-Chain für digitales Lernen & "The Future of Learning Experience" Welche neuen Herausforderungen bringt der wirtschaftliche Gesamtkontext für L&D mit sich? Welche Schlüsse ziehen wir aus der aktuellen globalen Studie von Adobe "The Future of Learning Experience"? Dirk Schmitz stellt ausgewählte Erkenntnisse aus der Studie vor und leitet Strategien ab, wie Sie den neuen Herausforderungen begegnen können, z.B. mit der neu aufgestellten Adobe Tool-Chain für digitales Lernen. Während des kostenlosen Webinars beantworten wir zudem Fragen wie: Wo bekomme ich geeigneten Content her? Wie gestalte ich die Lernlogistik effizient, durchgängig und zukunftssicher? Wie motiviere ich die Lerner pragmatisch, effektiv und nachhaltig? » mehr erfahren und anmelden: https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3422332785 ------------------------------- 21. März Adobe Captivate Prime in 30 Minuten https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3338515897 Ein Überblick über das Lernmanagement System von Adobe | 16 Uhr 22. März Adobe Connect: Neues vom global Leader https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3461606089 Die nächste Generation virtueller Klassenräume | 11 Uhr 27. März Adobes neue Tool-Chain für digitales Lernen https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3422332785 & "The Future of Learning Experience" | 11 Uhr 28. März Ihre Learning Content Produktion im Griff https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3461994992 One-Stop Tools & Services rund um Adobe Captivate im neuen reflact Portal | 11 Uhr 04. April Adobe Connect in 30 Minuten https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3338508881 Das Wichtigste, das Sie wissen müssen | 16 Uhr 05. April Adobe Captivate Prime https://events-emea1.adobeconnect.com/content/connect/c1/977978177/en/events/event/shared/1257094219/event_landing.html?sco-id=3466447120 Warum die Lernwelt auf dieses LMS gewartet hat! | 11 Uhr http://www.reflact.com/ https://adobeelearning.eu/ https://adobeconnect.eu/ ------------------------------- --------- IMPRESSUM --------- Vorsitzender des Aufsichtsrats: Dr. Christian Knebel Vorstand: Hartmut Scholl Registergericht: Amtsgericht Duisburg Registernummer: HRB 13453 Ust. ID-Nr.: DE 212722522 Inhaltlich Verantwortlicher gemäß § 6 MDStV: Hartmut Scholl » Gesamtes Impressum http://www.reflact.com/impressum/ » Hinweis zum Datenschutz http://www.reflact.com/datenschutz/ » AGBs der reflact AG http://www.reflact.com/agbs/ » Newsletter abmelden # » Facebook: https://www.facebook.com/reflact » LinkedIn: https://www.linkedin.com/company/reflact-ag » Xing: https://www.xing.com/companies/reflactag » YouTube: https://www.youtube.com/user/reflactAG/ --------- KONTAKT --------- Reflact AG Technologiezentrum I Essener str. 3 D-46047 Oberhausen fon: +49 208 77899 700 fax: +49 208 77899 701 e-mail: kommunikation@reflact.com',
 '5 Erfolgsgeheimnisse im Social Learning und Adobes neue Tool-Chain  ', 2, 19, 15, 1, 15, 9, 4, 2, 3843, 3844, 256, 407, 2, 378, 1244, 68, 3844,
 0, 0, 32, 27, 51, 7690, 285, 151, 142, 75, 7162, 3737, 15, 0.004, 0.258363355635615, 1]
    rf = Model(model = model, vec = vec, nlp = nlp, stop_words=stop_words, punc = punc)
    print(rf.model_predict(observation))
