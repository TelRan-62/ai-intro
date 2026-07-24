from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy
import re


class TextModel:
    def __init__(self, text: str):
        text_arr = re.split(r'[.!?…]', text)
        self.text_model = [res for s in text_arr if (res := s.strip())]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_text = self.vectorizer.fit_transform(self.text_model)

    def get_answers(self, query: str, n_answers: int):
        tfidf_query = self.vectorizer.transform([query])
        sims = cosine_similarity(self.tfidf_text, tfidf_query)
        sims_flat = sims.ravel()
        indices = numpy.argsort(sims_flat)[::-1]  # reverse order
        res = [self.text_model[i] for i in indices[:n_answers] if sims_flat[i]]
        return res
