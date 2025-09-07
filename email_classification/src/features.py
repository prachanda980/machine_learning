from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

VECTOR_PATH = 'models/tfidf_vectorizer.pk'

def get_vectorizer():
    try:
        with open(VECTOR_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        vectorizer = TfidfVectorizer(max_features=1000)
    return vectorizer

def extract_features(texts, fit=False):
    vectorizer = get_vectorizer()
    if fit:
        features = vectorizer.fit_transform(texts)
        with open(VECTOR_PATH, 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        features = vectorizer.transform(texts)
    return features
