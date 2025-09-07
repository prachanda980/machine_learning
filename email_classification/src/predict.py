import pickle
from src.features import extract_features
from src.preprocessing import clean_text

MODEL_PATH = 'models/email_classifier_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def predict_email(email_text):
    model = load_model()
    text = clean_text(email_text)
    features = extract_features([text], fit=False)
    return int(model.predict(features)[0])
