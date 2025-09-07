import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from src.preprocessing import load_and_preprocess
from src.features import extract_features

MODEL_PATH = 'models/email_classifier_model.pkl'
DATA_PATH = 'data/sample_dataset.csv'

def train():
    df = load_and_preprocess(DATA_PATH)
    X = extract_features(df['text'], fit=True)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model trained. Accuracy: {clf.score(X_test, y_test):.2f}")
