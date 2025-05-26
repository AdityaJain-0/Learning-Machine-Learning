import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def load_data(path):
    df = pd.read_csv(path)
    X = df[["size"]]   # Features
    y = df["price"]    # Target
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)