# xgboost.py
import joblib

def predict(X_numpy):
    model = joblib.load("models/best_xgboost_model.pkl")
    return model.predict(X_numpy)