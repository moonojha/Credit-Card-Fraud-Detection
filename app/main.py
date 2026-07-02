from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title='Credit Card Fraud Detection API')
model = joblib.load("models/xgb_model.pkl")
sample_ds = pd.read_csv('data/test_sample.csv')

class Transaction(BaseModel):
  features:list[float]

@app.get("/sample_transaction")
def get_sample():
  row = sample_ds.sample(1)
  return row.to_dict(orient='records')[0]

@app.post("/predict")
def prediction(transaction: Transaction):
  X = np.array(transaction.features).reshape(1,-1)
  prob = model.predict_proba(X)[0][1]
  pred = int(prob >= 0.15)
  return {"prob_fraud": float(prob), "is_fraud": pred}

@app.get("/health")
def health_check():
  return {"status": "good"}