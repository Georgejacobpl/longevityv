from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

# Load model
with open('model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

@app.post("/predict")
async def predict(request: PredictionRequest):
    prediction = model.predict(vectorizer.transform([request.text]))[0]
    confidence = model.predict_proba(vectorizer.transform([request.text])).max()
    
    return {
        "sentiment": prediction,
        "confidence": float(confidence),
        "text": request.text
    }

@app.get("/")
async def root():
    return {"message": "Supplement Analyzer API"}
