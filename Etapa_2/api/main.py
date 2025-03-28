from typing import List, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataInstance(BaseModel):
    ID: str
    Titulo: str
    Descripcion: str
    Fecha: str


class PredictionInstance(DataInstance):
    Label: str

class PredictionRequest(BaseModel):
    instances: List[DataInstance]

class RetrainingRequest(BaseModel):
    instances: List[PredictionInstance]
    target: List[Union[int, str]]

app = FastAPI()

class ModelWrapper:
    def __init__(self, base_model=None):
        self.model = base_model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def fit(self, X, y, classes=None):
        if classes is None:
            classes = np.unique(y)
        
        self.model.fit(X, y)
    
    def save(self, filename="models/best_model_log.pkl"):
        joblib.dump(self.model, filename)
    
    @classmethod
    def load(cls, filename="models/best_model_log.pkl"):
        try:
            model = joblib.load(filename)
            instance = cls()
            instance.model = model
            return instance
        except FileNotFoundError:
            return cls()


def preprocess_data(instances: List[DataInstance]) -> pd.DataFrame:
    df = pd.DataFrame([instance.model_dump() for instance in instances])
    
    return df

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        online_model = ModelWrapper.load()
        
        input_df = preprocess_data(request.instances)
        
        predictions = online_model.predict(input_df)
        prediction_proba = online_model.predict_proba(input_df)
        
        results = [
            {
                "ID": instance.ID,
                "prediction": pred, 
                "probability": max(proba)
            } 
            for instance, pred, proba in zip(request.instances, predictions, prediction_proba)
        ]
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain(request: RetrainingRequest):
    try:
        online_model = ModelWrapper.load()
        input_df = preprocess_data(request.instances)
        
        online_model.partial_fit(
            input_df, 
            request.target, 
            classes=np.unique(request.target)
        )
        
        online_model.save()
        predictions = online_model.predict(input_df)
        precision, recall, f1, _ = precision_recall_fscore_support(
            request.target, 
            predictions, 
            average='weighted'
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "message": "Model successfully updated with online learning"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))