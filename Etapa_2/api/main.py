from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware import cors
from pydantic import BaseModel
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"


class DataInstance(BaseModel):
    ID: str
    Titulo: str
    Descripcion: str
    Fecha: str


class PredictionInstance(DataInstance):
    Label: int


class PredictionRequest(BaseModel):
    instances: List[DataInstance]


class RetrainingRequest(BaseModel):
    instances: List[PredictionInstance]


app = FastAPI()

app.add_middleware(
    cors.CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    def save(self, filename=MODEL_PATH):
        joblib.dump(self.model, filename)

    @classmethod
    def load(cls, filename=MODEL_PATH):
        try:
            model = joblib.load(filename)
            instance = cls()
            instance.model = model
            return instance
        except FileNotFoundError as e:
            print(e)
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

        return predictions.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
async def retrain(request: RetrainingRequest):
    try:
        online_model = ModelWrapper.load()
        input_df = preprocess_data(request.instances)

        X = input_df.drop(columns=['Label'])
        y = input_df['Label']
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        online_model.fit(
            X_train,
            Y_train,
        )

        # online_model.save()
        predictions = online_model.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            Y_test,
            predictions["label"],
            average='weighted'
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "message": "Modelo exitósamente reentrenado"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
