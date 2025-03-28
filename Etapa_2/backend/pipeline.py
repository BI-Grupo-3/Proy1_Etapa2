from pipies import Preprocessing, Vectorizer, Model
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd


def createPipeline(data):

    pipeline = Pipeline([
        ('cleaner', Preprocessing(isTraining=True)),
        # ('vectorizer', Vectorizer(isTraining=True)),
        # ('model', Model())
    ])

    pipeline.fit(data)  
    dump(pipeline, './assets/model.joblib', compress=True)

if __name__ == "__main__":
    print("[Pipeline] Pipeline Started")
    df = pd.read_csv('fake_news_spanish.csv', sep = ';', encoding = 'utf-8')
    createPipeline(df)
    print("[Pipeline] Pipeline Finished")
