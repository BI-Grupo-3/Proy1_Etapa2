from pipies import Preprocesamiento, Vectorizer, Model
from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd


def createPipeline(data):

    pipeline = Pipeline([
        ('cleaner', Preprocesamiento(isTraining=True)),
        ('vectorizer', Vectorizer(isTraining=True)),
        ('model', Model())
    ])

    X = data.drop(columns=['Label'])  
    y = data['Label'] 
    pipeline.fit(X, y)
    
    dump(pipeline, 'model.joblib', compress=True)

if __name__ == "__main__":
    print(" Pipeline Started")
    df = pd.read_csv('fake_news_spanish.csv', sep = ';', encoding = 'utf-8')
    createPipeline(df)
    print(" Pipeline Finished")
