import re
import sys
import unicodedata
from collections import Counter

import contractions
import inflect
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn import svm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import (CountVectorizer,
                                             HashingVectorizer,
                                             TfidfVectorizer)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC
from wordcloud import WordCloud

datos = pd.read_csv('fake_news_spanish.csv', sep = ';', encoding = 'utf-8')
data = datos.copy()

class Preprocesamiento(BaseEstimator, TransformerMixin):
    def __init__(self, isTraining=False):
        self.data = data
        self.isTraining = isTraining

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            if word is not None:
                new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
                new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        return [word.lower() for word in words if word is not None]

    def remove_punctuation(self, words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if word and re.sub(r'[^\w\s]', '', word) != '']

    def replace_numbers(self, words):
        p = inflect.engine()
        return [p.number_to_words(word) if word.isdigit() else word for word in words]

    def remove_stopwords(self, words):
        stop_words = set(stopwords.words('spanish'))
        return [w for w in words if w not in stop_words]

    def preprocessing(self, text: str):
        words = word_tokenize(text)
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return words  


    def stem_words(self, words):
        stemmer = LancasterStemmer()
        return [stemmer.stem(word) for word in words]

    def lemmatize_verbs(self, words):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word, pos='v') for word in words]

    def stem_and_lemmatize(self, words):
        stems = self.stem_words(words)
        lemmas = self.lemmatize_verbs(words)
        return stems + lemmas

    def fit(self, data, target=None):
        self.df = data.copy()
        print('Preprocesamiento - Fitting Finished!!')
        return self

    def transform(self, data):
        df = data.copy()
        df['Titulo'] = df['Titulo'].fillna('')
        df['Descripcion'] = df['Descripcion'].fillna('')

        df['Titulo'] = df['Titulo'].apply(contractions.fix)
        df['Descripcion'] = df['Descripcion'].apply(contractions.fix)

        df['tokens_titulo'] = df['Titulo'].apply(self.preprocessing)
        df['tokens_descripcion'] = df['Descripcion'].apply(self.preprocessing)

        df['tokens_titulo'] = df['tokens_titulo'].apply(self.stem_and_lemmatize)
        df['tokens_descripcion'] = df['tokens_descripcion'].apply(self.stem_and_lemmatize)

        df['prep_titulo'] = df['tokens_titulo'].apply(lambda x: ' '.join(x))
        df['prep_descripcion'] = df['tokens_descripcion'].apply(lambda x: ' '.join(x))

        df['concatenado'] = df['prep_titulo'] + ' ' + df['prep_descripcion']
        
    
        df['concatenado'] = df['concatenado'].fillna('').str.strip()
        df = df[df['concatenado'] != '']
        

        print('Preprocesamiento - Transformation Finished!!')
        return df[['concatenado', 'Label']] if 'Label' in df.columns else df[['concatenado']]



    def predict(self, data):
        return self

class Vectorizer:

    def __init__(self, isTraining = False):
        self.vectorizer = TfidfVectorizer()
        self.isTraining = isTraining
        self.vector = None
        self.data = None
        self.all_data = pd.DataFrame()
    
   
    
    def fit(self, data, y=None):
        
        """ if y is not None:
            df = data.copy()
            df['Label'] = y
            self.setImpact(df) """

        self.all_data = pd.concat([self.all_data, data], ignore_index=True)

        X = self.vectorizer.fit_transform(self.all_data['concatenado'])
        self.data = X  
        print('Vectorizer - Fitting Finished!!')
        return self
        
    def transform(self, data):
        self.vector = self.vectorizer.transform(data['concatenado'])

        print('Vectorizer - Transformation Finished!!')    
        return self.vector

        
        
    def predict(self, data):
        return self 
    
class Model():

    def __init__(self):
        self.model = LogisticRegression(C=1, max_iter=1000, penalty='l1', solver='saga', warm_start=True)
        self.precision = None
        self.recall = None
        self.report = None
        self.f1 = None
    
    
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Se necesita el target `y` en el modelo.")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.report = classification_report(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        self.precision = precision_score(y_test, y_pred, average='weighted')
        print('Modelo Entrenado')
        return self

    
    def transform(self, data):
        return data
    
    def predict(self, data):
        labels = self.model.predict(data)
        probabilities = self.model.predict_proba(data)
        prediction = pd.DataFrame(labels, columns=['label'])
        for i in range(probabilities.shape[1]):
            prediction[f'prob_class_{i}'] = probabilities[:, i]
        print('Modelo Predicciones Realizadas')
        return prediction

