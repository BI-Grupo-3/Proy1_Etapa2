import pandas as pd
import numpy as np
import sys
import re, unicodedata
import contractions
import inflect
import joblib
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer


from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter
from wordcloud import WordCloud
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.base import BaseEstimator, TransformerMixin

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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
    
    def getVectorWeights(self, data):
        data = data.copy()
        data['concatenado'] = data['concatenado'].fillna('').str.strip()
        data = data[data['concatenado'] != '']

        if data.empty:
            raise ValueError("[Vectorizer] Todos los textos están vacíos después del preprocesamiento.")

        vectorizer = TfidfVectorizer(max_features=1000)
        vector = vectorizer.fit_transform(data['concatenado'])
        vect_score = np.asarray(vector.mean(axis=0)).ravel().tolist()
        vect_array = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'weight': vect_score})
        vect_array.sort_values(by='weight', ascending=False, inplace=True)
        return vect_array


    def setImpact(self, df):
        df0 = df[df['Label'] == 0]
        df1 = df[df['Label'] == 1]
 

        self.impact3 = self.getVectorWeights(df0)
        self.impact4 = self.getVectorWeights(df1)
   
    
    def fit(self, data, y=None):
        if y is not None:
            df = data.copy()
            df['Label'] = y
            self.setImpact(df)
        
        X = self.vectorizer.fit_transform(data['concatenado'])
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
        self.model = LogisticRegression(C=1, max_iter=1000, penalty='l1', solver='saga', warm_start= True)
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

