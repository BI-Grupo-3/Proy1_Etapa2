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

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


datos = pd.read_csv('fake_news_spanish.csv', sep = ';', encoding = 'utf-8')
data = datos.copy()

# class Preprocessing():
#     def __init__(self, isTraining = False):
#         self.data = data
#         self.isTraining = isTraining
        
#     def remove_non_ascii(self, words):
#         new_words = []
#         for word in words:
#             if word is not None:
#                 new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#                 new_words.append(new_word)
#         return new_words

#     def to_lowercase(self, words):
#         new_words = []
#         for word in words:
#             if word is not None:
#                 new_word = word.lower()
#                 new_words.append(new_word)
#         return new_words
                
#     def remove_punctuation(self, words):
#         new_words = []
#         for word in words:
#             if word is not None:
#                 new_word = re.sub(r'[^\w\s]', '', word)
#                 if new_word != '':
#                     new_words.append(new_word)
#         return new_words

#     def replace_numbers(self, words):
#         p = inflect.engine()
#         new_words = []
#         for word in words:
#             if word.isdigit():
#                 new_word = p.number_to_words(word)
#                 new_words.append(new_word)
        
#             else:
#                 new_words.append(word)
#         return new_words

#     def remove_stopwords(self, words):
#         """Remove stop words from list of tokenized words"""
#         new_words = []
#         stop_words = set(stopwords.words('spanish'))

#         for w in words:
#             if w not in stop_words:
#                 new_words.append(w)
#         return new_words

#     def preprocessing(self, words):
#         words = to_lowercase(words)
#         words = replace_numbers(words)
#         words = remove_punctuation(words)
#         words = remove_non_ascii(words)
#         words = remove_stopwords(words)
#         return words

#     def stem_words(words):
#         stemmer = LancasterStemmer()
#         stems = []
#         for word in words:
#             stem = stemmer.stem(word)
#             stems.append(stem)
#         return stems

#     def lemmatize_verbs(words):
#         lemmatizer = WordNetLemmatizer()
#         lemmas = []
#         for word in words:
#             lemma = lemmatizer.lemmatize(word, pos='v')
#             lemmas.append(lemma)
#         return lemmas

#     def stem_and_lemmatize(words):
#         stems = stem_words(words)
#         lemmas = lemmatize_verbs(words)
#         return stems + lemmas


#     def preprocesamiento_completo(data):
#         data.dropna()
#         data['Titulo'] = data['Titulo'].fillna(' ')
#         data = data.drop_duplicates(subset = ['Titulo', 'Descripcion'], keep = 'first')
#         data['Fecha'] = pd.to_datetime(data['Fecha'], errors='coerce')
#         data = data.drop(columns='ID')
#         data['Descripcion'] = data['Descripcion'].apply(contractions.fix)
#         data['Titulo'] = data['Titulo'].apply(contractions.fix)
#         data['words_descripcion'] = data['Descripcion'].apply(word_tokenize)
#         data['words_titulo'] = data['Titulo'].apply(word_tokenize) 
#         data['words_descripcion'].dropna()
#         data['words_titulo'].dropna()
#         data['prep_descripcion'] = data['words_descripcion'].apply(preprocessing)
#         data['prep_titulo'] = data['words_titulo'].apply(preprocessing)
#         data['prep_descripcion'] = data['prep_descripcion'].apply(stem_and_lemmatize) 
#         data['prep_titulo'] = data['prep_titulo'].apply(stem_and_lemmatize)
#         data['prep_descripcion'] = data['prep_descripcion'].apply(lambda x: ' '.join(map(str, x)))
#         data['prep_titulo'] = data['prep_titulo'].apply(lambda x: ' '.join(map(str, x)))
#         data["concatenado"] = data["prep_titulo"] + " " + data["prep_descripcion"]
        
#         return data['concatenado', 'Label']
    
    
#     def fit(self, data, target=None):
#         self.df = data
#         if self.isTraining:
           
#             self.df['Descripcion'] = data['Descripcion'].apply(self.preprocessing)
  
#         print('[CleaningTrain] Fitting Finished!!')
#         return self
    
#     def transform(self, data):
#         del self.df
#         self.df = data
#         self.df['Descripcion'] = data['Descripcion'].apply(self.preprocessing)
#         print('[CleaningTrain] Transformation Finished!!')
#         return self.df
    
#     def predict(self, data):
#         return self



from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessing(BaseEstimator, TransformerMixin):
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

    def preprocessing(self, text:str):
        words = word_tokenize(text)
        words = self.to_lowercase(words)
        words = self.replace_numbers(words)
        words = self.remove_punctuation(words)
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        return ' '.join(words)

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
        print('[CleaningTrain] Fitting Finished!!')
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

        print('[CleaningTrain] Transformation Finished!!')
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
        vectorizer = TfidfVectorizer(max_features=1000)
        vector = vectorizer.fit_transform(data['concatenado'])
        vectorizer.get_feature_names_out()
        vect_score = np.asarray(vector.mean(axis=0)).ravel().tolist()
        vect_array = pd.DataFrame({'term': vectorizer.get_feature_names_out(), 'weight': vect_score})
        vect_array.sort_values(by='weight',ascending=False,inplace=True)
        return vect_array

    def setImpact(self, df):
        df0 = df[df['Label'] == 0]
        df1 = df[df['Label'] == 1]
 

        self.impact3 = self.getVectorWeights(df0)
        self.impact4 = self.getVectorWeights(df1)
   
    
    def fit(self, data , target = None):
        self.setImpact(data)
        X =  self.vectorizer.fit_transform(data['concatenado'])
        self.data = pd.DataFrame(X.todense())
        self.data['Label'] = data['Label']
        print('[Vectorizer] Fitting Finished!!')
        return self

    def transform(self, data):
        self.vector = self.vectorizer.transform(data['concatenado'])
        transformed_data = pd.DataFrame(self.vector, columns=self.vectorizer.get_feature_names_out())
        if self.isTraining:
            transformed_data['Label'] = data['Label'].values
        print('[Vectorizer] Transformation Finished!!')
        return self.vector
        
    def predict(self, data):
        return self 
    
# class Model():

#     def __init__(self):
#         self.model = LogisticRegression(C=1, max_iter=1000, penalty='l1', solver='saga', warm_start='True')
#         self.precision = None
#         self.recall = None
#         self.report = None
#         self.f1 = None
    
#     def fit(self, data, target=None):
#         Y = data['Label']
#         X = data.drop(['Label'], axis = 1)
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#         self.model.fit(X_train, Y_train)
#         Y_test_predict = self.model.predict(X_test)
#         self.report = classification_report(Y_test, Y_test_predict)
#         self.f1 = f1_score(Y_test, Y_test_predict, average='weighted')
#         self.recall = recall_score(Y_test, Y_test_predict, average='weighted')
#         self.precision = precision_score(Y_test, Y_test_predict, average='weighted')
#         print('[Model] Modelo Entrenado')
#         return self
    
#     def transform(self, data):
#         return data
    
#     def predict(self, data):
#         labels = self.model.predict(data)
#         probabilities = self.model.predict_proba(data)
#         prediction = pd.DataFrame(labels, columns=['label'])
#         for i in range(probabilities.shape[1]):
#             prediction[f'prob_class_{i}'] = probabilities[:, i]
#         print('[Model] Predicciones Realizadas')
#         return prediction

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score

class Model:
    def __init__(self):
        self.model = LogisticRegression(
            C=1, max_iter=1000, penalty='l1', solver='saga', warm_start=True
        )
        self.precision = None
        self.recall = None
        self.report = None
        self.f1 = None

    def fit(self, X, y=None):
        # y puede venir aparte o estar en una columna separada en el pipeline anterior
        if y is None:
            raise ValueError("El target (y) no puede ser None")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        self.report = classification_report(y_test, y_pred)
        self.f1 = f1_score(y_test, y_pred, average='weighted')
        self.recall = recall_score(y_test, y_pred, average='weighted')
        self.precision = precision_score(y_test, y_pred, average='weighted')
        print('[Model] Modelo Entrenado')
        return self

    def transform(self, data):
        return data

    def predict(self, X):
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        prediction = pd.DataFrame(labels, columns=['label'])
        for i in range(probabilities.shape[1]):
            prediction[f'prob_class_{i}'] = probabilities[:, i]
        print('[Model] Predicciones Realizadas')
        return prediction
