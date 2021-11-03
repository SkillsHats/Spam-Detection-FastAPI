import re
import numpy as np
import pandas as pd

from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from utils import preprocessor


# Read the Data
data = pd.read_csv('./data/spam_data.csv')

X = data['Message'].apply(preprocessor)
y = data['Category']


# Train, Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Training a Neural Network Pipeline
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, 
                        max_features=700, 
                        ngram_range=(1,1))
neural_net_pipeline = Pipeline([('vectorizer', tfidf), 
                                ('nn', MLPClassifier(hidden_layer_sizes=(700, 700)))])

neural_net_pipeline.fit(X_train, y_train)

# Testing the Pipeline

y_pred = neural_net_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy: {} %'.format(100 * accuracy_score(y_test, y_pred)))


# Saving the Pipeline
dump(neural_net_pipeline, 'models/spam_classifier.joblib')
