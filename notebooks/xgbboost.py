import nltk
import re
import pandas as pd
import numpy as np
from enum import Enum
from scipy.sparse import hstack

from gensim.models import Word2Vec

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from nltk import pos_tag, ne_chunk, Tree
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from textblob import TextBlob
import spacy

df = pd.read_csv('./raw_data/fulltrain.csv', header=None, names=['Verdict', 'Text'])
df_test = pd.read_csv('./raw_data/balancedtest.csv', header=None, names=['Verdict', 'Text'])

X_train = df['Text']
y_train = df['Verdict']
X_test = df_test['Text']
y_test = df_test['Verdict']

def lemmatize(text):
  lemmatizer = WordNetLemmatizer()
  words = word_tokenize(text)
  return ' '.join([lemmatizer.lemmatize(word) for word in words])

X_train = X_train.apply(lemmatize)
X_test = X_test.apply(lemmatize)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=10, random_state=42))  # Adjust XGBoost parameters as needed
])

# Define the parameter grid
param_grid = {
    'xgb__n_estimators': [200, 300, 400],
    'xgb__max_depth': [10, 12, 14],
    'xgb__learning_rate': [0.3, 0.5, 0.7]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, n_jobs=5)

y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1

# Train the model on the training data
grid_search.fit(X_train, y_train_adjusted)

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)

# Evaluate the model on the test data
predictions = grid_search.predict(X_test)
with open('classification_report.txt', 'w') as f:
    f.write(classification_report(y_test_adjusted, predictions))