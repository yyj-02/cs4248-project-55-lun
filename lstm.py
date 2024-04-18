import nltk
import re
import pandas as pd
import numpy as np
from enum import Enum
from scipy.sparse import hstack

from gensim.models import Word2Vec, KeyedVectors

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


from nltk import pos_tag, ne_chunk, Tree
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from textblob import TextBlob
import spacy

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('./raw_data/fulltrain.csv', header=None, names=['Verdict', 'Text'])
df_test = pd.read_csv('./raw_data/balancedtest.csv', header=None, names=['Verdict', 'Text'])

X_train = df['Text']
y_train = df['Verdict']
X_test = df_test['Text']
y_test = df_test['Verdict']

# Adjust target values to start from 0
y_train_adjusted = y_train - 1
y_test_adjusted = y_test - 1

# Tokenizer and sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index

print(sequences_train)

# Load Google News Word2Vec model
word_vectors = Word2Vec(vector_size=300, min_count=1)
word_vectors.build_vocab(sequences_train)
word_vectors.wv.intersect_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, lockf=1.0)
word_vectors.train(sequences_train, total_examples=word_vectors.corpus_count, epochs=word_vectors.epochs)

# Prepare embedding matrix
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word_vectors:
        embedding_vector = word_vectors.get_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)
print(np.sum(np.all(embedding_matrix == 0, axis=1)))
print(embedding_matrix)

# # Pad sequences
# max_length = max(max(len(x) for x in sequences_train), max(len(x) for x in sequences_test))
# X_train_pad = pad_sequences(sequences_train, maxlen=max_length, padding='post')
# X_test_pad = pad_sequences(sequences_test, maxlen=max_length, padding='post')

# # Define the LSTM model
# model = Sequential([
#     Embedding(len(word_index) + 1, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
#               trainable=False),
#     LSTM(units=64, dropout=0.2, recurrent_dropout=0.2),
#     Dense(4, activation='softmax')  # Output layer for 4 classes
# ])

# # Compile the model
# model.compile(optimizer='sgd',
#               loss='sparse_categorical_crossentropy',
#               metrics=[SparseCategoricalAccuracy(name='accuracy')])

# # Train the model
# model.fit(X_train_pad, y_train_adjusted, epochs=5, batch_size=16, validation_split=0.2)

# # Evaluate the model with F1 score
# predictions = model.predict(X_test_pad)
# y_pred = np.argmax(predictions, axis=1)

# # Calculate F1 score
# f1 = f1_score(y_test_adjusted, y_pred, average='weighted')
# print(f'F1 Score: {f1:.4f}')
# print(classification_report(y_test_adjusted, y_pred))

# with open('lstm_classification_report.txt', 'w') as f:
#     f.write(classification_report(y_test_adjusted, predictions))