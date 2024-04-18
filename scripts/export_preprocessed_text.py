import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import re

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()

# Preprocessors
def lowercase(text):
  return text.lower()

def lemmatize(text):
  words = word_tokenize(text)
  return ' '.join([lemmatizer.lemmatize(word) for word in words])

def stem(text):
  words = word_tokenize(text)
  return ' '.join([ps.stem(word) for word in words])

def remove_numbers(text):
  return re.sub(r'\d+', '', text)

def remove_punctuations(text):
  return re.sub(r'[^\w\s]', '', text)

def remove_stopwords(text):
  words = word_tokenize(text)
  return ' '.join([word for word in words if word not in stopwords.words('english')])

Preprocessor = {
  'LOWERCASE': lowercase,
  'LEMMATIZE': lemmatize,
  'STEM': stem,
  'REMOVE_NUMBERS': remove_numbers,
  'REMOVE_PUNCTUATIONS': remove_punctuations,
  'REMOVE_STOPWORDS': remove_stopwords
}

def save_df(df, filename):
  df.to_csv("../preprocessed_text/" + filename, index=False)

if __name__ == '__main__':
  print("Data are stored as pandas dataframe with column name 'Text', 'Preprocessed Text' and 'Verdict'")
  print('🤯 Importing dataset...', end=' ')
  df = pd.read_csv('../raw_data/fulltrain.csv', header=None, names=['Verdict', 'Text'])
  df_test = pd.read_csv('../raw_data/balancedtest.csv', header=None, names=['Verdict', 'Text'])
  print('Done 🐳')

  print("Preparing some initial texts 👀")
  print("Lemmatizing text...", end=' ')
  lemmatized_text = df['Text'].apply(lemmatize)
  lemmatized_text_test = df_test['Text'].apply(lemmatize)
  print('Done ✅')
  print("Stemming text...", end=' ')
  stemmed_text = df['Text'].apply(stem)
  stemmed_text_test = df_test['Text'].apply(stem)
  print('Done ✅')
  print("Removing stopwords...", end=' ')
  removed_stopwords_text = df['Text'].apply(remove_stopwords)
  removed_stopwords_text_test = df_test['Text'].apply(remove_stopwords)
  print('Done ✅')

  def generate_preprocessed_text(preprocessors, name, df=df, df_test=df_test):
    preprocessed = df['Text']
    preprocessed_test = df_test['Text']

    if Preprocessor['REMOVE_STOPWORDS'] in preprocessors:
      preprocessed = removed_stopwords_text
      preprocessed_test = removed_stopwords_text_test
      preprocessors.remove(Preprocessor['REMOVE_STOPWORDS'])
    elif Preprocessor['LEMMATIZE'] in preprocessors:
      preprocessed = lemmatized_text
      preprocessed_test = lemmatized_text_test
      preprocessors.remove(Preprocessor['LEMMATIZE'])
    elif Preprocessor['STEM'] in preprocessors:
      preprocessed = stemmed_text
      preprocessed_test = stemmed_text_test
      preprocessors.remove(Preprocessor['STEM'])

    for preprocessor in preprocessors:
      preprocessed = preprocessed.apply(preprocessor)
      preprocessed_test = preprocessed_test.apply(preprocessor)
    
    preprocessed_df = preprocessed.to_frame(name="Preprocessed Text")
    preprocessed_test_df = preprocessed_test.to_frame(name="Preprocessed Text")
    
    save_df(pd.concat([df, preprocessed_df], axis=1), f'../preprocessed_text/{name}_train.csv')
    save_df(pd.concat([df_test, preprocessed_test_df], axis=1), f'../preprocessed_text/{name}_test.csv')

  tasks = [
    {'name': 'unprocessed', 'preprocessors': []},
    {'name': 'lemmatize', 'preprocessors': [Preprocessor['LEMMATIZE']]},
    {'name': 'stem', 'preprocessors': [Preprocessor['STEM']]},
    {'name': 'remove_stopwords', 'preprocessors': [Preprocessor['REMOVE_STOPWORDS']]},
    {'name': 'remove_stopwords_lemmatize', 'preprocessors': [Preprocessor['REMOVE_STOPWORDS'], Preprocessor['LEMMATIZE']]},
    {'name': 'remove_stopwords_stem', 'preprocessors': [Preprocessor['REMOVE_STOPWORDS'], Preprocessor['STEM']]},
    # {'name': 'lowercase_remove_numbers', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS']]},
    # {'name': 'lowercase_remove_punctuations', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_PUNCTUATIONS']]},
    # {'name': 'lowercase_lemmatize', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['LEMMATIZE']]},
    # {'name': 'lowercase_stem', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['STEM']]},
    # {'name': 'lowercase_remove_stopwords', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_STOPWORDS']]},
    # # Remove numbers + punctuation combinations
    # {'name': 'lowercase_remove_numbers_remove_punctuations', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS']]},
    # {'name': 'lowercase_remove_numbers_remove_punctuations_lemmatize', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS'], Preprocessor['LEMMATIZE']]},
    # {'name': 'lowercase_remove_numbers_remove_punctuations_stem', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS'], Preprocessor['STEM']]},
    # # Remove numbers + punctuation + stopwords combinations
    # {'name': 'lowercase_remove_numbers_remove_punctuations_remove_stopwords', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS'], Preprocessor['REMOVE_STOPWORDS']]},
    # {'name': 'lowercase_remove_numbers_remove_punctuations_remove_stopwords_lemmatize', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS'], Preprocessor['REMOVE_STOPWORDS'], Preprocessor['LEMMATIZE']]},
    # {'name': 'lowercase_remove_numbers_remove_punctuations_remove_stopwords_stem', 'preprocessors': [Preprocessor['LOWERCASE'], Preprocessor['REMOVE_NUMBERS'], Preprocessor['REMOVE_PUNCTUATIONS'], Preprocessor['REMOVE_STOPWORDS'], Preprocessor['STEM']]},
  ]

  for i, task in enumerate(tasks):
    print(f'{i+1}.\tGenerating {task["name"]}...', end=' ')
    generate_preprocessed_text(task['preprocessors'], task['name'])
    print('Done ✅')
