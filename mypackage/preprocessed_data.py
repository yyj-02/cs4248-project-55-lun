import pandas as pd
import re

HOME_DIR = '../preprocessed_text/'
LIST_OF_AVAILABLE_PREPROCESSED_DATA = [
    {'name': 'unprocessed', 'lowercase': False, 'remove_stopwords': False, 'remove_punctuations': False, 'remove_numbers': False, 'stem': False, 'lemmatize': False},
    {'name': 'lowercase', 'lowercase': True, 'remove_stopwords': False, 'remove_punctuations': False, 'remove_numbers': False, 'stem': False, 'lemmatize': False},
    {'name': 'remove_numbers', 'lowercase': False, 'remove_stopwords': False, 'remove_punctuations': False, 'remove_numbers': True, 'stem': False, 'lemmatize': False},
    {'name': 'remove_numbers_remove_punctuations', 'lowercase': False, 'remove_stopwords': False, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': False},
    {'name': 'remove_stopwords', 'lowercase': False, 'remove_stopwords': True, 'remove_punctuations': False, 'remove_numbers': False, 'stem': False, 'lemmatize': False},
    {'name': 'lowercase_remove_numbers_remove_punctuations', 'lowercase': True, 'remove_stopwords': False, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': False},
    {'name': 'lowercase_remove_stopwords', 'lowercase': True, 'remove_stopwords': True, 'remove_punctuations': False, 'remove_numbers': False, 'stem': False, 'lemmatize': False},
    {'name': 'lowercase_remove_numbers_remove_punctuations_remove_stopwords', 'lowercase': True, 'remove_stopwords': True, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': False},
    {'name': 'lemmatize_remove_numbers_remove_punctuations', 'lowercase': False, 'remove_stopwords': False, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': True},
    {'name': 'lemmatize_remove_numbers_remove_punctuations_remove_stopwords', 'lowercase': False, 'remove_stopwords': True, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': True},
    {'name': 'lowercase_lemmatize_remove_numbers_remove_punctuations_remove_stopwords', 'lowercase': True, 'remove_stopwords': True, 'remove_punctuations': True, 'remove_numbers': True, 'stem': False, 'lemmatize': True},
]

class PreprocessedData:
    def __init__(self, lowercase=False, remove_stopwords=False, remove_punctuations=False, remove_numbers=False, stem=False, lemmatize=False):
        # data is in the format of (train, test)
        self.data = None
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.remove_punctuations = remove_punctuations
        self.remove_numbers = remove_numbers
        self.stem = stem
        self.lemmatize = lemmatize
    
    def print_preprocessing_steps(self):
        print("lowercase: ", self.lowercase)
        print("remove_stopwords: ", self.remove_stopwords)
        print("remove_punctuation: ", self.remove_punctuations)
        print("remove_numbers: ", self.remove_numbers)
        print("stem: ", self.stem)
        print("lemmatize: ", self.lemmatize)

    def populate_data(self):
        if self.stem and self.lemmatize:
            raise ValueError("Cannot stem and lemmatize at the same time")
        
        # We have a set of base preprocessed data that we can use
        train_df, test_df = (pd.read_csv('unprocessed_train.csv'), pd.read_csv('unprocessed_test.csv'))
        if self.remove_stopwords and self.stem:
            train_df, test_df = (pd.read_csv(HOME_DIR + 'remove_stopwords_stem_train.csv'), pd.read_csv(HOME_DIR + 'remove_stopwords_stem_test.csv'))
        elif self.remove_stopwords and self.lemmatize:
            train_df, test_df = (pd.read_csv(HOME_DIR + 'remove_stopwords_lemmatize_train.csv'), pd.read_csv(HOME_DIR + 'remove_stopwords_lemmatize_test.csv'))
        elif self.stem:
            train_df, test_df = (pd.read_csv(HOME_DIR + 'stem_train.csv'), pd.read_csv(HOME_DIR + 'stem_test.csv'))
        elif self.lemmatize:
            train_df, test_df = (pd.read_csv(HOME_DIR + 'lemmatize_train.csv'), pd.read_csv(HOME_DIR + 'lemmatize_test.csv'))
        elif self.remove_stopwords:
            train_df, test_df = (pd.read_csv(HOME_DIR + 'remove_stopwords_train.csv'), pd.read_csv(HOME_DIR + 'remove_stopwords_test.csv'))
        
        # Applying the remaining transformation
        if self.lowercase:
            train_df['Preprocessed Text'] = train_df['Preprocessed Text'].apply(self.__lowercase)
            test_df['Preprocessed Text'] = test_df['Preprocessed Text'].apply(self.__lowercase)
        
        if self.remove_punctuations:
            train_df['Preprocessed Text'] = train_df['Preprocessed Text'].apply(self.__remove_punctuations)
            test_df['Preprocessed Text'] = test_df['Preprocessed Text'].apply(self.__remove_punctuations)
        
        if self.remove_numbers:
            train_df['Preprocessed Text'] = train_df['Preprocessed Text'].apply(self.__remove_numbers)
            test_df['Preprocessed Text'] = test_df['Preprocessed Text'].apply(self.__remove_numbers)
        
        # Shuffle data
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        
        self.data = (train_df, test_df)
        

    def get_all(self):
        if self.data is None:
            self.populate_data()

        return self.data

    def get_train(self):
        if self.data is None:
            self.populate_data()

        return self.data[0]

    def get_test(self):
        if self.data is None:
            self.populate_data()

        return self.data[1]
    
    def get_raw_data(self):
        if self.data is None:
            self.populate_data()

        return self.data[0]['Text'], self.data[1]['Text']
    
    def get_verdicts(self):
        if self.data is None:
            self.populate_data()
        
        return self.data[0]['Verdict'], self.data[1]['Verdict']
    
    def get_preprocessed_data(self):
        if self.data is None:
            self.populate_data()
        
        return self.data[0]['Preprocessed Text'], self.data[1]['Preprocessed Text']
    
    @staticmethod
    def generator():
        # Gives a list of (name, train, test)
        for preprocessors in LIST_OF_AVAILABLE_PREPROCESSED_DATA:
            train_df, test_df = PreprocessedData(preprocessors['lowercase'], preprocessors['remove_stopwords'], preprocessors['remove_punctuations'], preprocessors['remove_numbers'], preprocessors['stem'], preprocessors['lemmatize']).get_all()
            yield (preprocessors['name'], train_df, test_df)
    
    def __lowercase(self, text):
        return text.lower()
    
    def __remove_numbers(text):
        return re.sub(r'\d+', '', text)

    def __remove_punctuations(text):
        return re.sub(r'[^\w\s]', '', text)

