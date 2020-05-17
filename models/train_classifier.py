import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
# Basic modules for data manipulation
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

## NLTK Modules for NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

##Scikit Learn modules for Machine Learning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#import pickle for saving the model
import pickle

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    '''
        Input is the filepath of the database, with that function loads data from database
        Returns X (predictor), Y (variables to predict), category names 
    '''
    database_filepath = 'sqlite:///' + database_filepath
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('messages', engine)
    X = df.message
    Y = df[df.columns[4:]]
    Y = Y.drop(['child_alone'], axis = 1) ## Dropping as all values are 0, adds no additional information and affects some ML algorithms
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
        Input is text, which in this case it will be used for the messages.
        Custom tokenizer to use in our model that converts words into clean, lemmatized tokens for easier processing in the model
    '''
    
    tokens = word_tokenize(text)
    
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
        
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return clean_tokens


def build_model():
    '''
        We define the model here using the Pipeline functionality from sklearn.
        Model uses CountVectorizer and TfidfTransformer as transformers and SGDCClassifier as classifier
        It returns the model.
    '''
    
    model = Pipeline([
                            ('vect', CountVectorizer(tokenizer = tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultiOutputClassifier(SGDClassifier()))
                          ])
    
    return model


def evaluate_model(model, X_test, Y_test):
    '''
        Function gets as input:
            Model returned from build_model() function
            X_test to predict values from the model
            Y_test to compared predicted values
        Function prints results of the model on how well it predicted Y
    '''
    
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
        Inputs the model and a filepath
        Saves the model as a pkl file for later use in the model_filepath location
    '''
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()