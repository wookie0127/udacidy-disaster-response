'''
Train script for modeling classification
'''
# global
import sys

# dependenies
import pandas as pd
import nltk

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib

# custom


def load_data(database_filepath:str) -> tuple:
    """
    Load data from database file

    Args:
        database_filepath (str): data merged disaster messages and categories

    Returns:
        tuple: features, answer, category_names 
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response', engine)
    X = df.message.values
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text:str) -> list:
    """
    tokenize text

    Args:
        text (str)

    Returns:
        list: consisting of tokens
    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok.strip().lower())
        clean_tokens.append(clean_tok)
    return clean_tokens



def build_model() -> object:
    """
    Build model using pipeline to handle text data

    Returns:
        object: Pipeline containing vectorizeing and classifier
    """
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])



def evaluate_model(model:object, X_test:object, Y_test:object, category_names:list) -> None:
    """
    Evaluate model and display

    Args:
        model (object): Pipeline
        X_test (object): Pandas.Series
        Y_test (object): Pandas.Series
        category_names (list): list containing category names
    """
    y_pred = model.predict(X_test)
    for idx, colname in enumerate(category_names):
        print('-'*20)
        print(colname)
        print(classification_report(Y_test[colname], y_pred[:,idx]))
        print('-'*20)


def save_model(model:object, model_filepath:str) -> None:
    """
    Save model into model_filepath

    Args:
        model (object): trained model
        model_filepath (str): filepath having pickle as extension 
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'''
              Loading data...
              DATABASE: {database_filepath}
              ''')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'''
              Saving model...
              MODEL: {model_filepath}
              ''')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()