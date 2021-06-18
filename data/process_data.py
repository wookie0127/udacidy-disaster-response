"""
Process data files and insert them into database file
"""
# global
import sys

# dependencies
import pandas as pd

from sqlalchemy import create_engine

# custom


def load_data(messages_filepath:str, categories_filepath:str) -> object:
    """
    Load data from csv files and merge
    Args:
        messages_filepath (str)
        categories_filepath (str)

    Returns:
        object: Pandas.DataFrame
    """
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df:object) -> object:
    """
    Clean data to train

    Args:
        df (object): DataFrame made from load_data function

    Returns:
        object: [description]
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories.loc[:, column].apply(lambda x: x[-1])
        categories[column] = categories.loc[:, column].apply(lambda x: int(x) != 2)
        categories[column] = categories.loc[:, column].astype(int)
        
    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df:object, database_filename:str) -> None:
    """
    Save data in DataFrame to database

    Args:
        df (object): DataFrame made from clean_data
        database_filename (str): filepath for database
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'''
              Loading data...
              MESSAGES  : {messages_filepath}
              CATEGORIES: {categories_filepath}
              ''')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'''
              Saving data...
              DATABASE: {database_filepath}
              ''')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()