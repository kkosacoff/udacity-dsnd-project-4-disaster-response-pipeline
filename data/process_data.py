import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
		This function gets as input 2 files, one with messages information, and another one with categories
		Then it will merge both of them into a single dataframe on the 'id' column
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on=['id'])
    
    return df


def clean_data(df):
	categories = df['categories'].str.split(';', expand=True)
	'''
		This function cleans the dataframe returned by the load_data function
		It expands the categories column into one category per column and the leaves a numerical value of 1 or 0 for each observation
		Then removes categories column from the original dataframe.
		Finally it returns a dataframe with all the new category columns
	'''

    # select the first row of the categories dataframe
	row = categories.iloc[0]

	category_colnames = row.apply(lambda x: x[:-2]) ##extract category names from 1st row, slicing full string, minus 2 last characters.
	categories.columns = category_colnames
    
	for column in categories:
        # set each value to be the last character of the string
		categories[column] = categories[column].str[-1]
        # convert column from string to numeric
		categories[column] = categories[column].apply(lambda x: int(x))
    
	##Drop original categories column from original df
	df.drop(['categories'], inplace=True, axis=1)

	##Concatenate df with newly created and cleaned categories df
	df = pd.concat([df, categories], axis=1)

	## Remove duplicates
	df.drop_duplicates(inplace=True)

	return df

def save_data(df, database_filename):
	'''
		Function that has 2 inputs, a Dataframe, and a database file name.
		This function then creates a database if not yet created, or will overwrite the previous one with the contents of the dataframe
		This function doesn't return anything, it only writes the database.
	'''
	database_filename = 'sqlite:///' + database_filename
	engine = create_engine(database_filename)
	df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
	if len(sys.argv) == 4:

	    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

	    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
	          .format(messages_filepath, categories_filepath))
	    df = load_data(messages_filepath, categories_filepath)

	    print('Cleaning data...')
	    df = clean_data(df)
	    
	    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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