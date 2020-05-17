# Udacity Data Science Nanodegree Project 4 - Disaster Response Pipeline

In this project, I apply the skills learned in the Data Engineering module from the Data Science nanodegree to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. Data is composed of 2 datasets messages.csv which contains the actual messages and categories.csv that contains the categories each message belongs to.

This includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Project Components
There are three main components used in this app.

1. ETL Pipeline
In the process_data.py script, I write a data cleaning pipeline that:

-Loads the messages and categories datasets
-Merges the two datasets
-Cleans the data
-Stores it in a SQLite database

2. ML Pipeline
In the train_classifier.py, I write a machine learning pipeline that:

-Loads data from the SQLite database
-Splits the dataset into training and test sets
-Builds a text processing and machine learning pipeline
-Trains and tunes a model using GridSearchCV
-Outputs results on the test set
-Exports the final model as a pickle file

3. Flask Web App
Web app that show count of categories and allow users to input a message and the app will clasify to which categories it belongs to
