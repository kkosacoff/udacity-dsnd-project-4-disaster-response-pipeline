# Udacity Data Science Nanodegree Project 4 - Disaster Response Pipeline

In this project, I apply the skills learned in the Data Engineering module from the Data Science nanodegree to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

This includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

## Project Components
There are three components you'll need to complete for this project.

1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you
