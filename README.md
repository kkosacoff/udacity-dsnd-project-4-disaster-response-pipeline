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

## How to use the App

First you need to ensure you have Python 3 installed in your computer

1. Download this repository
2. Run the following script from your terminal to clean the datasets, last argument is the name of the database you want to save your cleaned dataset.
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

3. Run the following from your terminal to train your model, providing where your dataset is saved (DisasterResponse.db), and where you want to save your model (classifier.pkl)

```
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

4. Run the following script from your terminal.
```
python run.py
```
5. Go to [this](http://0.0.0.0:3001/) address in your browser, or copy and paste http://0.0.0.0:3001/ and hit enter in your browser and hit enter.
