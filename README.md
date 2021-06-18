# Disaster Response Pipeline Project

## Description
 - Disaster Response by inputting message

## Run
  1. Run ETL
    - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  2. Train Classifier
    - python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  3. Run web app
    - python app/run.py

## Explanation
  - app
    - html files for web
    - web app script
  - data
    - data for train
      - disaster_categories.csv
      - disaster_messages.csv
    - databasefile to save cleaned data
    - process_data: ETL script
  - models
    - train script