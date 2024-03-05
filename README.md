# Disaster Response Pipeline

## Summery
In This Project, I built both an ETL pipeline to clean and prepare the datasets and an ML pipeline to build then train a model. The pipelines were built to build a model that can
take any message and then classify it into the appropriate disaster response categories. Then, a Web App was built to allow users to enter any message they want and receive the categorization of that message.

## Files
Inside the app directory, there is a run.py file that is used to run the web app. And inside the data directory, there is  a file named process_data.py that cleans and prepares the data and then stores it in a database.
In the models directory, there is a train_classifier.py file that is responsible for building a model that is used to classify messages.

## Running the App
(1) To run the ETL pipeline that cleans data and stores in the database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
(2) To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

(3) To Run the web app
        `python app/run.py`

(4) Go to http://0.0.0.0:3001/
