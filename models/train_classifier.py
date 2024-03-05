import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Args:
        - database_filepath (str): The path to the SQLite database file.

    Returns:
        - X (Series): The messages as the input features.
        - Y (DataFrame): The categories as the target variables.
        - category_names (Index): The names of the categories.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages_And_Categories_Table', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    Y.drop(columns=Y.columns[Y.nunique() == 1], inplace=True)  # remove any features that only have 1 unique value

    return X, Y, Y.columns


def tokenize(text):
    """
    Tokenizes text data.

    This function tokenizes text data by:
    1. Tokenizing the text into words.
    2. Lemmatizing (reducing words to their base form).
    3. Converting to lowercase and stripping whitespace.

    Args:
        - text (str): The text to be tokenized.

    Returns:
        - list: A list of clean tokens.
    """
    tokens = word_tokenize(text)
    clean_tokens = []

    for token in tokens:
        cleaned_token = WordNetLemmatizer().lemmatize(token).lower().strip()
        clean_tokens.append(cleaned_token)

    return clean_tokens


def build_model():
    """
    Builds a machine learning pipeline.

    This function builds a machine learning pipeline that processes text messages and then
    applies a multi-output classifier on the processed text. The classifier is based on
    logistic regression.

    Returns:
        - Pipeline: The constructed machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mpc', MultiOutputClassifier(
            LogisticRegression(
                random_state=42,
                max_iter=1000,
                warm_start=True,
                multi_class='ovr'
            )
        )),
    ])

    """
            The following was used to determine the best parameters, it was commented to save some time when running the code

            pipeline.get_params()
            parameters = {
                'mopc__estimator__max_iter': [100, 500, 1000],
                'mopc__estimator__warm_start': [True, False],
                'mopc__estimator__multi_class': ['auto', 'ovr', 'multinomial'],
            }

            cv = GridSearchCV(pipeline, parameters)

            the performance of Other models such as KNeighborsClassifier and RandomForestClassifier was tested, but a lower
            testing score was archived.

            the same result was also found when adding a standard scaler to the pipe line.
    """

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of the machine learning model.

    This function prints out the classification report for each category, including precision,
    recall, f1-score, and support.

    Args:
        - model (Pipeline): The trained machine learning model.
        - X_test (Series): The test features.
        - Y_test (DataFrame): The test target variables.
        - category_names (Index): The names of the categories.
    """
    y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        print(f"Classification Report for {column}:")
        print(classification_report(Y_test[column], y_pred[:, i], zero_division=1))
        print("=" * 60)


def save_model(model, model_filepath):
    """
    Saves the trained model to a Python pickle file.

    Args:
        - model (Pipeline): The trained machine learning model.
        - model_filepath (str): The filepath where to save the model.
    """
    with open(f'{model_filepath}', 'wb') as file:
        pickle.dump(model, file)


def main():
    """
    Main function to run the machine learning pipeline.

    This function orchestrates the model training and evaluation process by:
    1. Loading the data from a SQLite database.
    2. Splitting the dataset into training and test sets.
    3. Building the machine learning model.
    4. Training the model.
    5. Evaluating the model's performance on the test set.
    6. Saving the trained model as a pickle file.
    """
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
