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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages_And_Categories_Table', engine)

    X = df['message']
    Y = df.iloc[:, 4:]
    Y.drop(columns=Y.columns[Y.nunique() == 1], inplace=True)  # remove any features that only have 1 unique value

    return X, Y, Y.columns


def tokenize(text):
    tokens = word_tokenize(text)
    clean_tokens = []

    for token in tokens:
        cleaned_token = WordNetLemmatizer().lemmatize(token).lower().strip()
        clean_tokens.append(cleaned_token)

    return clean_tokens


def build_model():
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
            The following was used to determine the best parameters

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
    y_pred = model.predict(X_test)

    for i, column in enumerate(category_names):
        print(f"Classification Report for {column}:")
        print(classification_report(Y_test[column], y_pred[:, i], zero_division=1))
        print("=" * 60)


def save_model(model, model_filepath):
    import pickle
    with open(f'{model_filepath}', 'wb') as file:
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
