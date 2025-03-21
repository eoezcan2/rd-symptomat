import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.corpus import stopwords

try:
    german_stopwords = stopwords.words('german')
except LookupError:
    nltk.download('stopwords')

def train_and_save_model(data, model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to train the model and save it.
    :param data: DataFrame with 'symptoms' and 'illness' columns.
    :param model_filename: The filename to save the trained model.
    :param vectorizer_filename: The filename to save the vectorizer.
    """
    # Extract symptoms and illness
    symptoms = data['symptoms']
    illnesses = data['illness']

    # Vectorizing the symptoms (text data)
    vectorizer = TfidfVectorizer(stop_words=german_stopwords)
    X = vectorizer.fit_transform(symptoms)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, illnesses, test_size=0.2, random_state=42)

    # Create a Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model and vectorizer to disk
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

    print(f"Model and vectorizer saved to {model_filename} and {vectorizer_filename}")
