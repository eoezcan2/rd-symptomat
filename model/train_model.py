import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.corpus import stopwords

try:
    my_stopwords = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def train_and_save_model(data, model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to train the model and save it.
    :param data: DataFrame with 'Disease' and symptom columns.
    :param model_filename: The filename to save the trained model.
    :param vectorizer_filename: The filename to save the vectorizer.
    """
    # Combine symptoms into a single text representation
    symptom_columns = [col for col in data.columns if col.startswith('Symptom_')]
    data['symptoms_text'] = data[symptom_columns].fillna('').agg(' '.join, axis=1).str.strip()

    # Remove empty symptom entries
    data = data[data['symptoms_text'] != '']

    # Extract symptoms and diseases
    symptoms = data['symptoms_text']
    diseases = data['Disease']

    # Vectorizing the symptoms (text data)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(symptoms)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, diseases, test_size=0.2, random_state=42, stratify=diseases)

    # Create a Random Forest model (more robust than Logistic Regression)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and vectorizer to disk
    joblib.dump(model, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

    print(f"Model and vectorizer saved to {model_filename} and {vectorizer_filename}")

if __name__ == "__main__":
    # Load the existing data
    existing_data = pd.read_csv('existing_data.csv')

    # Train the model with the existing data and save it
    train_and_save_model(existing_data)
