import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_symptoms(symptom_text):
    """
    Clean and normalize symptom text
    """
    if pd.isna(symptom_text) or symptom_text == '':
        return ''
    
    # Convert to string and lowercase
    symptom_text = str(symptom_text).lower().strip()
    
    # Replace underscores with spaces
    symptom_text = symptom_text.replace('_', ' ')
    
    # Remove extra spaces and normalize
    symptom_text = re.sub(r'\s+', ' ', symptom_text)
    
    return symptom_text

def train_and_save_model(data, model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to train the model and save it.
    :param data: DataFrame with 'Disease' and symptom columns.
    :param model_filename: The filename to save the trained model.
    :param vectorizer_filename: The filename to save the vectorizer.
    """
    print("Starting model training...")
    
    # Get symptom columns (all columns except 'Disease')
    symptom_columns = [col for col in data.columns if col.startswith('Symptom_')]
    
    # Clean and combine symptoms
    print(f"Processing {len(data)} records with {len(symptom_columns)} symptom columns...")
    
    # Process each row to combine symptoms
    combined_symptoms = []
    diseases = []
    
    for idx, row in data.iterrows():
        # Collect all non-empty symptoms for this row
        symptoms = []
        for col in symptom_columns:
            symptom = clean_symptoms(row[col])
            if symptom and symptom != '' and symptom != 'nan':
                symptoms.append(symptom)
        
        # Only include rows with at least one symptom
        if symptoms:
            combined_symptoms.append(' '.join(symptoms))
            diseases.append(row['Disease'].strip())  # Clean disease name too
    
    print(f"Processed {len(combined_symptoms)} valid symptom records")
    
    if len(combined_symptoms) == 0:
        raise ValueError("No valid symptom data found!")
    
    # Create DataFrame for easier processing
    df = pd.DataFrame({
        'symptoms_text': combined_symptoms,
        'disease': diseases
    })
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {len(df)} records")
    
    # Check disease distribution
    disease_counts = df['disease'].value_counts()
    print(f"Number of unique diseases: {len(disease_counts)}")
    print("Top 10 diseases:")
    for disease, count in disease_counts.head(10).items():
        print(f"  {disease}: {count} records")
    
    # Vectorizing the symptoms (text data)
    print("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words='english',
        max_features=2000,
        min_df=2,
        max_df=0.9
    )
    
    X = vectorizer.fit_transform(df['symptoms_text'])
    y = df['disease']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of unique diseases: {len(y.unique())}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save the trained model and vectorizer
    print(f"Saving model to {model_filename}...")
    joblib.dump(model, model_filename)
    
    print(f"Saving vectorizer to {vectorizer_filename}...")
    joblib.dump(vectorizer, vectorizer_filename)
    
    print("Model training completed successfully!")
    
    return model, vectorizer, accuracy

if __name__ == "__main__":
    # Load the existing data
    print("Loading data...")
    existing_data = pd.read_csv('existing_data.csv')
    
    # Train the model with the existing data and save it
    train_and_save_model(existing_data)
