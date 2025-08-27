import joblib
import re
import sys
import numpy as np

def clean_symptoms(symptom_text):
    """
    Clean and normalize symptom text
    """
    if not symptom_text or symptom_text == '':
        return ''
    
    # Convert to string and lowercase
    symptom_text = str(symptom_text).lower().strip()
    
    # Replace underscores with spaces
    symptom_text = symptom_text.replace('_', ' ')
    
    # Remove extra spaces and normalize
    symptom_text = re.sub(r'\s+', ' ', symptom_text)
    
    return symptom_text

def load_model(model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to load the trained model and vectorizer.
    :param model_filename: The filename to load the trained model.
    :param vectorizer_filename: The filename to load the vectorizer.
    :return: The loaded model and vectorizer.
    """
    try:
        model = joblib.load(model_filename)
        vectorizer = joblib.load(vectorizer_filename)
        return model, vectorizer
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model files not found. Please train the model first. Error: {e}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

def predict_illness(symptoms, model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to predict the illness based on symptoms.
    :param symptoms: A string of symptoms separated by commas or spaces.
    :param model_filename: The filename of the trained model.
    :param vectorizer_filename: The filename of the vectorizer.
    :return: Dictionary with prediction and confidence.
    """
    try:
        # Load model and vectorizer
        model, vectorizer = load_model(model_filename, vectorizer_filename)
        
        # Clean and process symptoms
        if isinstance(symptoms, list):
            symptom_text = ' '.join([clean_symptoms(s) for s in symptoms])
        else:
            symptom_text = clean_symptoms(symptoms)
        
        if not symptom_text:
            return {
                'prediction': 'No symptoms provided',
                'confidence': 0.0,
                'error': 'Please provide valid symptoms'
            }
        
        # Vectorize the symptoms
        symptom_vector = vectorizer.transform([symptom_text])
        
        # Make prediction
        prediction = model.predict(symptom_vector)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(symptom_vector)[0]
        confidence = np.max(probabilities) * 100
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            if probabilities[idx] > 0.01:  # Only include predictions with >1% probability
                top_predictions.append({
                    'disease': model.classes_[idx],
                    'probability': probabilities[idx] * 100
                })
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'symptoms_processed': symptom_text
        }
        
    except Exception as e:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'error': str(e)
        }

def get_available_symptoms():
    """
    Get list of available symptoms from the training data
    """
    try:
        import pandas as pd
        data = pd.read_csv('existing_data.csv')
        symptom_columns = [col for col in data.columns if col.startswith('Symptom_')]
        
        all_symptoms = set()
        for col in symptom_columns:
            symptoms = data[col].dropna().unique()
            for symptom in symptoms:
                if symptom and str(symptom).strip():
                    cleaned = clean_symptoms(symptom)
                    if cleaned:
                        all_symptoms.add(cleaned)
        
        # Sort symptoms alphabetically and return as list
        return sorted(list(all_symptoms))
    except Exception as e:
        print(f"Error getting available symptoms: {e}")
        return []

def get_symptoms_by_category():
    """
    Get symptoms organized by category for better UI organization
    """
    try:
        import pandas as pd
        data = pd.read_csv('existing_data.csv')
        symptom_columns = [col for col in data.columns if col.startswith('Symptom_')]
        
        all_symptoms = set()
        for col in symptom_columns:
            symptoms = data[col].dropna().unique()
            for symptom in symptoms:
                if symptom and str(symptom).strip():
                    cleaned = clean_symptoms(symptom)
                    if cleaned:
                        all_symptoms.add(cleaned)
        
        # Categorize symptoms (you can expand this categorization)
        categories = {
            'General': [],
            'Respiratory': [],
            'Digestive': [],
            'Skin': [],
            'Neurological': [],
            'Cardiovascular': [],
            'Eye & Vision': [],
            'Other': []
        }
        
        # Define category keywords
        category_keywords = {
            'Respiratory': ['cough', 'breath', 'sneezing', 'sputum', 'asthma', 'pneumonia', 'tuberculosis'],
            'Digestive': ['stomach', 'abdominal', 'vomiting', 'diarrhoea', 'nausea', 'ulcer', 'gastro', 'appetite', 'gases'],
            'Skin': ['itching', 'rash', 'skin', 'patches', 'eruptions', 'dischromic', 'impetigo', 'psoriasis'],
            'Neurological': ['headache', 'migraine', 'dizziness', 'vertigo', 'paralysis', 'seizure', 'epilepsy'],
            'Cardiovascular': ['chest', 'heart', 'blood', 'pressure', 'attack', 'hypertension'],
            'Eye & Vision': ['eye', 'vision', 'blurred', 'yellowing', 'watering']
        }
        
        # Categorize symptoms
        for symptom in sorted(all_symptoms):
            categorized = False
            for category, keywords in category_keywords.items():
                if any(keyword in symptom for keyword in keywords):
                    categories[category].append(symptom)
                    categorized = True
                    break
            
            if not categorized:
                categories['Other'].append(symptom)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
        
    except Exception as e:
        print(f"Error categorizing symptoms: {e}")
        return {'General': get_available_symptoms()}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'symptom1, symptom2, symptom3'")
        print("Example: python predict.py 'fever, cough, headache'")
        sys.exit(1)
    
    # Get input from sys args
    test_input = sys.argv[1]
    
    result = predict_illness(test_input)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Predicted illness: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"Processed symptoms: {result['symptoms_processed']}")
        
        if result['top_predictions']:
            print("\nTop predictions:")
            for i, pred in enumerate(result['top_predictions'], 1):
                print(f"{i}. {pred['disease']}: {pred['probability']:.2f}%")
