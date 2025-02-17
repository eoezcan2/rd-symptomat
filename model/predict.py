import joblib
import sys

def load_model(model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to load the trained model and vectorizer.
    :param model_filename: The filename to load the trained model.
    :param vectorizer_filename: The filename to load the vectorizer.
    :return: The loaded model and vectorizer.
    """
    model = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
    return model, vectorizer

def predict_illness(symptom_text, model_filename='illness_model.pkl', vectorizer_filename='vectorizer.pkl'):
    """
    Function to predict the illness based on symptoms.
    :param symptom_text: The input symptoms as text.
    :param model_filename: The filename of the trained model.
    :param vectorizer_filename: The filename of the vectorizer.
    :return: Predicted illness.
    """
    model, vectorizer = load_model(model_filename, vectorizer_filename)
    symptom_vector = vectorizer.transform([symptom_text])
    prediction = model.predict(symptom_vector)
    return prediction[0]

if __name__ == "__main__":
    # Get input from sys args
    test_input = sys.argv[1]

    predicted_illness = predict_illness(test_input)
    print(f"The predicted illness is: {predicted_illness}")
