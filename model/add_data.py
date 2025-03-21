import pandas as pd
from train_model import train_and_save_model
import sys

def add_new_data(new_data, existing_data_filename='existing_data.csv'):
    """
    Function to add new data to the existing dataset and retrain the model.
    :param new_data: New data to be added (list of dictionaries with keys matching column names).
    :param existing_data_filename: The filename where the existing data is saved.
    """
    # Load existing data (if exists)
    try:
        existing_data = pd.read_csv(existing_data_filename)
    except FileNotFoundError:
        columns = ['Disease'] + [f'Symptom_{i}' for i in range(1, 18)]
        existing_data = pd.DataFrame(columns=columns)

    # Convert new data to a DataFrame and append
    new_data_df = pd.DataFrame(new_data)
    updated_data = pd.concat([existing_data, new_data_df], ignore_index=True)

    # Save the updated data to the file
    updated_data.to_csv(existing_data_filename, index=False)

    # Train the model with the updated data and save it
    train_and_save_model(updated_data)

if __name__ == "__main__":
    # Get new data from sys args and check validity
    new_data = []
    for i in range(1, len(sys.argv), 2):
        if i + 1 < len(sys.argv):
            new_data.append((sys.argv[i], sys.argv[i + 1]))
        else:
            print("Invalid input format. Please provide symptoms and illness as pairs.")
            sys.exit

    # Add the new data and retrain the model
    add_new_data(new_data)
