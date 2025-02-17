if __name__ == "__main__":
    import pandas as pd
    from model.train_model import train_and_save_model

    # Load the data
    data = pd.read_csv('existing_data.csv')

    # Train the model and save it
    train_and_save_model(data)

    print("Model trained and saved.")

  