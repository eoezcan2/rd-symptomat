#!/usr/bin/env python3
"""
RD-Symptomat: Disease Prediction System
Main entry point for training the model and starting the web server
"""

if __name__ == "__main__":
    import pandas as pd
    import os
    import sys
    from model.train_model import train_and_save_model
    from server.app import WebServer

    print("=" * 50)
    print("RD-Symptomat - Disease Prediction System")
    print("=" * 50)

    # Check if data file exists
    if not os.path.exists('existing_data.csv'):
        print("Error: existing_data.csv not found!")
        print("Please make sure the data file is in the current directory.")
        sys.exit(1)

    try:
        # Load the data
        print("Loading training data...")
        data = pd.read_csv('existing_data.csv')
        print(f"Loaded {len(data)} records from existing_data.csv")

        # Check if model files already exist
        if os.path.exists('illness_model.pkl') and os.path.exists('vectorizer.pkl'):
            print("Model files already exist. Skipping training...")
            print("To retrain the model, delete illness_model.pkl and vectorizer.pkl")
        else:
            print("Training new model...")
            # Train the model and save it
            model, vectorizer, accuracy = train_and_save_model(data)
            print(f"Model training completed with accuracy: {accuracy:.4f}")

        print("\nStarting web server...")
        
        # Get port from environment variable or use default
        port = int(os.environ.get('PORT', 5000))
        
        # In Docker, bind to all interfaces
        host = '0.0.0.0'
        
        print(f"The application will be available at: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        print("-" * 50)

        # Run the flask web server
        WebServer().run(host=host, port=port, debug=False)

    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data file and try again.")
        sys.exit(1)
