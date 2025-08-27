# RD-Symptomat: Disease Prediction System

An AI-powered disease prediction system that uses machine learning to predict diseases based on symptoms. The system provides a web interface and API for easy interaction.

## Features

- **Machine Learning Model**: Uses Random Forest classifier with TF-IDF vectorization
- **Web Interface**: User-friendly Flask web application
- **API Endpoints**: RESTful API for programmatic access
- **Confidence Scores**: Provides prediction confidence and alternative diagnoses
- **Multi-language Support**: German interface with English backend
- **Error Handling**: Robust error handling and validation
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Quick Start with Docker

### Prerequisites
- Docker installed on your system
- Docker Compose (optional, but recommended)

### Option 1: Quick Deployment (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd rd-symptomat

# Run the deployment script
./deploy.sh
```

The application will be available at: http://localhost:5000

### Option 2: Manual Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build -d

# Or build and run with Docker directly
docker build -t rd-symptomat .
docker run -d -p 5000:5000 --name rd-symptomat-app rd-symptomat
```

### Option 3: Production Deployment with Nginx
```bash
# Deploy with nginx reverse proxy
docker-compose --profile production up --build -d
```

The application will be available at: http://localhost (port 80)

## Local Development Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rd-symptomat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, flask, joblib; print('All dependencies installed successfully!')"
   ```

## Usage

### Quick Start

1. **Run the application**:
   ```bash
   python main.py
   ```

2. **Open your browser** and go to: `http://localhost:5000`

3. **Enter symptoms** in the text area (e.g., "fever, cough, headache")

4. **Get predictions** with confidence scores and alternative diagnoses

### Training the Model

The model is automatically trained when you first run the application. To retrain:

```bash
# Delete existing model files
rm illness_model.pkl vectorizer.pkl

# Run the application (will retrain automatically)
python main.py
```

### Command Line Prediction

You can also make predictions from the command line:

```bash
python model/predict.py "fever, cough, headache"
```

## API Usage

### Prediction Endpoint

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "fever, cough, headache"}'
```

Response:
```json
{
  "prediction": "Common Cold",
  "confidence": 85.2,
  "top_predictions": [
    {"disease": "Common Cold", "probability": 85.2},
    {"disease": "Bronchial Asthma", "probability": 12.1},
    {"disease": "Pneumonia", "probability": 2.7}
  ],
  "symptoms_processed": "fever cough headache"
}
```

### Available Symptoms Endpoint

```bash
curl http://localhost:5000/api/symptoms
```

### Health Check Endpoint

```bash
curl http://localhost:5000/health
```

## Docker Commands

### Basic Commands
```bash
# Build the image
docker build -t rd-symptomat .

# Run the container
docker run -d -p 5000:5000 --name rd-symptomat-app rd-symptomat

# View logs
docker logs rd-symptomat-app

# Stop the container
docker stop rd-symptomat-app

# Remove the container
docker rm rd-symptomat-app
```

### Docker Compose Commands
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build -d
```

## Data Format

The system expects a CSV file (`existing_data.csv`) with the following format:

```csv
Disease,Symptom_1,Symptom_2,Symptom_3,...
Fungal infection,itching,skin_rash,nodal_skin_eruptions,...
Allergy,continuous_sneezing,shivering,chills,...
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: TF-IDF vectorization with n-gram features (1-2)
- **Training**: 80% training, 20% testing split with stratification
- **Performance**: Typically achieves 85-95% accuracy on test data

## File Structure

```
rd-symptomat/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── existing_data.csv       # Training data
├── illness_model.pkl       # Trained model (generated)
├── vectorizer.pkl          # TF-IDF vectorizer (generated)
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── nginx.conf              # Nginx configuration for production
├── deploy.sh               # Deployment script
├── .dockerignore           # Docker ignore file
├── .gitignore              # Git ignore file
├── README.md               # This file
├── model/
│   ├── train_model.py      # Model training logic
│   └── predict.py          # Prediction functions
└── server/
    ├── app.py              # Flask web server
    └── templates/
        ├── index.html      # Main page
        └── result.html     # Results page
```

## Environment Variables

- `PORT`: Port number for the application (default: 5000)
- `FLASK_ENV`: Flask environment (default: production)
- `PYTHONUNBUFFERED`: Python output buffering (default: 1)

## Troubleshooting

### Common Issues

1. **Model files not found**:
   - Run `python main.py` to train the model
   - Check that `existing_data.csv` exists

2. **Import errors**:
   - Install dependencies: `pip install -r requirements.txt`
   - Check Python version (3.7+ required)

3. **Port already in use**:
   - Change port in `server/app.py` or kill existing process
   - Use: `lsof -ti:5000 | xargs kill -9`

4. **Docker build fails**:
   - Check Docker is running
   - Ensure all files are present
   - Try: `docker system prune -a`

5. **Container won't start**:
   - Check logs: `docker logs rd-symptomat-app`
   - Verify port 5000 is available
   - Check disk space

### Performance Tips

- Use specific symptom names from the training data
- Include multiple symptoms for better accuracy
- Avoid very rare or misspelled symptoms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes only. The predictions should not be considered as medical advice.

## Disclaimer

**Important**: This system is designed for educational and research purposes only. The predictions are based on machine learning algorithms and should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical advice and treatment.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error messages
3. Open an issue on GitHub with detailed information
