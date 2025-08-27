from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os

class WebServer:
    def __init__(self):
        self.app = Flask(__name__, static_folder='static')
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                symptom_text = request.form.get('symptoms', '').strip()
                
                if not symptom_text:
                    return render_template('result.html', 
                                         result="No symptoms provided",
                                         confidence=0,
                                         error="Please enter your symptoms")
                
                # Import here to avoid circular imports
                from model.predict import predict_illness
                
                # Make prediction
                result = predict_illness(symptom_text)
                
                if 'error' in result:
                    return render_template('result.html',
                                         result="Error occurred",
                                         confidence=0,
                                         error=result['error'])
                
                return render_template('result.html',
                                     result=result['prediction'],
                                     confidence=result['confidence'],
                                     top_predictions=result.get('top_predictions', []),
                                     symptoms_processed=result.get('symptoms_processed', ''))
                
            except Exception as e:
                return render_template('result.html',
                                     result="Error occurred",
                                     confidence=0,
                                     error=f"An error occurred: {str(e)}")
        
        @self.app.route('/api/predict', methods=['POST'])
        def api_predict():
            """API endpoint for predictions"""
            try:
                data = request.get_json()
                symptom_text = data.get('symptoms', '').strip()
                
                if not symptom_text:
                    return jsonify({
                        'error': 'No symptoms provided',
                        'prediction': None,
                        'confidence': 0
                    }), 400
                
                from model.predict import predict_illness
                result = predict_illness(symptom_text)
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'prediction': None,
                    'confidence': 0
                }), 500
        
        @self.app.route('/api/symptoms', methods=['GET'])
        def api_symptoms():
            """API endpoint to get available symptoms"""
            try:
                from model.predict import get_available_symptoms
                symptoms = get_available_symptoms()
                return jsonify({'symptoms': symptoms})
            except Exception as e:
                return jsonify({'error': str(e), 'symptoms': []}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for Docker"""
            return jsonify({'status': 'healthy', 'service': 'rd-symptomat'})
        
        @self.app.route('/favicon.ico')
        def favicon():
            """Serve favicon - return empty response since we use data URI"""
            return Response(status=204)  # No Content
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        print(f"Starting web server on http://{host}:{port}")
        print("Make sure the model is trained before making predictions!")
        
        # Check if model files exist
        if not os.path.exists('illness_model.pkl') or not os.path.exists('vectorizer.pkl'):
            print("Warning: Model files not found. Please run the training script first.")
            print("Run: python model/train_model.py")
        
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    server = WebServer()
    server.run()
