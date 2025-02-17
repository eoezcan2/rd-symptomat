from flask import Flask, render_template
from flask import request

class WebServer:
    def __init__(self):
        self.app = Flask(__name__)

    def run(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.post('/predict')
        def predict():
            symptom_text = request.form['symptoms']
            from model.predict import predict_illness
            result = predict_illness(symptom_text)
            return render_template('result.html', result=result)

        self.app.run(debug=True)
