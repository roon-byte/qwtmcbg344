
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import joblib
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

model = joblib.load('knn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        glucose = data['glucose']
        bmi = data['bmi']
        age = data['age']
        insulin = data['insulin']

        if not all(isinstance(x, (int, float)) for x in [glucose, bmi, age, insulin]):
            return jsonify({'error': 'All inputs must be numeric'}), 400

        input_array = np.array([[glucose, bmi, age, insulin]])
        prediction = model.predict(input_array)

        if prediction[0] == 0:
            result_message = "You have to consult a doctor."
        elif prediction[0] == 1:
            result_message = "You have no risk for diabetes, but maintaining a healthy lifestyle is always important."

        return jsonify({'prediction': result_message})

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
