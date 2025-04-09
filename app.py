from flask import Flask, jsonify, request
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open('trained_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working and update version!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pregnancies = float(request.form.get('pregnancies'))
        glucose = float(request.form.get('glucose'))
        blood_pressure = float(request.form.get('blood_pressure'))
        insulin = float(request.form.get('insulin'))
        bmi = float(request.form.get('bmi'))
        age = float(request.form.get('age'))

        # Convert input and scale it
        input_data = np.array([[pregnancies, glucose, blood_pressure, insulin, bmi, age]])
        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)

        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})
    

    # 1--> person is diabetic
    # 0--> person is non diabetic

if __name__ == '__main__':
    app.run(debug=True)
