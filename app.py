from flask import Flask, jsonify, request
import pickle
import numpy as np

model = pickle.load(open('trained_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask is working!"


@app.route('/predict', methods=['POST'])
def predict():
    #data = request.form.get  # Get JSON data from Android app
    pregnancies = request.form.get('pregnancies')
    glucose = request.form.get('glucose')
    blood_pressure = request.form.get('blood_pressure')
    insulin = request.form.get('insulin')
    bmi = request.form.get('bmi')
    age = request.form.get('age')


    input_data = np.array([[pregnancies,glucose,blood_pressure,insulin, bmi, age]])
    prediction = model.predict(input_data)

    return jsonify({'prediction': int(prediction[0])})  # 1


if __name__ == '__main__':
    app.run(debug=True)
#host='0.0.0.0', port=5000