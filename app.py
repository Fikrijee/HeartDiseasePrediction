from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)


model = joblib.load('heart_disease_model.pkl')

def fix_case(value, column):
    mappings = {
        'ChestPainType': {
            'typical angina': 'TA',
            'atypical angina': 'ATA',
            'non-anginal pain': 'NAP',
            'asymptomatic': 'ASY',
            'ta': 'TA',
            'ata': 'ATA',
            'nap': 'NAP',
            'asy': 'ASY'
        },
        'RestingECG': {
            'normal': 'Normal',
            'st': 'ST',
            'left ventricular hypertrophy': 'LVH',
            'st-t wave abnormality': 'ST',
            'lvh': 'LVH'
        },
        'ExerciseAngina': {
            'yes': 'Y',
            'no': 'N',
            'y': 'Y',
            'n': 'N'
        },
        'ST_Slope': {
            'up': 'Up',
            'flat': 'Flat',
            'down': 'Down'
        }
    }
    
    value = value.strip().lower()
    if column in mappings and value in mappings[column]:
        return mappings[column][value]
    return value  


@app.route('/')
def home():
    return render_template('index.html')  


@app.route('/predict', methods=['POST'])
def predict():
  
    features = [x.strip() for x in request.form.values()]


    input_data = pd.DataFrame({
        'Age': [int(features[0])],
        'Sex': [int(features[1])],
        'ChestPainType': [fix_case(features[2], 'ChestPainType')],
        'RestingBP': [int(features[3])],
        'Cholesterol': [int(features[4])],
        'FastingBS': [int(features[5])],
        'RestingECG': [fix_case(features[6], 'RestingECG')],
        'MaxHR': [int(features[7])],
        'ExerciseAngina': [fix_case(features[8], 'ExerciseAngina')],
        'Oldpeak': [float(features[9])],
        'ST_Slope': [fix_case(features[10], 'ST_Slope')]
    })


    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = "❌ Ka rrezik për sëmundje të zemrës! Konsultohu me mjekun."
    else:
        result = "✅ Nuk ka rrezik për sëmundje të zemrës."

    return render_template('index.html', prediction_text=result)


if __name__ == '__main__':
    app.run(debug=True)
