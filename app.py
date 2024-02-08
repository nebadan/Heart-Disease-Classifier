from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve input data from the form
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        # Load the machine learning model
        model = joblib.load('Heart_Disease_Random_Forest.pkl')

        # Make prediction
        prediction = model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Interpret the prediction Result
        if prediction[0] ==1:
            prediction_text = "The person has heart disease"
        else:
            prediction_text = "The person does not have heart disease"

        # Return the prediction result
        return render_template('index.html', prediction_text=prediction_text)


if __name__ == '__main__':
    app.run(debug=True)
