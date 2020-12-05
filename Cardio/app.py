


from flask import Flask,render_template,request
import numpy as np
import pickle

# Load the model
classifier = pickle.load(open('xgbmodel_1.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == ['POST']:
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        bmi = float(request.form['bmi'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        data = np.array([[age, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
