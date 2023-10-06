import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        country = int(request.form['country'])
        year = int(request.form['year'])
        schizophrenia = float(request.form['schizophrenia'])
        bipolar_disorder = float(request.form['bipolar_disorder'])
        eating_disorder = float(request.form['eating_disorder'])
        anxiety = float(request.form['anxiety'])
        drug_usage = float(request.form['drug_usage'])
        depression = float(request.form['depression'])
        alcohol = float(request.form['alcohol'])

        input_data = [[country, year, schizophrenia, bipolar_disorder, eating_disorder, anxiety, drug_usage, depression, alcohol]]
        prediction = model.predict(input_data)

        return render_template('index.html', prediction=prediction[0])
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
