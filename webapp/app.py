import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))# loads ML model

@app.route('/')
def home():
    return render_template('index.html')# renders index.html

@app.route('/predict',methods=['POST'])# gets the values that were sent to '/predict' by 'index.html'
def predict():
    int_features = [float(x) for x in request.form.values()]# defines the form values in an array
    final_features = [np.array(int_features)]# turns the form values into a Numpy array
    prediction = model.predict(final_features)# makes a prediction using the values in the created Numpy array

    output = prediction[0]# gets the prediction as a string

    return render_template('index.html', prediction_text='Iris species: {}'.format(output))# displays the prediction inside the '<b>{{ prediction_text }}</b>' that we've seen in 'index.html'

if __name__ == "__main__":
    app.run(debug=True)# Runs the Web App
