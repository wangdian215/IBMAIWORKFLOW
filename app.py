from flask import Flask, jsonify, request, render_template, redirect
import joblib
import socket
import json
import pandas as pd
import os
import sys
import requests
from model import model_predict, model_train

#MODEL_DIR = "models"
#DATA_DIR = "data"

#MODEL_DIR = os.path.join(os.path.expanduser('~'),"AI Workflow Last Module AI in Prod","Capstone","models")
#DATA_DIR = os.path.join(os.path.expanduser('~'),"AI Workflow Last Module AI in Prod","Capstone","data")

app = Flask(__name__)

@app.route("/")
def hello():
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/train', methods=['GET'])
def my_form():
    return render_template('training.html')

@app.route('/train', methods=['POST'])
def train():
    text = request.form['text']
    model_train(text)
    print("Completed model training!")
    return (jsonify("Completed model training!"))

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form["Date"]
    year = text.split('-')[0]
    month = text.split('-')[1]
    date = text.split('-')[2]
    country = request.form["Country"]
    prediction = model_predict(country, year, month, date)
    prediction_jsonify = prediction['y_pred'].tolist()[0]
    print('Given number is:', prediction_jsonify)
    output_text = country+": Predicted Forecast for 30 day period on "+text+" is: "+str(round(prediction_jsonify, 2))
    print("Expected output:", output_text)
    return jsonify(output_text)


if __name__ == '__main__':
    #MODEL_DIR = os.path.join(os.path.expanduser('~'),"AI Workflow Last Module AI in Prod","Capstone","models")
    saved_model = 'models/sl-all-0_1.joblib'
    model = joblib.load(saved_model)
    #model = joblib.load(os.path.join(MODEL_DIR, saved_model))
    app.run(host='0.0.0.0', port=8080,debug=True)
