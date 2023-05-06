from flask import Flask, request, jsonify
import numpy as np
import pickle

classifier = pickle.load(open('logistic.sav','rb'))
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello'

@app.route('/predict', methods = ['POST'])
def predict():
    gender = request.form.get('gender')
    sscPercentage = request.form.get('sscPercentage')
    sscBoard = request.form.get('sscBoard')
    hscPercentage = request.form.get('hscPercentage')
    hscBoard = request.form.get('hscBoard')
    hscStream = request.form.get('hscStream')
    degreePercentage = request.form.get('degreePercentage')
    degreeTechnology = request.form.get('degreeTechnology')
    workExperience = request.form.get('workExperience')
    etestPercentage = request.form.get('etestPercentage')
    specilization = request.form.get('specilization')
    mbaPercentage = request.form.get('mbaPercentage')

    inputData = np.array([[gender,sscPercentage,sscBoard,
                           hscPercentage, hscBoard, hscStream,
                           degreePercentage, degreeTechnology, workExperience,
                           etestPercentage, specilization, mbaPercentage
                           ]])

    modelOutput = classifier.predict(inputData)[0]

    return jsonify({'Output' : str(modelOutput)})

if __name__ == "__main__":
   app.run(debug = True)
