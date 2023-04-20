from flask import Flask
from flask import request
import juno.junoutils as junoutils
import juno.junodb as junodb
import json

import pandas as pd
import numpy as np
import juno.junoutils as junoutils
import juno.junodb as junodb
import json
import datetime

import pickle

import last10minutes

# read the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# load random forest model
with open('models/last10randomforest.pickle', 'rb') as pickle_file:
    random_forest = pickle.load(pickle_file)

app = Flask(__name__)

# controller logic

@app.route("/")
def hello():
    return "Simple Machine Learning API for Einstein.Exchange"

@app.route("/interac_check", methods = ['POST'])
def interac():
    '''
    Should receive a post request with an email in the body and 
    return a response that says if the user is fraudulent or not
    '''

    # pull the email out of the request
    email = request.json['email']
    nth = int(request.json['nth'])


    # get the summary of the last transactions previous 10 minuts of events
    summary = last10minutes.getLastInteractSummary(email, nth)

    print("Columns Match? ",np.array_equal(summary.columns, random_forest['column_order']))

    # if the summary exists
    if type(summary) != type(None):
        # run a prediction through the random forest
        prediction = random_forest['model'].predict(summary[random_forest['column_order']].values)[0]

        if prediction == True:
            return "Fraudulent"
        if prediction == False:
            return "Not Fradulent"
    
    else:
        return "No Events Prior to Request or No Interac Requests"

# model/database logic
if __name__ == '__main__':
    app.run(debug=True)