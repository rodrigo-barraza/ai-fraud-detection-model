from flask import Flask
from flask import request

import json
import datetime

import pymongo
from pymongo import MongoClient

import apiutils

from bson.objectid import ObjectId

import requests

# read the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# initialize a connection to the MongoDB
client = MongoClient(creds['connection_string'])

# initialize the flask app
app = Flask(__name__)

# controller logic
@app.route("/")
def root():
    return """Pre-processes all credit credit card and interact transactions
            by finding the user's previous hour of event history and inserts 
            a new record into the requestEvents60 collection in MongoDB."""
            

@app.route("/process_all_requests", methods = ['POST'])
def process_request():
    '''
    Should receive a post request with an email in the body and 
    return a response that says if the user is fraudulent or not
    '''

    # should add more validation here
    print(json.dumps(request.json))

    # pull the email out of the request
    try: 
        after_time = datetime.datetime.strptime(request.json['after_time'], '%Y-%m-%dT%H:%M:%S.%f')
    except:
        after_time = datetime.date(2017, 11, 1)

    try:
        before_time = datetime.datetime.strptime(request.json['before_time'], '%Y-%m-%dT%H:%M:%S.%f')
    except:
        before_time = datetime.datetime.now()

    # get all the requests between the after and before times
    request_ids = [ObjectId(r['_id']) for r in client['production']['eventCollection'].find({
        'created': {'$gte': after_time, '$lte': before_time}, 
        'eventCategory': {'$in': ['interac', 'buy']},
        'metadata.email': {'$ne': None}
        })]

    print(request_ids)

    # for each request

    for request_id in request_ids:
        # check that the request doesn't already exist within the collection
        if client['ml']['requestEvents60'].count({'request._id': request_id}) <= 0:

            # get the actual request from the event_log
            purchase_request = client['production']['eventCollection'].find_one({'_id': request_id})
            
            # get the email of the user making the request
            email = purchase_request['metadata']['email']

            # get the time of the request
            purchase_request_time = purchase_request['created']

            # get the time to look backwards to (1 hour before the request)
            lookback_time = purchase_request_time - datetime.timedelta(seconds=60*60) # one hour before

            # get the events from the 60 minutes prior to the request for that user
            previous_60_events = list(client['production']['eventCollection'].find({
                'metadata.email': email,
                'created': {
                    '$gte': lookback_time,
                    '$lte': purchase_request_time
                    }
                })
            )
            
            # insert the new request summary into the database
            client['ml']['requestEvents60'].insert_one({
                'request': purchase_request,
                'events': previous_60_events
                })

    return 'Request summaries created.'

# model/database logic
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')