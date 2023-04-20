# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

import pickle

from confluent_kafka import Producer

import bson
from bson import json_util

import time

# load the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# initialize the client
client = MongoClient(creds['connection_string'])

p = Producer({'bootstrap.servers': 'kafka:29092'})

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Event delivered to {} [{}]'.format(msg.topic(), msg.partition()))

last_request = None

while True:

    if last_request == None:
        
        events = list(client['production']['eventCollection'].find({'eventCategory': {
                                                                        '$in': ['buy', 'interac'], 
                                                                        '$ne': 'interac-confirm'}, 
                                                                    'eventAction': 'request'
                                                                    }).sort([('created', -1)]).limit(1000))
        
        if events != None and len(events) > 0:
            last_request = events[-1]

    else:
        events = list(client['production']['eventCollection'].find({'created': {'$gt': last_request['created']},
                                                                    'eventCategory': {'$in': ['buy', 'interac'], '$ne': 'interac-confirm'}, 
                                                                    'eventAction': 'request'}))
        if events != None and len(events) > 0:
            last_request = events[-1]

    for data in events:
            
        key = data['metadata'].get('email') if data['metadata'].get('email') != None else 'noemail'
    
        # Trigger any available delivery report callbacks from previous produce() calls
        p.poll(0)

        # Asynchronously produce a message, the delivery report callback
        # will be triggered from poll() above, or flush() below, when the message has
        # been successfully delivered or failed permanently.
        p.produce('requests', key=key.encode('utf-8'), value=json_util.dumps(data).encode('utf-8'), callback=delivery_report)

    time.sleep(10) # sleep for 10 seconds

    

# Wait for any outstanding messages to be delivered and delivery report
# callbacks to be triggered.
p.flush()