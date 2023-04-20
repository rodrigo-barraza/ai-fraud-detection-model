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
from bson import ObjectId

import time

from einsteinds.utils import summarize_request_set

# load the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# initialize the client
client = MongoClient(creds['connection_string'])

from confluent_kafka import Consumer, KafkaError

c = Consumer({
    'bootstrap.servers': 'kafka:29092',
    'group.id': 'einstein',
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
    }
})

c.subscribe(['request_sets'])

p = Producer({'bootstrap.servers': 'kafka:29092'})

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Event delivered to {} [{}]'.format(msg.topic(), msg.partition()))

while True:
    msg = c.poll(1.0)

    if msg is None:
        continue
    if msg.error():
        if msg.error().code() == KafkaError._PARTITION_EOF:
            continue
        else:
            print(msg.error())
            break

    # decode the message
    request_set = json_util.loads(msg.value().decode('utf-8'))

    # generate the request set summary
    request_summary = summarize_request_set(request_set)

    print(request_summary)

    # set the key as interac or buy
    key = request_summary['request']['transaction_type']
    
    # Trigger any available delivery report callbacks from previous produce() calls
    p.poll(0)

    # Asynchronously produce a message, the delivery report callback
    # will be triggered from poll() above, or flush() below, when the message has
    # been successfully delivered or failed permanently.
    p.produce('request_summaries', key=key.encode('utf-8'), value=json_util.dumps(request_set).encode('utf-8'), callback=delivery_report)

    
# Wait for any outstanding messages to be delivered and delivery report
# callbacks to be triggered.
p.flush()
c.close()





    

