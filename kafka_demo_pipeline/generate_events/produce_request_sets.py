# Connects to MongoDB and produces anonymized random user sessions based off the real user data. Maximum time between events is cut to 5 minutes.


# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import time

import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

from confluent_kafka import Producer

import bson
from bson import json_util

import hashlib

import re


producer = Producer({'bootstrap.servers': 'localhost:9092'})

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

def produce_events():
    '''
    Produces a set of anonymized session events to a Kafka topic to mimic a real user session.
    '''
    
    n_events = 10

    count = 0
    
    while True:

        # for each event in the session
        for i in range(n_events):
            
            if i == n_events-1:
                event = {
                    "eventCategory": "interac",
                    "eventAction": "request",
                    "metadata": {
                        "email": "test{}@test.com".format(count)
                        }
                }
            
            else:
                event = {
                    "eventCategory": "not_buy",
                    "eventAction": "not_request",
                    "metadata": {
                        "email": "test{}@test.com".format(count)
                        }
                }
            
            key = event["metadata"]["email"]

            # Trigger any available delivery report callbacks from previous produce() calls
            producer.poll(0)

            # Asynchronously produce a message, the delivery report callback
            # will be triggered from poll() above, or flush() below, when the message has
            # been successfully delivered or failed permanently.
            producer.produce('events', key=key.encode('utf-8'), value=json_util.dumps(event).encode('utf-8'), callback=delivery_report)
            
            print(json_util.dumps(event))

            # wait to publish the next event
            time.sleep(1)
        
        count += 1
            
    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    producer.flush()
    
    return 'Done'


def main():
    produce_events()


if __name__ == "__main__":
    main()






