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

from pymongo import InsertOne, DeleteOne, ReplaceOne
from pymongo.errors import BulkWriteError

# load the database credentials from file
with open('../creds/creds.json') as json_data:
    creds = json.load(json_data)

# initialize the client
client = MongoClient(creds['connection_string'])

# dictionary to hold user events
user_events_dict = {}

def lookback_from(time):
    
    return time - datetime.timedelta(seconds=60*60)


def add_request_to_dict(event):
    
    if event.get('metadata') != None and event.get('metadata').get('email') != None:
    
        user_email = event['metadata']['email']
        
        if user_events_dict.get(user_email) == None:
            user_events_dict[user_email] = {'requests': [], 
                                            'max': event['created'], 
                                            'min': event['created'], 
                                            'lookback': event['created'] - datetime.timedelta(seconds=60*60),
                                            'events': []
                                           }
            
        user_events_dict[user_email]['requests'].append(event)
        user_events_dict[user_email]['max'] = max(user_events_dict[user_email]['max'], event['created'])
        user_events_dict[user_email]['min'] = min(user_events_dict[user_email]['min'], event['created'])
        user_events_dict[user_email]['lookback'] = user_events_dict[user_email]['min'] - datetime.timedelta(seconds=60*60)
        
        return True
    
    return False

def add_event_to_dict(event):
    
    if event.get('metadata') != None and event.get('metadata').get('email') != None:
    
        user_email = event['metadata']['email']
        
        user_events_dict[user_email]['events'].append(event)
        
        return True
    
    return False

def generate_request_set(min_time, max_time, events):
    
    return [e for e in events if e['created'] >= min_time and e['created'] < max_time]


def generate_user_request_sets(user_email):
    
    uo = user_events_dict[user_email]
    
    requests = uo['requests']
    
    user_events = uo['events']
    
    rsets = [{
        'request_type': request['eventCategory'],
        'user_email': user_email,
        'request': request,
        'events': generate_request_set(lookback_from(request['created']), request['created'], user_events)
    } for request in requests]
    
    return rsets


def insert_user_request_sets(rsets):
    
    bulk_replaces = [ReplaceOne({"request._id": rset['request']['_id']}, rset, upsert=True) for rset in rsets]
    
    try:
        client['ml']['requestEvents60'].bulk_write(bulk_replaces)
    except BulkWriteError as bwe:
        print(bwe.details)
        
        return False
    
    return True


def generate_all_request_sets():
    
    results = []
    
    count = 0
    
    for user_email in user_events_dict.keys():
        
        user_rsets = generate_user_request_sets(user_email)
        
        results.append((user_email,insert_user_request_sets(user_rsets)))
        
        if count%10 == 0:
            print(count)
        
        count += 1
        
    return results

def main():
    
    print("Getting all requests from the database.")
    # get all the requests from 
    all_requests = list(client['production']['eventCollection'].find({'eventCategory': {
                                                                            '$in': ['buy', 'interac'], 
                                                                            '$ne': 'interac-confirm'}, 
                                                                        'eventAction': 'request'
                                                                        }).sort([('created', -1)]))
    
    print("Processing {} requests".format(len(all_requests)))
    # process requests into the dictionary
    results = [add_request_to_dict(request) for request in all_requests]

    # get all the datetimes from the requests
    times = [request['created'] for request in all_requests]

    # get the first request and last request time
    max_request_time = np.max(times)
    min_request_time = np.min(times)

    # get the time 1 hour before the earliest request
    global_lookback = lookback_from(min_request_time)

    # get the list of users who have made requests
    users = list(user_events_dict.keys())
    print('{} users have made deposit requests'.format(len(users)))
    
    print('Getting all events related to requests')
    # get all the events that relate to all of the requests
    all_events_for_requests = list(client['production']['eventCollection'].find({'metadata.email': {'$in': users}, 
                                                                                 'created': {
                                                                                     '$gte': global_lookback, 
                                                                                     '$lte': max_request_time
                                                                                 }
                                                                                }))
    print("Processing events.")
    # process the events into the dictionary
    add_event_results = [add_event_to_dict(event) for event in all_events_for_requests]
    
    print("Generating request sets and uploading to MongoDB requestEvents60 collection.")
    # process all the requests
    generate_all_request_sets()


if __name__ == "__main__":
    main()