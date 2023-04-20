# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

import pickle

# load the database credentials from file
with open('../creds/creds.json') as json_data:
    creds = json.load(json_data)
    
client = MongoClient(creds['connection_string'])

# check if the collection exists
if 'requestEvents60' not in client['ml'].collection_names():
    print('requestEvents60 collection did not exist. Creating it now.')
    client['ml'].create_collection('requestEvents60')

# get the latest request in the database
max_existing_request_date = list(client['ml']['requestEvents60'].find().sort([('request.created',-1)]).limit(1))

# if there's alread request sets in the database get the max date
if len(max_existing_request_date) > 0:
    
    max_existing_request_date = max_existing_request_date[0]
    
    print("Getting all interac and credit card requests from the eventCollection")
    
    # get the new credit card and interac requests
    all_requests = list(client['production']['eventCollection'].find({
            'created': {'$gt': max_existing_request_date},
            'eventCategory': {'$in': ['interac', 'buy']},
            'eventAction': 'request',
            'metadata.email': {'$ne': None}}))
    
# otherwise get all the requests
else:
    # get the full history of interac and credit card requests
    all_requests = list(client['production']['eventCollection'].find({
            'eventCategory': {'$in': ['interac', 'buy']},
            'eventAction': 'request',
            'metadata.email': {'$ne': None}}))

# if there's requests to process, process them otherwise nothing to do
if len(all_requests) > 0:

    # get all the dates
    dates = [r['created'] for r in all_requests]

    # get the min and max dates
    max_date = max(dates)
    min_date = min(dates)

    # calculate the min and max search dates
    min_search_date = min_date - datetime.timedelta(seconds=60*60) # one hour before the first event
    max_search_date = max_date

    # get all the users
    users = list(set([r['metadata']['email'] for r in all_requests]))

    print("Removing requests by whitelist/test users")

    def remove_whitelist_emails(user_list):
        '''Remove whitelisted or test emails'''

        # get all the events related to the requests aka within 60 minutes before the first request for the users who mader requests
        whitelist_emails = [r['email'] for r in list(client['production']['emailWhitelistCollection'].find({'level': 'ALLOWED'}))]

        return [user for user in user_list if not ((user in whitelist_emails) or ('test' in user) or ('einstein.exchange' in user) or ('fingerfoodstudios' in user))]

    # remove whitelist accounts
    clean_users = remove_whitelist_emails(users)

    # remove events from whitelist users
    all_requests = [r for r in all_requests if r['metadata']['email'] in clean_users]

    print("Getting all user events within 60 minutes of the requests")

    # get all the events related to the requests aka within 60 minutes before the first request for the users who mader requests
    all_user_events = list(client['production']['eventCollection'].find({
            'created': {'$gte': min_search_date, '$lte': max_search_date},
            'metadata.email': {'$in': clean_users}}))

    # generate a list of user events for each user
    user_event_dict = {}

    def add_event(event):
        '''Adds an event to the user_event_dict'''

        email = event['metadata']['email']

        if email in user_event_dict.keys():
            user_event_dict[email].append(event)
        else:
            user_event_dict[email] = [event]

    # populate a dict of events for each user to avoid going through the whole event list for every request
    _ = [add_event(event) for event in all_user_events]

    def get_request_user_events(request, all_user_events):
        '''Gets all the user events that are within 60 minutes before the request'''

        email = request['metadata']['email']
        time = request['created']
        request_type = request['eventCategory'] # record if it's interac or credit card
        lookback = time - datetime.timedelta(seconds=60*60)

        events = [event for event in user_event_dict[email] if (event['created'] >= lookback) & (event['created'] <= time)]

        result = {'request': request, 'events': events}

        return result


    print("Processing records into dataframe")

    # generate the sets of events for each interac or credit card request
    request_sets = [get_request_user_events(request, all_user_events) for request in all_requests]

    print('Inserting new requests into the database')
    # insert the new requests into the database
    col = client['ml']['requestEvents60']
    col.insert_many(documents=request_sets)
    
else:
    
    print("No new requests to process")
    
print('Done')