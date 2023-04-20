# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import time
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

import pickle

import bson
from bson import json_util

import math

from einsteinds import db as edb
from einsteinds import event_processing
from einsteinds import ml
from einsteinds import plots
from einsteinds import utils

clean_events = event_processing.clean_events

# load the database credentials from file
with open('../creds/local_creds.json') as json_data:
    creds = json.load(json_data)
    
client = MongoClient(creds['connection_string'])

# initialize the database with the credentials
db = edb.Database(creds)

# load the model and features from file
model = ml.load_random_forest_model('random_forest.p')


LOOKBACK_DURATION_SECONDS = 60*60 # one hour lookback

def last_two_hours():
    
    return datetime.datetime.now() - datetime.timedelta(seconds=LOOKBACK_DURATION_SECONDS*2)

def initial_data():
    
    return db.get_events_in_range(start_date=last_two_hours(), end_date=datetime.datetime.now(), clean=True)

def new_data(event_id):
    
    new_data = db.get_events_after_id(event_id=event_id, clean=True)

    return new_data

def request_sets_from_clean_df(new_df, full_df):
    
    rsets = []
    
    # if there's no new events
    if new_df.shape[0] == 0:
        return []
    
    requests = new_df[new_df.category_action.isin(['interact_request','buy_request'])].to_dict('records')
    
    for request in requests:
        
        user_email = request['user_email']
        created = request['created']
        events = full_df[(full_df.created >= created - datetime.timedelta(seconds=LOOKBACK_DURATION_SECONDS)) & (full_df.created < created) & (full_df.user_email == user_email)]
        
        rsets.append(db.get_clean_deposit_request_set(request, events.to_dict('records')))
    
    return rsets

def log_results(results):
    
    for result in results:
        client.ml.predictions.replace_one(filter={'request_id': result['request_id']}, replacement=result, upsert=True)


def poll_mongo(frequency_s=10):
    
    print("Getting last two hours of history")
    # get the inital two hours of data
    full_df = initial_data()
    
    # get only the last hour to look for requests
    new_df = full_df[full_df.created >= full_df.created.max() - datetime.timedelta(seconds=LOOKBACK_DURATION_SECONDS)]
    
    # get the request_sets
    request_sets = request_sets_from_clean_df(new_df, full_df)
    
    if len(request_sets) > 0:
        print('Generating predictions')
        # generate predictions
        results = ml.predict_from_clean_request_sets(request_sets=request_sets, db=db, model=model).to_dict('records')
        log_results(results)
        print(json_util.dumps(results, indent=2))
    
    time.sleep(frequency_s)
    
    while True:
        print("Getting latest data")
        # get latest data
        new_df = new_data(full_df['_id'].max())
        
        # full df
        full_df = pd.concat([full_df, new_df], sort=False)
        
        # get the request_sets
        request_sets = request_sets_from_clean_df(new_df, full_df)
        
        if len(request_sets) > 0:
            print('Generating predictions')
            # generate predictions
            results = ml.predict_from_clean_request_sets(request_sets=request_sets, db=db, model=model).to_dict('records')
            log_results(results)
            print(json_util.dumps(results, indent=2))
        
        # drop the older data
        max_time = full_df.created.max()
        drop_time = max_time - datetime.timedelta(seconds=LOOKBACK_DURATION_SECONDS)
        full_df = full_df[full_df.created >= drop_time]
        
        # pause
        time.sleep(frequency_s)


poll_mongo()

