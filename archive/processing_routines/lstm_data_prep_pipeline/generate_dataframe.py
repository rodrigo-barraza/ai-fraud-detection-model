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
    
# get the new credit card and interac requests
all_requests = list(client['ml']['requestEvents60'].find())

# flatten into one object per event
flat_requests = [{'request_id': rs['request_id'], 
                  'request_created': rs['request_created'], 
                  'request_email': rs['request_email'], 
                  'request_type': rs['request_type'], 
                  'event': event} for rs in  all_requests for event in rs['events']]

all_events = pd.DataFrame(json_normalize(flat_requests))

all_events.columns = [col if 'event.' not in col else col.replace('event.','') for col in all_events.columns]

# create a dataframe with the results
df_with_id = all_events

# sort by request id and date
df_with_id = df_with_id.sort_values(by=['request_id','created'])

# calculate the previous event time and the time between events
df_with_id['previous_event_time'] = df_with_id.groupby(['_id'])['created'].shift(1)
df_with_id['time_since_last_event'] = pd.to_numeric(df_with_id['created']-df_with_id['previous_event_time'])*1e-9

# replace string versions of infinity with proper inf object
df_with_id = df_with_id.replace('Infinity', np.inf)

# convert columns that should be to numeric
df_with_id['metadata.amount'] = pd.to_numeric(df_with_id['metadata.amount'])
df_with_id['metadata.rate'] = pd.to_numeric(df_with_id['metadata.rate'])
df_with_id['metadata.cents'] = pd.to_numeric(df_with_id['metadata.cents'])
df_with_id['value'] = pd.to_numeric(df_with_id['value'])

df_with_id.head()

# choose the columns to keep
columns_to_keep = ['request_id',
                   'request_email', 
                   'created', 
                   'eventAction', 
                   'eventCategory', 
                   'eventLabel', 
                   'metadata.addressCity', 
                   'metadata.addressCountry', 
                   'metadata.addressProvince', 
                   'metadata.amount', 
                   'metadata.cents', 
                   'metadata.city', 
                   'metadata.country', 
                   'metadata.currency', 
                   'metadata.instrument', 
                   'metadata.lastTradedPx', 
                   'metadata.mongoResponse.amount',
                   'metadata.mongoResponse.price', 
                   'metadata.mongoResponse.product', 
                   'metadata.price', 
                   'metadata.product', 
                   'metadata.prossessorError.billingDetails.city', 
                   'metadata.prossessorError.billingDetails.country', 
                   'metadata.prossessorError.billingDetails.state', 
                   'metadata.prossessorError.card.type', 
                   'metadata.prossessorResponse.billingDetails.city', 
                   'metadata.prossessorResponse.billingDetails.country', 
                   'metadata.prossessorResponse.billingDetails.province', 
                   'metadata.prossessorResponse.billingDetails.state', 
                   'metadata.prossessorResponse.card.type', 
                   #'metadata.prossessorResponse.cardType', 
                   'metadata.prossessorResponse.card_type', 
                   'metadata.prossessorResponse.charge_amount', 
                   'metadata.province', 
                   'metadata.rate', 
                   'metadata.requestParams.amount', 
                   'metadata.requestParams.charge_amount', 
                   'metadata.requestParams.currency', 
                   'metadata.requestParams.price', 
                   'metadata.requestParams.product', 
                   'metadata.requestParams.product_amount', 
                   'metadata.secondAmount', 
                   'metadata.tradesResponse', 
                   'metadata.type', 
                   #'previous_event_time',  
                   'time_since_last_event', 
                   'value']

# choose the columns to expand
columns_to_expand = ['eventAction', 
                     'eventCategory', 
                     'eventLabel', 
                     'metadata.addressCity', 
                     'metadata.addressCountry', 
                     'metadata.addressProvince', 
                     'metadata.city', 
                     'metadata.country', 
                     'metadata.currency', 
                     'metadata.instrument', 
                     'metadata.mongoResponse.product', 
                     'metadata.product', 
                     'metadata.prossessorError.billingDetails.city', 
                     'metadata.prossessorError.billingDetails.country', 
                     'metadata.prossessorError.billingDetails.state', 
                     'metadata.prossessorError.card.type', 
                     'metadata.prossessorResponse.billingDetails.city', 
                     'metadata.prossessorResponse.billingDetails.country', 
                     'metadata.prossessorResponse.billingDetails.province', 
                     'metadata.prossessorResponse.billingDetails.state', 
                     'metadata.prossessorResponse.card.type', 
                     #'metadata.prossessorResponse.cardType', 
                     'metadata.prossessorResponse.card_type', 
                     'metadata.province', 
                     'metadata.requestParams.currency', 
                     'metadata.requestParams.product', 
                     'metadata.tradesResponse', 
                     'metadata.type']

# get the subset of columns
subset = df_with_id[columns_to_keep]

# fill inf values with na        
subset = subset.replace([np.inf, -np.inf], np.nan)

# create columns to track na status of each column
for column in subset.columns:
    if column not in ['request_id','request_email','created']:
        subset[column+"_NaN"] = subset[column].isna().astype(int)
        
# convert categorical columns to binary
subset = pd.get_dummies(subset, columns=columns_to_expand)

# fill na values with 0
subset = subset.fillna(0)

# convert the datetime to integer nanoseconds since 1970
subset['created_utc_ns'] = pd.to_numeric(subset['created'])

# sort ascending by request_id and descending by time
subset = subset.sort_values(by=['request_id','created'], ascending=[True, False])

# grab the emails
user_emails = subset['request_email']

# get the fraudulent emails
def get_fraud_labels(user_emails):
    '''Remove whitelisted or test emails'''

    # get all the events related to the requests aka within 60 minutes before the first request for the users who mader requests
    bl_emails = [r['email'] for r in list(client['production']['emailBlacklistCollection'].find({'level': 'BLOCKED'}))]
 
    return np.array([1 if user in bl_emails else 0 for user in user_emails])

# get the fraud labels
fraud = get_fraud_labels(user_emails)

# drop the time and email columns
subset = subset.drop(['request_email','created'], axis = 1)
subset = subset.reset_index(drop=True)

# hacky way to number the sequence events prior to each interac request with 0 being the request and 10 being the 10th event prior to the request
subset['index_int'] = subset.index
event_index = subset.groupby('request_id')['index_int'].agg(lambda x: list(np.abs(min(x)-x)))
all_index = [[item] if type(item) == type(int) else list(item) for item in event_index]
all_index = [item for sublist in all_index for item in sublist]
subset['timesteps'] = all_index

print("Dataframe is",np.sum(subset.memory_usage())*1e-9,'gigabytes in memory')

print("Saving to hdf5 file")

subset.to_hdf('../lstm_data_prep_pipeline/results/all_request_events.hdf5', 'table')

print("Done")