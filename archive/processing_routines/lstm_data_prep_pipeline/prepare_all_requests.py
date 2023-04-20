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

print("Getting all interac and credit card requests from the eventCollection")
# get the full history of interac and credit card requests
all_requests = list(client['production']['eventCollection'].find({
        'eventCategory': {'$in': ['interac', 'buy']},
        'eventAction': 'request',
        'metadata.email': {'$ne': None}}))

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

_ = [add_event(event) for event in all_user_events]

def get_request_user_events(request, all_user_events):
    '''Gets all the user events that are within 60 minutes before the request'''
    
    email = request['metadata']['email']
    time = request['created']
    lookback = time - datetime.timedelta(seconds=60*60)
    
    events = [event for event in user_event_dict[email] if (event['created'] >= lookback) & (event['created'] <= time)]
    
    return {'request_id': request['_id'], 'request_created': time, 'request_email': email, 'events': events}


print("Processing records into dataframe")

# generate the sets of events for each interac or credit card request
request_sets = [get_request_user_events(request, all_user_events) for request in all_requests]

# flatten into one object per event
flat_requests = [{'request_id': rs['request_id'], 'request_created': rs['request_created'], 'request_email': rs['request_email'], 'event': event} for rs in request_sets for event in rs['events']]

len(flat_requests)

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

# drop the time and email columns
subset = subset.drop(['request_email','created'], axis = 1)
subset = subset.reset_index(drop=True)

# hacky way to number the sequence events prior to each interac request with 0 being the request and 10 being the 10th event prior to the request
subset['index_int'] = subset.index
event_index = subset.groupby('request_id')['index_int'].agg(lambda x: list(np.abs(min(x)-x)))
all_index = [[item] if type(item) == type(int) else list(item) for item in event_index]
all_index = [item for sublist in all_index for item in sublist]
subset['timesteps'] = all_index

ids = subset.request_id.unique()
ints = np.arange(0, len(ids))

request_id_mapper = dict(zip(ids, ints))

# get the dimensions of the data
n_examples = subset.request_id.unique().size
n_timesteps = subset.timesteps.max()+1 # because the indexing starts at 0 so the event with index 10 is actually the 11th event
n_features = subset.columns.drop(['request_id','index_int','timesteps']).size
features = list(subset.columns.drop(['request_id','index_int','timesteps']))

# create a boolean map to set the na_columns to 1
na_col_map = np.array([i for i, col in enumerate(features) if '_NaN' in col])

# set up the empty dataframe
data = np.zeros((n_examples,n_timesteps,n_features))

# initialize all the na cols to 1                      
# data[:,:,na_col_map] = 1.0

# get only the features
data_df = subset[features]

# for each row in the dataframe of request events
for i in range(data_df.shape[0]):
    
    # print out a status every 1000 records
    if (i+1)%1000 == 0:
        print(i, 'events processed out of', data_df.shape[0])
    # get the example index
    example_id = int(request_id_mapper[subset.request_id.iloc[i]])
    
    # get the timestep index
    timestep_id = int(subset.timesteps.iloc[i])
    
    # get the values index
    values = data_df.iloc[i,:].values.astype('float64')
    
    # update the values of that example
    data[example_id, timestep_id,:] = values

print("Shape of dataset:",data.shape)

def get_fraud_labels(user_emails):
    '''Remove whitelisted or test emails'''

    # get all the events related to the requests aka within 60 minutes before the first request for the users who mader requests
    bl_emails = [r['email'] for r in list(client['production']['emailBlacklistCollection'].find({'level': 'BLOCKED'}))]
 
    return np.array([1 if user in bl_emails else 0 for user in user_emails])

# get the fraud labels
fraud = get_fraud_labels(user_emails)

# set up data
X = data
y = fraud
groups = np.array(user_emails)

# create a dict with the results
results_dict = {'X': X,
                'y': y, 
                'groups': groups, 
                'feature_names': np.array(data_df.columns)}

print("Saving results to a pickle file.")

# save results to a pickle file
with open("../lstm_data_prep_pipeline/results/all_request_results.pickle", 'wb') as outfile:
    pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)