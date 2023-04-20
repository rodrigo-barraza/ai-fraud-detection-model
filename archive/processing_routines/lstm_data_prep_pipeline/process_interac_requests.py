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

# set up a database with credentials
client = MongoClient(creds['connection_string'])

# read the saved data instead of reading from the database
all_events = pd.read_csv('data/all_events.csv', low_memory=False)

# rename the older 'bitcoin' events to BTC so they match
all_events.loc[all_events.eventLabel.str.lower() == 'bitcoin', 'eventLabel'] = 'BTC'

# convert created to a datetime instead of a string
all_events['created'] = pd.to_datetime(all_events['created'])

# get all of the interac requests
interac_requests = all_events[(all_events['metadata.email'].isnull() == False) & (all_events.eventCategory == 'interac') & (all_events.eventAction == 'request')].reset_index(drop=True)

def flagFraudsters(df):

    blemails = list(pd.DataFrame(json_normalize(list(client['production']['emailBlacklistCollection'].find({'level': 'BLOCKED'})))).email) + ['gaelkevin@hotmail.com', 'royer.8383@gmail.com','adventurous7381@gmail.com']
    
    df['fraud'] = df['metadata.email'].isin(blemails)
    
    return df

def removeWhitelistRecords(df):

    wlemails = pd.DataFrame(json_normalize(list(client['production']['emailWhitelistCollection'].find({'level': 'ALLOWED'})))).email
    
    df = df[df['metadata.email'].str.contains('test') == False]
    df = df[df['metadata.email'].str.contains('fingerfoodstudios') == False]
    df = df[df['metadata.email'].str.contains('einstein.exchange') == False]    
    df = df[df['metadata.email'].isin(wlemails) == False]
    
    return df 

# flag the fraudulnet records and remove the whitelist and test accounts
interac_requests = removeWhitelistRecords(flagFraudsters(interac_requests))

print("Interac Requests Shape:", interac_requests.shape)

result = interac_requests[['_id','metadata.email','created','metadata.amount','metadata.rate','value']]


def subset_by_request(row):
    
    # get the email from the row
    email = row['metadata.email']
    
    # get the created time from the row
    created = row['created']
    
    # get the id from the row
    _id = row['_id']
    
    # time to lookback an hour prior to the request
    lookbacktime = created-datetime.timedelta(seconds=60*60) # one hour 
    
    # get the events for this user where the time is before the request but not later than an hour before the request
    events = all_events[(all_events['metadata.email'] == email) & (all_events['created'] <= created) & (all_events['created'] >= lookbacktime)]
    
    # give them a request_id for later group by operations
    events['request_id'] = _id
    
    # convert the dataframe to a list of json records
    return events.to_dict(orient='records')


# convert the interac results to a dict
result_dict = result.to_dict(orient='records')

# extract the list of events for each request
subsets = list(map(subset_by_request, result_dict))

# flatten the subsets so they aren't nested
subsets_flat = [item for sublist in subsets for item in sublist]

# create a dataframe with the results
df_with_id = pd.DataFrame(subsets_flat)

# create the combined category action and category-action-label fields
df_with_id['ca'] = df_with_id.eventCategory + '_' + df_with_id.eventAction
df_with_id['cal'] = df_with_id.eventCategory + '_' + df_with_id.eventAction+ '_' + df_with_id.eventLabel

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

# choose the columns to keep
columns_to_keep = ['request_id',
                   'metadata.email', 
                   'created', 
                   'ca',
                   'cal', 
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
                   #'metadata.ip', 
                   'metadata.lastTradedPx', 
                   'metadata.mongoResponse.amount', 
                   #'metadata.mongoResponse.email', 
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
                   'metadata.prossessorResponse.cardType', 
                   'metadata.prossessorResponse.card_type', 
                   'metadata.prossessorResponse.charge_amount', 
                   #'metadata.prossessorResponse.email', 
                   'metadata.province', 
                   'metadata.rate', 
                   'metadata.requestParams.amount', 
                   'metadata.requestParams.charge_amount', 
                   'metadata.requestParams.currency', 
                   #'metadata.requestParams.email', 
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
columns_to_expand = ['ca', 
                     'cal', 
                     'eventAction', 
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
                     'metadata.prossessorResponse.cardType', 
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
    if column not in ['request_id','metadata.email','created']:
        subset[column+"_na"] = subset[column].isna().astype(int)
        
# convert categorical columns to binary
subset = pd.get_dummies(subset, columns=columns_to_expand)

# fill na values with 0
subset = subset.fillna(0)

# convert the datetime to integer nanoseconds since 1970
subset['created_utc_ns'] = pd.to_numeric(subset['created'])

# sort ascending by request_id and descending by time
subset = subset.sort_values(by=['request_id','created'], ascending=[True, False])

# drop the time and email columns
subset = subset.drop(['metadata.email','created'], axis = 1)
subset = subset.reset_index(drop=True)

# hacky way to number the sequence events prior to each interac request with 0 being the request and 10 being the 10th event prior to the request
subset['index_int'] = subset.index
event_index = subset.groupby('request_id')['index_int'].agg(lambda x: list(np.abs(min(x)-x)))
all_index = [[item] if type(item) == type(int) else list(item) for item in event_index]
all_index = [item for sublist in all_index for item in sublist]
subset['timesteps'] = all_index

# gather the columns together
melt = subset.melt(id_vars=['request_id','timesteps'])

# number of features is the number of columns at each timestep
n_features = len(melt.variable.unique())

# save the feature names
features = list(melt.variable.unique())

# create new variable names that contain the timesteps
melt['variable'] = melt.timesteps.astype(str) + "_" + melt['variable']

# drop the timesteps column so it doesn't end up as a feature column
melt = melt.drop('timesteps', axis=1)

# convert the value to a numeric value - not sure why this happened
melt['value'] = pd.to_numeric(melt.value)

# spread the values out so there's one row per request the columns contain the timesteps
one_row_per_request = melt.pivot_table(values='value', index=['request_id'], columns=['variable']).reset_index()

# fill NA values for timesteps with zeros
one_row_per_request = one_row_per_request.fillna(0)

# get the labels and groups for a GroupKFold cross validation.
fraud = interac_requests.fraud.astype(int)
emails = interac_requests['metadata.email']

n_examples = one_row_per_request.shape[0]

n_timesteps = (one_row_per_request.shape[1] - 1) / n_features

print("Exampes:",n_examples, "Features:", n_features, "Timesteps:",n_timesteps)

# get the values of the array
data = one_row_per_request.drop('request_id', axis=1).values

# reshape the data from (n_example, n_features*n_timesteps) to (n_examples, n_timesteps, n_features)
data = data.reshape((n_examples, int(n_timesteps), n_features))

print("Number of Frauds per User")
# print the number of interac requests per fraudulent user
for email in emails[fraud == True].unique():
    print(email, np.sum(fraud[emails == email] == True))

# create a dict with the results
results_dict = {'data': data,
                'labels': fraud, 
                'users': emails, 
                'feature_names': features}

# save results to a pickle file
with open("results/lstm_interac_results.pickle", 'wb') as outfile:
    pickle.dump(results_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)