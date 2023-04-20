# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

# load the database credentials from file
with open('../creds/creds.json') as json_data:
    creds = json.load(json_data)

# set up a database with credentials
client = MongoClient(creds['connection_string'])

print("Downloading all events from the database")
# download all events from the mongo database
all_events_json = list(client['production']['eventCollection'].find())

print("Flattening events")
# flatten the events
all_events_flat = json_normalize(all_events_json)

print("Creating a dataframe of events")
# convert to a pandas dataframe
all_events = pd.DataFrame(all_events_flat)

print("Saving events to csv file")
# save the data to a file so if the kernel crashes you don't need to re-read from the database
# need to have a data directory inside notebooks
all_events = all_events.to_csv('data/all_events.csv', index=False)

print("Done")