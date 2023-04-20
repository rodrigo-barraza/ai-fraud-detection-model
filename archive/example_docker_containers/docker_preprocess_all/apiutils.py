# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient


# load the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)
    
client = MongoClient(creds['connection_string'])

def create_collection_ifnotexists(client, database_name, collection_name):
    '''Checks if a collection exists within a mongodb and creates it if not'''

    # check if the collection exists
    if collection_name not in client[database_name].collection_names():
        print(collection_name, 'collection did not exist. Creating it now.')
        client[database_name].create_collection(collection_name)