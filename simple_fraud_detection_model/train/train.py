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
with open('../creds.json') as json_data:
    creds = json.load(json_data)
    
client = MongoClient(creds['connection_string'])

# initialize the database with the credentials
db = edb.Database(creds)

# train a random forest using the data in the date range
ml.train_random_forest(start_date=datetime.datetime(2018, 1, 1), end_date=datetime.datetime(2018,4,1), db=db)