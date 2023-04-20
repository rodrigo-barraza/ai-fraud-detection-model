# tests the api

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pymongo

import json

from pymongo import MongoClient

from bson.objectid import ObjectId

import requests

# read the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# initialize a connection to the MongoDB
client = MongoClient(creds['connection_string'])

latest_request = client['production']['eventCollection'].find({'eventCategory': {'$in': ['buy','interac']}}).sort([('created', -1)]).limit(1)[0]

latest_request_id  = str(latest_request['_id'])

print(latest_request_id)

r = requests.post("http://0.0.0.0:5000/process_single_request", json={'request_id': latest_request_id}, )
print(r.status_code, r.reason)
print(r.text[:300] + '...')

