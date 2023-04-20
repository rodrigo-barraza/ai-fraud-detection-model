import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import pymongo

import json

from pymongo import MongoClient

from bson.objectid import ObjectId

import requests

import datetime

from bson import json_util
import json

after_time = (datetime.datetime.now()-datetime.timedelta(days=7)).isoformat()

r = requests.post("http://0.0.0.0:5000/process_all_requests", json={'after_time': after_time})
print(r.status_code, r.reason)
print(r.text[:300] + '...')

