# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

import pickle

from confluent_kafka import Producer

import bson
from bson import json_util

import time

from pymongo import InsertOne, DeleteOne, ReplaceOne
from pymongo.errors import BulkWriteError

from utils import summarize_request_set

# load the database credentials from file
with open('./creds.json') as json_data:
    creds = json.load(json_data)

# initialize the client
client = MongoClient(creds['connection_string'])


def try_to_summarize(rset):

    try:
        return summarize_request_set(rset)

    except Exception as e:
        s = str(e)
        rset['error'] = s
        client['ml']['requestEvents60SummaryFailures'].replace_one(filter={'request._id': rset['request']['_id']}, 
                                                                    replacement=rset, 
                                                                    upsert=True)
        
        return None


def create_all_request_summaries(rsets):

    summaries = []

    for rset in rsets:
        result = try_to_summarize(rset)
        if result != None:
            summaries.append(result)
    
    return summaries



def insert_all_request_summaries(rsummaries):
    
    bulk_replaces = [ReplaceOne({"request._id": rsummary['request']['_id']}, rsummary, upsert=True) for rsummary in rsummaries]
    
    try:
        client['ml']['requestEvents60Summaries'].bulk_write(bulk_replaces)
        return True

    except Exception as e:
        print(e)
        print("Encountered error. Moving to single record writes to find problem record.")
        
        for rsummary in rsummaries:
            try:

                client['ml']['requestEvents60Summaries'].replace_one(filter={'request._id': rsummary['request']['_id']}, 
                                                                        replacement=rsummary, 
                                                                        upsert=True)
            except Exception as e:

                print(e)
                print("Error Inserting:")

                for col in rsummary.keys():
                    print(col, type(rsummary[col]))

                #print(json_util.dumps(rsummary, indent=4, sort_keys=True))

    return False


def try_to_upsert_one(rset):

    try:
        rsummary = try_to_summarize(rset)

        if rsummary != None:
            client['ml']['requestEvents60Summaries'].replace_one(filter={'request._id': rsummary['request']['_id']}, 
                                                                    replacement=rsummary, 
                                                                    upsert=True)
        else:
            print('Could not summarize: ',rset['request']['_id'])


    except Exception as e:

        print(e)
        print("Error Inserting:")

        for col in rsummary.keys():
            print(col, type(rsummary[col]))

        return False

    return True


def main():

    print('Getting all request sets from the database.')
    all_rsets = list(client['ml']['requestEvents60'].find())


    for i, rset in enumerate(all_rsets):
        _ = try_to_upsert_one(rset)

        if (i+1)%1000 == 0:
            print(i, 'records summarized.')


    # for i in range(0, len(all_rsets), 1000):

    #     if (len(all_rsets) - i) < 1000:
    #         sets = all_rsets[i:]
    #     else:
    #         sets = all_rsets[i:i+1000]

    #     print("Creating summaries for {} request sets.".format(len(sets)))
    #     all_rsummaries = create_all_request_summaries(sets)

    #     print("Inserting request summaries into MongoDB requestEvents60Summaries")
    #     insert_all_request_summaries(all_rsummaries)


if __name__ == "__main__":
    main()