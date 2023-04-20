'''poll_to_kinesis.py

This file demonstrates how to send events to AWS Kinesis Firehose using the Python boto3 library.
This could be used to send existing events in Mongo through a Kinesis Firehose stream to process events for machine learning and other uses.
'''


import boto3
from pymongo import MongoClient
import bson
from bson import json_util
from bson import ObjectId
import json
import time
import sys

# load the database credentials from file
with open('../creds/local_creds.json') as json_data:
    creds = json.load(json_data)

# clients for db and aws
firehose_client = boto3.client('firehose')
mongo_client = MongoClient(creds['connection_string'])
dynamodb_client = boto3.client('dynamodb')

# config variables
DELIVERY_STREAM_NAME = 'einstein-kinesisfh-elasticsearch'
STATE_STORE_TABLE = 'einstein-table-statestore'
MONGO_STATE_KEY = 'mongo_event_collection_to_elastic_search'
MONGO_BATCH_SIZE = 10000
MAX_KINESIS_BATCH_SIZE = 500
MAX_KINESIS_BATCH_BYTES = 4e6*0.30
MAX_KINESIS_RECORD_BYTES = 1000000
MAX_KINESIS_RECORD_LENGTH = 1024000
MAX_BYTES_PER_SECOND = 5000000


def get_next_events(latest_id=None):
    '''Gets MONGO_BATCH_SIZE events from the MongoDB. If latest_id is present it gets events after the latest_id
    
    Keyword Arguments:
        latest_id {bson.ObjectId} -- The id of the last record processed in Mongo. (default: {None})
    
    Returns:
        list -- The list of bson events retrieved from the MongoDB
    '''

    
    if latest_id == None:
        events = list(mongo_client['production']['eventCollection'].find().limit(MONGO_BATCH_SIZE))
    else:
        events = list(mongo_client['production']['eventCollection'].find({'_id': {'$gt': latest_id}}).limit(MONGO_BATCH_SIZE))

    return events


def bytecode_bson_event(event):
    '''Converts a bson event to a utf-8 bytestring.
    
    Arguments:
        event {bson object} -- The event details
    
    Returns:
        bytestring -- The encoded bytestring of the event.
    '''


    encoded = json_util.dumps(event).encode('utf-8')
    
    return encoded


def create_record(event):
    '''Converts an event into a record for insertion into the Kinesis Firehose Stream
    
    Arguments:
        event {bson object} -- The object containing the event info.
    
    Returns:
        record -- The AWS Firehose Record
    '''

    
    record = {'Data': bytecode_bson_event(event)}
    
    return record


def get_byte_size(bytestring):
    '''Calculates the size in bytes of the encoded string.
    
    Arguments:
        bytestring {bytestring} -- The encoded bytestring
    
    Returns:
        int -- Size in bytes of the encoded string
    '''

    
    return sys.getsizeof(bytestring)


def create_record_batches(events):
    '''Creates batches of records to insert into the AWS Kinesis Firehose.
    
    Arguments:
        events {list of bson events} -- The list of events to convert into batches.
    
    Returns:
        list -- The list of record batches.
    '''

    
    record_batches = []
    
    batch_size = 0
    record_count = 0
    current_batch = []
    
    for event in events:
        
        record = create_record(event)
        
        #size = get_byte_size(record['Data'])
        size = sys.getsizeof(json_util.dumps(event))
        if size > MAX_KINESIS_RECORD_BYTES:
            print('Too large - _id: ', str(event['_id']))
            
            continue
        
        if len(json_util.dumps(record).encode('utf-8')) > MAX_KINESIS_RECORD_LENGTH:
            print('Too long - _id: ', str(event['_id']))
            
            continue
        
        if (record_count < MAX_KINESIS_BATCH_SIZE) & ((batch_size + size) < MAX_KINESIS_BATCH_BYTES):
            
            current_batch.append(record)
            batch_size += size
            record_count +=1
        
        else:
            print('Batch: Size  - {}, Records - {}'.format(batch_size, record_count))
            record_batches.append({'batch': current_batch, 'size': batch_size, 'records': record_count})
            current_batch = [record]
            batch_size = size
            record_count = 1
            
    record_batches.append({'batch': current_batch, 'size': batch_size, 'records': record_count})
            
    return record_batches


def put_batch_to_firehose(record_batch):
    '''Inserts a record batch into the AWS Kinesis Firehose Stream
    
    Arguments:
        record_batch {list} -- The list of records to insert.
    
    Returns:
        response -- dictionary containing the response from the batch insert operation.
    '''

    
    print("Trying batch with {} records, total size {} bytes".format(record_batch['records'], record_batch['size']))
    # try sending the batch to the stream 
    response = firehose_client.put_record_batch(
        DeliveryStreamName=DELIVERY_STREAM_NAME,
        Records=record_batch['batch']
    )
    
    # sleep in propotion to the max amount of records per second
    pause = max(0.5,record_batch['size']/MAX_BYTES_PER_SECOND)
    print('Pausing {} seconds'.format(pause))
    time.sleep(pause)
    
    # if there were failed records
    if response['FailedPutCount'] > 0:
        print('{} records failed'.format(str(response['FailedPutCount'])))
        print(response)
        rr = response['RequestResponses']
    
        for i, r in enumerate(rr):
            if r == {'ErrorCode': 'ServiceUnavailableException', 'ErrorMessage': 'Slow down.'}:
                break
        
        sub_batch_records = record_batch['batch'][i:]
        sub_batch_size = sum([get_byte_size(r['Data']) for r in sub_batch_records])
        sub_batch_n = len(sub_batch_records)
        
        sub_batch = {'batch': sub_batch_records, 'size':  sub_batch_size, 'records': sub_batch_n}
        
        sub_response = put_batch_to_firehose(sub_batch)
    
        response = [response, sub_response]
    
    return response


def pause_on_error(response, pause_time):
    '''Reads the response from the input, and pauses if there is a slow down warning from the Kinesis Firehose Service.
    
    Arguments:
        response {json} -- The response from the AWS Firehose Service
        pause_time {float} -- The amount of time to pause in seconds if the response contains a slow down error.
    
    Returns:
        float -- The number of seconds to pause on the next error.
    '''


    error = response['RequestResponses'][-1].get('ErrorMessage')

    if error != None and error == 'Slow down.':
        time.sleep(pause_time)
        print(error, 'Pausing {} seconds'.format(str(pause_time)))
        return True, pause_time
    else:
        return False, 0

    
def put_events_to_firehose(events):
    '''Put a set of events into the AWS Kinesis Firehose.
    
    Arguments:
        events {list} -- The list of bson objects to insert.
    
    Returns:
        json -- The response from the API calls to insert records into the AWS Firehose. 
    '''

    
    results = []
    
    record_batches = create_record_batches(events)
    
    for batch in record_batches:
        results.append(put_batch_to_firehose(batch))
    
    return results


def update_state(latest_id):
    '''Updates a DynamoDB table that contains the id of the last processed event.
    
    Arguments:
        latest_id {bson.ObjectId} -- The latest id to update the DynamoDB table with.
    
    Returns:
        response -- Returns the response from the API call to DynamoDB
    '''

    
    response = dynamodb_client.put_item(
        Item={
            'state_key': {
                'S': MONGO_STATE_KEY,
            },
            'last_id': {
                'S': str(latest_id),
            }
        },
        ReturnConsumedCapacity='TOTAL',
        TableName=STATE_STORE_TABLE,
    )
    
    return response


def get_last_state():
    '''Retrieve the latest processed id from the DynamoDB table.
    
    Returns:
        bson.ObjectId -- The latest id processed.
    '''

    
    response = dynamodb_client.get_item(
        TableName=STATE_STORE_TABLE,
        Key={
            'state_key': {
                'S': MONGO_STATE_KEY
            }
        },
        AttributesToGet=[
            'last_id',
        ],
    )
    
    try:
        latest_id = response['Item']['last_id']['S']
    except:
        return None
    
    return ObjectId(latest_id)


def poll_mongo_to_kinesis():
    '''Repeatedly polls the MongoDB and inserts new records into the AWS Kinesis Firehose Stream.
    
    Returns:
        json -- The responses from all the calls to the API to insert records into the AWS Kinesis Firehose.
    '''

    
    total_records_processed = 0
    
    all_results = []
    
    last_id = get_last_state()
    
    next_events = get_next_events(last_id)
    
    while len(next_events) > 0:
        
        all_results.append(put_events_to_firehose(next_events))
        last_id = next_events[-1]['_id']
        all_results.append(update_state(last_id))
        
        total_records_processed += len(next_events)
        
        next_events = get_next_events(last_id)
        
        print("Total events processed from Mongo -> Kinesis Firehose -> S3:", total_records_processed)
        print("Last Mongo id processed:", str(last_id))
    
    return all_results


def put_single_record(record):
    '''Insert a single record into the AWS Kinesis Firehose Stream
    
    Arguments:
        record {record object} -- The encoded record to insert into the AWS Kinesis Firehose.
    
    Returns:
        json -- The response from the API after the insert.
    '''

    
    response = firehose_client.put_record(
        DeliveryStreamName=DELIVERY_STREAM_NAME,
        Record=record
    )
    
    return response


results = poll_mongo_to_kinesis()