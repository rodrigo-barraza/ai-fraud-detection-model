# Connects to MongoDB and produces anonymized random user sessions based off the real user data. Maximum time between events is cut to 5 minutes.


# for numerical work
import pandas as pd
import numpy as np

import pymongo

import datetime
import time

import json

from pandas.io.json import json_normalize
from pymongo import MongoClient

from confluent_kafka import Producer

import bson
from bson import json_util

import hashlib

import re

# load the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

# initialize the client
client = MongoClient(creds['connection_string'])

ec = client['production']['eventCollection']

producer = Producer({'bootstrap.servers': 'kafka:29092'})

# manually found data when sessions were first implemented
session_implementation_date = datetime.datetime(year=2018, month=3, day=3)

# define the fields that need to be anonymized and convert into a dictionary
anon_fields = [
    {"field": "metadata.addressCity", "dtype": "string"},
{"field": "metadata.addressCountry", "dtype": "string"},
{"field": "metadata.addressPostal", "dtype": "string"},
{"field": "metadata.addressProvince", "dtype": "string"},
{"field": "metadata.addressStreet", "dtype": "string"},
{"field": "metadata.authResponseAP.UserId", "dtype": "int"},
{"field": "metadata.cardNumberLastFour", "dtype": "int"},
{"field": "metadata.customerNumber", "dtype": "int"},
{"field": "metadata.email", "dtype": "email"},
{"field": "metadata.firstName", "dtype": "string"},
{"field": "metadata.ip", "dtype": "string"},
{"field": "metadata.lastName", "dtype": "string"},
{"field": "metadata.mongoResponse.email", "dtype": "email"},
{"field": "metadata.paysafeCardId", "dtype": "int"},
{"field": "metadata.paysaferProfileId", "dtype": "int"},
{"field": "metadata.prossessorResponse.addresses", "dtype": "string"},
{"field": "metadata.prossessorResponse.authentication.cavv", "dtype": "int"},
{"field": "metadata.prossessorResponse.billingDetails.city", "dtype": "string"},
{"field": "metadata.prossessorResponse.billingDetails.country", "dtype": "string"},
{"field": "metadata.prossessorResponse.billingDetails.street", "dtype": "string"},
{"field": "metadata.prossessorResponse.billingDetails.zip", "dtype": "string"},
{"field": "metadata.prossessorResponse.card.cardExpiry.month", "dtype": "int"},
{"field": "metadata.prossessorResponse.card.cardExpiry.year", "dtype": "int"},
{"field": "metadata.prossessorResponse.card.cardType", "dtype": "string"},
{"field": "metadata.prossessorResponse.card.lastDigits", "dtype": "int"},
{"field": "metadata.prossessorResponse.card.type", "dtype": "string"},
{"field": "metadata.prossessorResponse.cardExpiry.month", "dtype": "int"},
{"field": "metadata.prossessorResponse.cardExpiry.year", "dtype": "int"},
{"field": "metadata.prossessorResponse.cardType", "dtype": "string"},
{"field": "metadata.prossessorResponse.cards", "dtype": "string"},
{"field": "metadata.prossessorResponse.cavv", "dtype": "int"},
{"field": "metadata.prossessorResponse.cvv", "dtype": "int"},
{"field": "metadata.prossessorResponse.email", "dtype": "email"},
{"field": "metadata.prossessorResponse.holderName", "dtype": "string"},
{"field": "metadata.prossessorResponse.lastDigits", "dtype": "int"},
{"field": "metadata.prossessorResponse.profile.email", "dtype": "email"},
{"field": "metadata.requestParams.email", "dtype": "email"},
{"field": "metadata.wallet", "dtype": "string"},
]

anon_dict = {}

for field in anon_fields:
    anon_dict[field['field']] = field['dtype']

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

def get_session_list(client):

    ec = client['production']['eventCollection']
    
    session_ids = [session['_id'] for session in ec.aggregate( [
       {'$group' : {'_id' : "$metadata.sessionId"} }])]
    
    return session_ids

def get_sessions_with_request(client):

    ec = client['production']['eventCollection']

    session_ids = [event['metadata']['sessionId'] for event in ec.find({'metadata.sessionId': {'$ne': None}, 'eventAction': 'click', 'eventLabel': 'submit-purchase-request'})]
    
    return session_ids

def random_session(session_list):
    
    #np.random.seed(datetime.datetime.now().timestamp()/1000)
    
    return np.random.choice(session_list, size=1)[0]

def get_session_events(session, client):

    ec = client['production']['eventCollection']
    
    # get the events by session id
    session_events = [event for event in ec.find({'metadata.sessionId': session}).sort([('created',1)])]
    
    # get the time of the first and last event
    start_time = session_events[0]['created']
    end_time = session_events[-1]['created']
    
    # get emails from the session events if they exist
    user_emails = [event['metadata']['email'] for event in session_events if (event.get('metadata') != None and event.get('metadata').get('email') != None and event.get('metadata').get('email') != '')]
    
    # if there's emails
    if len(user_emails) > 0:

        # get the most common email
        email = max(set(user_emails)-set(['',None]), key=user_emails.count)
        print(email)

        if email not in ['', None]:
        
            # get the events by the user during the session time period but where there's no sessionId (not perfect - could break down with concurrent sessions by same user)
            events_by_email = list(ec.find({'metadata.email': email, 
                                            'created': {'$gte': start_time, '$lte': end_time}, 
                                            'metadata.sessionId': None}).sort([('created',1)]))
            
            if len(events_by_email) > 0:
                session_events += events_by_email

            for event in session_events:
                if event.get('metadata') != None:
                    event['metadata']['email'] = email
            
    return sorted(session_events, key=lambda event: event['created'])

def hash_string(string):
    
    hash_object = hashlib.md5(string.encode())
    
    return str(hash_object.hexdigest())[0:len(string)]

def hash_int(int_string):
    
    hash_object = hashlib.sha512(int_string.encode('utf-8'), )
    
    numbers = re.sub(r"\D", "", str(hash_object.hexdigest()))
    
    return numbers[0:len(int_string)]
    
def hash_email(email):
    
    if email == None or email == '':
        hashed_email = ''
    else:
    
        email_parts = email.split('@')
        email_start = email_parts[0]
        email_end = email_parts[1]

        domain = email_end.split('.')
        domain_start = domain[0]
        domain_end = domain[1]

        hashed_email = hash_string(email_start)+'@'+hash_string(domain_start)+'.'+hash_string(domain_end)
    
    return hashed_email


def anonymize_event(event, current_level=''):
    '''
    Anonymizes an event from the database by removing sensitive information about users stored in
    the event such as emails and credit card digits.
    '''
        
    for key in event.keys():
        
        if current_level == '':
            current_key = key
        else:
            current_key = current_level+'.'+key
            
        value = event[key]
        
        if type(value) == type({}):
            
            event[key] = anonymize_event(value, current_level=current_key)
        else:
            
            if anon_dict.get(current_key) != None:
                dtype = anon_dict[current_key]
                
                try:
                    if dtype == 'string':

                        event[key] = hash_string(str(value))

                    elif dtype == 'email':
                        event[key] = hash_email(str(value))

                    else: # dtype is int
                        event[key] = hash_int(str(value))
                except e:
                    print(e)
                    
    return event

def seconds_interval(diff):

    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis/1000


def produce_events(events, session):
    '''
    Produces a set of anonymized session events to a Kafka topic to mimic a real user session.
    '''
    
    n_events = len(events)
    
    # for each event in the session
    for i, event in enumerate(events):
        
        # anonymize the event
        event = anonymize_event(event)
        
        # save the event time for calculating pause below
        actual_event_time = event['created']
        
        # update the event time to now
        event['created'] = datetime.datetime.now()
        
        # # generate the event key
        # key = event['eventCategory']+'_'+event['eventAction']
        
        if event['metadata'].get('email') != None:
            key = event['metadata'].get('email')
        else:
            key = 'noemail'

        # Trigger any available delivery report callbacks from previous produce() calls
        producer.poll(0)

        # Asynchronously produce a message, the delivery report callback
        # will be triggered from poll() above, or flush() below, when the message has
        # been successfully delivered or failed permanently.
        producer.produce('events', key=key.encode('utf-8'), value=json_util.dumps(event).encode('utf-8'), callback=delivery_report)
        
        # if there's a next event
        if i < n_events-1:
            
            # figure out how long to wait before publishing the next event
            time_until_next = seconds_interval(events[i+1]['created']-actual_event_time)
            
            # if the time until the next event is too long just make it 5 minutes (prevents things from hanging on long sessions)
            if time_until_next > 5*60:
                time_until_next = 5*60
            
            print('Produced event for Session: {}. Wating {} seconds until next event'.format(session, time_until_next))
            
            # wait to publish the next event
            #time.sleep(time_until_next)
            time.sleep(5)
            
    # Wait for any outstanding messages to be delivered and delivery report
    # callbacks to be triggered.
    producer.flush()
    
    return 'Done'


def produce_random_sessions():

    # initialize the client
    client = MongoClient(creds['connection_string'])
    
    # get the current list of unique sessions
    #unique = get_session_list(client)

    unique = get_sessions_with_request(client)
    
    while True:
        
        session = random_session(unique)
        events = get_session_events(session, client)
        
        print('Starting to produce Session: {}'.format(session))
        result = produce_events(events, session)
        print('Finished producing {} events for Session: {}'.format(len(events), session))


def main():
    produce_random_sessions()


if __name__ == "__main__":
    main()






