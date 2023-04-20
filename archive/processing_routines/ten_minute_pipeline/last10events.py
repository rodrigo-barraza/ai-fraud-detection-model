import pandas as pd
import numpy as np
import juno.junoutils as junoutils
import juno.junodb as junodb
import json
import datetime

with open('../user_aggregation_pipeline//creds.json') as json_data:
    creds = json.load(json_data)

def getEmptyInteracDF():

    db = junodb.Database(creds)

    # get the unique combinations of eventCategory, eventAction and eventLabel
    cal = list(db._client['production']['eventCollection'].aggregate([{
        '$group': {
            '_id': {
                'eventCategory': "$eventCategory", 
                'eventAction': "$eventAction", 
                'eventLabel': "$eventLabel"
            }
        }
    }]))

    # pull out the combinations, create the category_action and category_action_label combinations
    # translate them into columns of a dataframe to join with user data later on so the correct number of columns in present
    cal = pd.DataFrame(list(map(lambda x: x['_id'],cal)))
    cal = cal[(cal.eventAction != 'start') & (cal.eventCategory != 'session')]
    cal['ca'] = cal.eventCategory + '_' + cal.eventAction
    cal['cal'] = cal.eventCategory + '_' + cal.eventAction + '_' + cal.eventLabel
    cal['eventCategory'] = 'eventCategory_' + cal.eventCategory
    cal['eventAction'] = 'eventAction_' + cal.eventAction
    cal['eventLabel'] = 'eventLabel_' + cal.eventLabel
    cal['ca'] = 'ca_' + cal.ca
    cal['cal'] = 'cal_' + cal.cal
    columns = sorted(list(cal['eventCategory'].unique())+list(cal['eventAction'].unique())+list(cal['eventLabel'].unique())+list(cal['ca'].unique())+list(cal['cal'].unique()))

    init_dict = {}

    for column in columns:
        init_dict[column] = [0]

    empty_df = pd.DataFrame(init_dict)
    empty_df['email'] = 'placeholder'
    
    return empty_df

def getAllInteractSummary():
    # connect to the database
    db = junodb.Database(creds)

    # get the latest interac request
    all_interac_requests = list(db._client['production']['eventCollection'].find({
        'eventCategory': 'interac',
        'eventAction': 'request',
        'metadata.email': {'$ne': None}}))
    
    records = []
    # need to do this to make sure the final record has all the columns
    empty = getEmptyInteracDF().to_dict(orient='records')[0]

    records.append(empty)
    
    for request in all_interac_requests:
        
        email = request['metadata']['email']

        # get the events directly preceeding the transaction
        event_list = list(db._client['production']['eventCollection'].find({'created': {'$lt': request['created']}, '$or': [{'metadata.email': email},
                    {'metadata.prossessorResponse.email': email},
                    {'metadata.prossessorResponse.profile.email': email},
                    {'metadata.requestParams.email': email}]}).limit(15))
        
        if len(event_list) > 10:

            # convert to a dataframe
            event_list = junoutils.flattenObjects(event_list)

            # subset the dataframe and sort by date in descending order
            event_list = event_list[['created','eventCategory','eventAction','eventLabel']].sort_values(by='created')

            # combine event categories
            event_list['ca'] = event_list.eventCategory + "_" + event_list.eventAction
            event_list['cal'] = event_list.eventCategory + "_" + event_list.eventAction + "_" + event_list.eventLabel

            # convert created to datetime
            event_list['created'] = pd.to_datetime(event_list['created'])

            # calculate the previous event time
            event_list['previous_event_time'] = event_list['created'].shift(1)
            event_list = event_list[event_list.previous_event_time.isnull() == False]

            # event list
            event_list = event_list[0:10]

            # calculate the time in seconds since the last event
            event_list['since_last_event'] = (event_list['created']-event_list['previous_event_time']).astype(int)*1e-9 # subtract dates, convert to int and convert from ns to seconds.
            event_list = event_list[['created','previous_event_time','since_last_event','eventCategory','eventAction','eventLabel','ca','cal']]

            # set columns to expand
            categorial = ['eventLabel', 'eventAction', 'eventCategory', 'ca','cal']

            # set numerical columns
            numerical = ['since_last_event']

            # convert categorical columns to binary
            expanded = pd.get_dummies(columns=categorial, data=event_list, drop_first=False)
            expanded.drop(['created','previous_event_time'], axis=1, inplace=True)

            # calculate stats on the previous event time
            time_since = expanded.since_last_event.describe()
            time_since_lables = 'since_last_'+time_since.index
            time_since_values = time_since.values

            # calculate stats on categorical columms
            categorical = expanded.drop('since_last_event', axis=1)
            categorical = categorical.sum()
            categorical_labels = categorical.index
            categorical_values = categorical.values

            # combine all the columns together
            labels = list(time_since_lables)+list(categorical_labels)
            values = list(time_since_values)+list(categorical_values)

            # combine the records together
            record = dict(zip(labels, values))
            record['email'] = email

            records.append(record)        
        
        
    records_df = pd.DataFrame(records)
    
    records = records_df[['email']+list(set(sorted(records_df.columns))-set(['email']))].fillna(0)
    
    return records

def getLastInteractSummary(email):
    # connect to the database
    db = junodb.Database(creds)

    # get the latest interac request
    latest_interac_request = list(db._client['production']['eventCollection'].find({'eventCategory': 'interac', 'eventAction': 'request','$or': [{'metadata.email': email},
                {'metadata.prossessorResponse.email': email},
                {'metadata.prossessorResponse.profile.email': email},
                {'metadata.requestParams.email': email}]}).sort('created', -1).limit(1))[0]

    # get the events directly preceeding the transaction
    event_list = list(db._client['production']['eventCollection'].find({'created': {'$lt': latest_interac_request['created']}, '$or': [{'metadata.email': email},
                {'metadata.prossessorResponse.email': email},
                {'metadata.prossessorResponse.profile.email': email},
                {'metadata.requestParams.email': email}]}).limit(15))

    # convert to a dataframe
    event_list = junoutils.flattenObjects(event_list)

    # subset the dataframe and sort by date in descending order
    event_list = event_list[['created','eventCategory','eventAction','eventLabel']].sort_values(by='created')

    # combine event categories
    event_list['ca'] = event_list.eventCategory + "_" + event_list.eventAction
    event_list['cal'] = event_list.eventCategory + "_" + event_list.eventAction + "_" + event_list.eventLabel

    # convert created to datetime
    event_list['created'] = pd.to_datetime(event_list['created'])

    # calculate the previous event time
    event_list['previous_event_time'] = event_list['created'].shift(1)
    event_list = event_list[event_list.previous_event_time.isnull() == False]
    
    # event list
    event_list = event_list[0:10]

    # calculate the time in seconds since the last event
    event_list['since_last_event'] = (event_list['created']-event_list['previous_event_time']).astype(int)*1e-9 # subtract dates, convert to int and convert from ns to seconds.
    event_list = event_list[['created','previous_event_time','since_last_event','eventCategory','eventAction','eventLabel','ca','cal']]

    # set columns to expand
    categorial = ['eventLabel', 'eventAction', 'eventCategory', 'ca','cal']

    # set numerical columns
    numerical = ['since_last_event']

    # convert categorical columns to binary
    expanded = pd.get_dummies(columns=categorial, data=event_list, drop_first=False)
    expanded.drop(['created','previous_event_time'], axis=1, inplace=True)

    # calculate stats on the previous event time
    time_since = expanded.since_last_event.describe()
    time_since_lables = 'since_last_'+time_since.index
    time_since_values = time_since.values

    # calculate stats on categorical columms
    categorical = expanded.drop('since_last_event', axis=1)
    categorical = categorical.sum()
    categorical_labels = categorical.index
    categorical_values = categorical.values

    # combine all the columns together
    labels = list(time_since_lables)+list(categorical_labels)
    values = list(time_since_values)+list(categorical_values)

    # combine the records together
    record = dict(zip(labels, values))
    record['email'] = email
    record = pd.DataFrame([record])
    getEmptyInteracDF().merge(record, on='email')

    # need to do this to make sure the final record has all the columns
    empty = getEmptyInteracDF()

    # get the record with the full set of columns
    record = empty.set_index('email').merge(record.set_index('email'), how='outer')[1:].fillna(0)
    record = record[sorted(record.columns)]
    
    return record