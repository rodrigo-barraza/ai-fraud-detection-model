'''db.py

This script wraps pymongo to create some convenient methods to read and process records from Einstein's MongoDB.

'''


# database stuff
import pymongo as mo
from pymongo import MongoClient
import json

import numpy as np

from einsteinds import utils
from einsteinds import event_processing
from einsteinds import ml

import datetime
import time

import pandas as pd

from pandas.io.json import json_normalize

# minimum number of days to search for records.
min_search_days = 7


class Database(object):

    def __init__(self, creds):
        '''Initialize the connection to the database.
        
        Arguments:
            creds {json} -- The credentials to connect to Mongo
        '''

        self._creds = creds
        self._conn_string = creds['connection_string']
        self._client = MongoClient(self._conn_string)


    def init_connection(self, credentials):
        '''Initialize the connection to the database.
        
        Arguments:
            creds {json} -- The credentials to connect to Mongo
        '''

        # load database credentials from file not in git repo
        self._creds = credentials

        # get the connection string
        self._conn_string = self._creds['connection_string']

        # initialize the mongo client
        self._client = MongoClient(self._conn_string)


    def _events(self):
        '''Get a connection the eventCollection table.
        
        Returns:
            mongo collection -- The event collection.
        '''

        return self._client['production']['eventCollection']


    def _blacklist(self):
        '''Get a connection to the emailBlacklistCollection
        
        Returns:
            mongo collection -- The email blacklist collection.
        '''

        return self._client['production']['emailBlacklistCollection']


    def _whitelist(self):
        '''Get a connection to the emailWhitelistCollection
        
        Returns:
            mongo collection -- The email whitelist collection.
        '''

        return self._client['production']['emailWhitelistCollection']


    def _users(self):

        return self._client['production']['userCollection']


    def get_users(self, flat=True, date=None):
        '''Gets a list of all the user JSON objects
        
        This method pulls from the users table, which isn't up to date.
        
        '''

        users = self._users()

        if date == None:
            user_list = list(users.find())
        else:
            user_list = list(users.find({'created': {'$gte': date}}))

        if flat == True:
            user_list = utils.flatten_objects(user_list)

        return user_list


    def get_events(self, flat=True, date=None):
        '''Gets a list of all the event JSON objects'''

        events = self._events()

        if date == None:
            event_list = list(events.find())
        else:
            event_list = list(events.find({'created': {'$gte': date}}))

        if flat == True:
            event_list = utils.flatten_objects(event_list)

        return event_list


    def get_user_events(self, email, flat=True, date=None):
        '''Gets a list of all the event JSON objects for a given user.'''

        if date == None:
            event_list = list(self._events().find({'$or': [{'metadata.email': email},
                {'metadata.prossessorResponse.email': email},
                {'metadata.prossessorResponse.profile.email': email},
                {'metadata.requestParams.email': email}]}))
        else:
            event_list = list(self._events().find({'created': {'$gte': date}, '$or': [{'metadata.email': email},
                {'metadata.prossessorResponse.email': email},
                {'metadata.prossessorResponse.profile.email': email},
                {'metadata.requestParams.email': email}]}))

        if flat == True:
            event_list = utils.flatten_objects(event_list)

        return event_list


    def get_blacklist(self, flat=True):
        '''Gets all the records from the blacklist as a list of json objects.
        
        Keyword Arguments:
            flat {bool} -- True if the result should be a pandas dataframe, False to get a list of json events. (default: {True})
        
        Returns:
            list, pandas.DataFrame -- The blacklist records, either as a list or a dataframe.
        '''

        bl = list(self._blacklist().find())

        if flat == True:
            bl = utils.flatten_objects(bl)

        return bl


    def get_fraudsters(self):
        '''Gets the list of emails that have been flagged as fraudulent.
        
        Returns:
            list -- The list of fraudulent emails, representing user accounts.
        '''


        return [item['email'] for item in self.get_blacklist(flat=False) if item['level'] == "BLOCKED"]


    def add_fraud_label(self, df, email_col):
        '''Takes in a dataframe with an email column and adds a fraud label generated from comparing with the email blacklist.
        
        Arguments:
            df {pandas.DataFrame} -- 
            email_col {string} -- The name of the email column in the dataframe to compare with the blacklist.
        
        Returns:
            pandas.DataFrame -- The input dataframe with the additional fraud label.
        '''

        df = df.copy()

        email_list = self.get_fraudsters()

        df['fraud'] = df[email_col].apply(lambda x: x in email_list)

        return df


    def get_whitelist(self, flat=True):
        '''Gets all the records from the whitelist as a list of json objects.
        
        Keyword Arguments:
            flat {bool} -- True if the result should be a pandas dataframe, False to get a list of json events. (default: {True})
        
        Returns:
            list, pandas.DataFrame -- The whitelist records, either as a list or a dataframe.
        '''
        wl = list(self._whitelist().find())

        if flat == True:
            wl = utils.flatten_objects(wl)

        return wl


    def get_sessions_by_day(self, start_date=None, end_date=None):
        '''Gets the number of sessions per day within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the count of sessions by day.
        '''

        # convert the dates to the same format as mongodb
        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        sessions_by_day = utils.flatten_objects(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 'metadata.sessionId': {'$exists': True}}},
            {'$project': {'day': { '$dateToString': {'format': "%Y-%m-%d", 'date': "$created" } }}},
            {'$group': {
                '_id': {
                    'day': '$day', 'session_id': '$metadata.sessionId'}, 
                'session_id': {'$sum': 1}}
            }
        ])))

        sessions_by_day.columns = ['day','n_sessions']

        return sessions_by_day.sort_values('day').reset_index(drop=True)


    def get_new_users_by_day(self, start_date=None, end_date=None):
        '''Gets the number of new users per day within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the count of new users by day.
        '''
        
        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        # check if the email is not a testing email
        def is_valid(email):
            
            if 'fingerfoodstudios.com' in email:
                return False
            if 'einstein.exchange' in email:
                return False
            if 'test' in email:
                return False
            if '@alican' in email:
                return False
            
            return True
        
        # get the emails that existed before the time period
        emails_before_period = [email for email in self._events().distinct('metadata.email', filter={
            'created': {'$lt': start_date}
            }) if is_valid(email) == True]
        
        
        # get the emails in the search time by day
        emails_by_day = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'metadata.email': {'$exists': True},
                        'eventCategory': {'$ne': 'login'}}},
            {'$project': {'day': { '$dateToString': {'format': "%Y-%m-%d", 'date': "$created" } }, 'email': '$metadata.email'}},
        ])))
        
        new_users = []
        
        # for each day in the search range
        for day in sorted(emails_by_day.day.unique()):
            
            # find the emails used that day
            days_emails = emails_by_day[emails_by_day.day == day].email.unique()
            
            day_count = 0
            
            # for each email
            for email in days_emails:
                
                # if the email has not been used before,
                if email not in emails_before_period and is_valid(email) == True:
                    
                    # increment the new email counter
                    day_count +=1
                    
                    # add the email to the history
                    emails_before_period.append(email)
                    
            new_users.append({'day': day, 'new_users': day_count})
        
        return pd.DataFrame(new_users)


    def get_users_by_day(self, start_date=None, end_date=None):
        '''Gets the number of users active per day within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the count of users active by day.
        '''
        
        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        emails_by_day = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 'metadata.email': {'$exists': True}}},
            {'$project': {'yearMonthDay': { '$dateToString': {'format': "%Y-%m-%d", 'date': "$created" } }}},
            {'$group': {
                '_id': {
                    'yearMonthDay': '$yearMonthDay', 'email': '$metadata.email'}, 
                'email': {'$sum': 1}}
            }
        ])))

        emails_by_day.columns = ['day','n_unique_users']

        return emails_by_day.sort_values('day').reset_index(drop=True)


    def get_activity_by_user(self, start_date=None, end_date=None):
        '''Gets the activity in terms of number of sessions by users in a given time range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the count of sessions by users.
        '''

        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        activity_by_user = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'metadata.email': {'$exists': True, '$nin': [None,'']},
                        'metadata.sessionId': {'$exists': True, '$nin': [None,'']}
                    }},
            {'$group': {
                '_id': {
                    'email': '$metadata.email', 'session_id': '$metadata.sessionId'}, 
                }
            },
            {'$group': {
                '_id': {
                    'email': '$_id.email'
                },
                'email': {'$sum': 1}
            }}
        ])))

        activity_by_user.columns = ['email','n_sessions']

        return activity_by_user.sort_values('n_sessions').reset_index(drop=True)


    def get_deposits_by_day(self, start_date=None, end_date=None):
        '''Gets the number of deposits per day within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the number of deposits by day.
        '''
    
        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        cc_deposits_by_day = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'eventCategory': 'buy',
                        'eventAction': 'transfer-funds'
                    }}
        ])))
        
        interac_deposits_by_day = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'eventCategory': 'interac',
                        'eventAction': 'transfer-funds'
                    }}
        ])))
        
        all_deposits = pd.concat([cc_deposits_by_day, interac_deposits_by_day])
        all_deposits['day'] = all_deposits.created.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
        all_deposits = all_deposits.groupby(['day','eventCategory'])['value'].sum().reset_index()
        all_deposits['value_usd'] = all_deposits['value']/100
        all_deposits['deposit_type'] = all_deposits['eventCategory']
        all_deposits = all_deposits[['day','deposit_type','value_usd']]
        all_deposits.loc[all_deposits.deposit_type == 'buy', 'deposit_type'] = 'credit_card'
        
        return all_deposits.sort_values('day')


    def get_trades_by_day(self, start_date=None, end_date=None):
        '''Gets the number of trades per day within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the number of trades by day.
        '''

        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        trades = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'eventCategory': 'trade',
                        'metadata.tradesResponse': 'Accepted'
                    }}
        ])))
        
        def get_estimated_value(row):
            try:
                if row['metadata.type'] == 'MARKET':
                    return float(row['metadata.lastTradedPx']) * float(row['metadata.amount'])
                else:
                    percent_diff_last_traded_price = abs(float(row['metadata.price']) - float(row['metadata.lastTradedPx']))/float(row['metadata.lastTradedPx'])

                    if percent_diff_last_traded_price > 0.10:
                        return 0
                    else:
                        return float(row['metadata.price']) * float(row['metadata.amount'])
                    
            except:
                return np.nan
        
        trades['metadata.amount'] = pd.to_numeric(trades['metadata.amount'], errors='coerce')
        trades['metadata.price'] = pd.to_numeric(trades['metadata.price'], errors='coerce')
        trades['metadata.lastTradedPx'] = pd.to_numeric(trades['metadata.lastTradedPx'], errors='coerce')
        trades['estimated_value'] = trades.apply(get_estimated_value, axis=1)
        trades['day'] = trades.created.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
        trades = trades[['day','metadata.side','metadata.type','estimated_value']]
        trades.columns = ['day','side','type','estimated_value']

        return trades.groupby(['day','side','type'])['estimated_value'].sum().reset_index().sort_values('day')


    def get_trades(self, start_date=None, end_date=None):
        '''Gets the trades within a certain data range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing the trades within the start_date and end_date.
        '''

        start_date, end_date = utils.get_search_dates(start_date, end_date)
        
        all_trades = json_normalize(list(self._events().aggregate([
            {'$match': {'created': {'$gte': start_date, '$lte': end_date}, 
                        'eventCategory': 'trade'
                    }}
        ])))

        all_trades = all_trades[['_id','created','metadata.email','metadata.type','metadata.side','metadata.against','metadata.instrument','metadata.price','metadata.amount','metadata.lastTradedPx','metadata.tradesResponse']]
        all_trades.columns = ['_id','created','email','trade_type','side','fiat_currency','trading_pair','fiat_currency_price','cryptocurrency_amount','last_traded_price','trade_result']

        def get_estimated_value(row):  
                try:
                    if row['trade_type'] == 'MARKET':
                        return float(row['last_traded_price']) * float(row['cryptocurrency_amount'])
                    else:
                        percent_diff_last_traded_price = abs(row['fiat_currency_price'] - row['last_traded_price'])/row['last_traded_price']

                        if percent_diff_last_traded_price > 0.10:
                            return 0
                        else:
                            return float(row['fiat_currency_price']) * float(row['cryptocurrency_amount'])
                except:
                    return np.nan

        all_trades.fiat_currency_price = pd.to_numeric(all_trades.fiat_currency_price, errors='coerce')
        all_trades.cryptocurrency_amount = pd.to_numeric(all_trades.cryptocurrency_amount, errors='coerce')
        all_trades.last_traded_price = pd.to_numeric(all_trades.last_traded_price, errors='coerce')
        all_trades['estimated_value'] = all_trades.apply(get_estimated_value, axis=1)
        all_trades = all_trades.replace(np.inf, np.nan)

        trade_values = all_trades.sort_values('estimated_value', ascending=False).reset_index(drop=True)
        
        return trade_values


    def get_daily_trades_summary(self, start_date=None, end_date=None):
        '''Gets a summary of the trades by day in a given date range.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            pandas.DataFrame -- Returns a pandas dataframe containing summary of trades by day.
        '''

    
        trade_values = self.get_trades(start_date, end_date)
        
        trade_values['day'] = trade_values.created.apply(lambda x: datetime.datetime(x.year, x.month, x.day))
        
        summary = (trade_values
                .groupby(['day','trading_pair','side','trade_type','trade_result'])[['cryptocurrency_amount','estimated_value','last_traded_price']]
                .aggregate({'cryptocurrency_amount': ['median', 'sum'], 
                            'estimated_value': ['median', 'sum'], 
                            'last_traded_price': ['median']})
                .reset_index())
        
        def join_if_tuple(col):
        
            if col[1] == '':
                return col[0]
            else:
                return "_".join(col)
        
        summary.columns = [join_if_tuple(col) for col in summary.columns.ravel()]
        
        return summary


    def get_user_list(self, start_date=None, end_date=None):
        '''Gets the list of users from the metadata.email field, which includes failed logins and other emails not included in the user collection.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
        
        Returns:
            list -- the list of user emails.
        '''


        start_date, end_date = utils.get_search_dates(start_date, end_date)

        users = list(set(self._events().distinct('metadata.email', filter={
            'created': {
                '$gte': start_date,
                '$lte': end_date
                }
                })) - set(['None','']))

        users = sorted(list(set([user.lower() for user in users if user != None])))

        return users


    def get_events_in_range(self, start_date=None, end_date=None, user=None, clean=False):
        '''Gets the events in a given date range, with the optional ability to get user specific events and whether or not the events should be clean or in their raw format.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
            user {string} -- The email address of the user (default: {None})
            clean {bool} -- If the results should be cleaned or left raw. (default: {False})
        
        Returns:
            list[json] -- List of json events that occurred in the search date range.
        '''

        start_date, end_date = utils.get_search_dates(start_date, end_date)

        if user == None:

            events = list(self._events().find({
                    'created': {
                        '$gte': start_date,
                        '$lte': end_date
                        }
                    }))
        else:

            events = list(self._events().find({
                    'created': {
                        '$gte': start_date,
                        '$lte': end_date
                        },
                    'metadata.email': user
                    }))

        if clean == True:
            events = event_processing.clean_events(events)

        df = None

        if len(events) > 0:
            df = json_normalize(events)
            df = df.sort_values(by='created', ascending=False)
            
            # convert value in cents to value in dollars
            if 'value' in df.columns:

                df['value'] = df['value']/100

            if 'metadata.email' in df.columns:
                df['metaadata.email'] = df['metadata.email'].str.lower()

        return df


    def get_events_after_id(self, event_id, clean=True):
        '''Gets the events after a certain mongo id, and the optional ability to get clean or raw events.
        
        Keyword Arguments:
            event_id {bson.ObjectId} -- The id to search for records after. (default: {None})
            clean {bool} -- If the results should be cleaned or left raw. (default: {False})
        
        Returns:
            list[json] -- List of json events that occurred in the search date range.
        '''

        # get the full history of interac requests for just the user
        events = list(self._events().find({'_id': {'$gt': event_id}}))

        if clean == True:
            events = event_processing.clean_events(events)

        df = pd.DataFrame()

        if len(events) > 0:
            df = json_normalize(events)
            df = df.sort_values(by='created', ascending=False)
            
            # convert value in cents to value in dollars
            if 'value' in df.columns:

                df['value'] = df['value']/100

            if 'metadata.email' in df.columns:
                df['metaadata.email'] = df['metadata.email'].str.lower()

        return df


    def get_deposit_request_sets(self, start_date=None, end_date=None, user=None):
        '''Gets the deposit request sets of events in a given date range, optionally for a specific user.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
            user {string} -- The email address of the user (default: {None})
        
        Returns:
            list[json] -- List of json events of the request sets that occurred in the search date range.
        '''

        requests = self.get_deposit_requests(start_date=start_date, end_date=end_date, user=user)
        rdf = json_normalize(requests)

        min_time = rdf.created.min() - datetime.timedelta(seconds=60*60)
        max_time = rdf.created.max()
        users = sorted(rdf['metadata.email'].dropna().unique())

        events = list(self._events().find({'metadata.email': {'$in': users}, 
                                        'created': {'$gte': min_time, '$lt': max_time}}))

        events_by_user = {}

        for event in events:
            if events_by_user.get(event['metadata']['email']) == None:
                events_by_user[event['metadata']['email']] = [event]
            else:
                events_by_user[event['metadata']['email']].append(event)

        rsets = []

        for request in requests:
            email = request['metadata']['email']
            time = request['created']

            events = [event for event in events_by_user[email] if event['created'] >= time - datetime.timedelta(seconds=60*60) and event['created'] < time]

            rsets.append(self.get_deposit_request_set(request=request, events=events))
 

        return rsets
    

    def get_deposit_requests(self, start_date=None, end_date=None, user=None):
        '''Gets the deposit requests in a given date range, optionally for a specific user.
        
        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
            user {string} -- The email address of the user (default: {None})
        
        Returns:
            list[json] -- List of json events of the requests that occurred in the search date range.
        '''

        start_date, end_date = utils.get_search_dates(start_date, end_date)

        if user == None:
            cc_requests = list(self._events().find({'created': {'$gte': start_date, '$lte': end_date}, 
                            'eventCategory': 'buy',
                            'eventAction': 'request',
                            'metadata.email': {'$exists': True}
                        }))
            
            interac_requests = list(self._events().find({'created': {'$gte': start_date, '$lte': end_date}, 
                            'eventCategory': 'interac',
                            'eventAction': 'request',
                            'metadata.email': {'$exists': True}
                        }))

        else:
            cc_requests = list(self._events().find({'created': {'$gte': start_date, '$lte': end_date}, 
                            'eventCategory': 'buy',
                            'eventAction': 'request',
                            'metadata.email': user
                        }))
            
            interac_requests = list(self._events().find({'created': {'$gte': start_date, '$lte': end_date}, 
                            'eventCategory': 'interac',
                            'eventAction': 'request',
                            'metadata.email': user
                        }))

        return cc_requests + interac_requests

    
    def get_deposit_request_set(self, request, events=None, clean=False):
        '''Creates a single deposit request set, either by querying the database, or by uses an optionally provided list of events.
        
        Keyword Arguments:
            request {bson} -- The request to build a request set around.
            events {list} -- The list of events corresponding to the events. (default: {None})
        
        Returns:
            json -- The request set created.
        '''

        # get the request time
        request_time = request['created']

        # get the time 1 hour before the request time
        lookback_time = request_time - datetime.timedelta(seconds=60*60) # look backwards in time an hour

        # get the user
        user_email = request['metadata']['email']

        if events == None:
            # get the request events
            events_list = list(self._events().find({'created': {'$gte': lookback_time, 
                                                                        '$lt': request_time}, 
                                                                        'metadata.email': user_email}))

        else:
            events_list = events

        if clean == False:
            request_set = {
                'request_type': request['eventCategory'],
                'user_email': user_email,
                'request': request,
                'events': events_list,
            }

        else:
            request_set = {
                'request_type': request['eventCategory'],
                'user_email': user_email,
                'request': event_processing.clean_event(request),
                'events': event_processing.clean_events(events_list),
            }  

        return request_set


    def get_clean_deposit_request_set(self, clean_request, clean_events_list):
        '''Creates a single deposit request set from a request and list of requests, corresponding to that request.
        
        Keyword Arguments:
            request {bson} -- The request to build a request set around.
            events {list} -- The list of events corresponding to the events. (default: {None})
        
        Returns:
            json -- The request set created.
        '''

        # get the user
        user_email = clean_request['user_email']


        # generate the request set
        request_set = {
            'request_type': clean_request['event_category'],
            'user_email': user_email,
            'request': clean_request,
            'events': clean_events_list,
        }

        return request_set


    def get_session_events(self, session, as_dataframe=False):
        '''This gets the complete list of events in the user's session. Key thing here is that some of the events don't have a session id. 
        So, we find the user events between the start and end time of the session as well as the events labelled with the session id.
        
        Arguments:
            session {string} -- The session identifier.
        
        Keyword Arguments:
            as_dataframe {bool} -- Whether or not the results should be returned as a dataframe or a list of events. (default: {False})
        
        Returns:
            list[json] or pandas.DataFrame -- The events that occurred in the user session.
        '''


        ec = self._events()
        
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
        
        result = sorted(session_events, key=lambda event: event['created']) 

        if as_dataframe == True:
            
            result = json_normalize(result)

        return result


    def get_summarized_request_sets(self, start_date=None, end_date=None, user=None, rsets=None):
        '''Gets the summarized request set information. This condenses all of the events in a request set into a single record, which can be used
        for machine learning, or analytics. This either reads the data from the database or works from a user supplied list of request sets.

        Keyword Arguments:
            start_date {datetime} -- The start date, if None, it defaults to seven days ago. (default: {None})
            end_date {datetime} -- The end date. If none, defaults to now. (default: {None})
            user {string} -- The email address of the user (default: {None})
            rsets {list} -- The list of request sets in clean format. (default: {None})
        
        Returns:
            list[json] -- The summarized request sets.
        '''

        
        if rsets == None:
            print("Reading Requests from the Database")
            rsets = self.get_deposit_request_sets(start_date=start_date, end_date=end_date, user=user)
        
        print("Flattening Request Sets")
        # separate out the events from the requests
        events = [{
            'request_id': event_processing.clean_event(rs['request'])['_id'], 
            'event': event_processing.clean_event(event)
        } for rs in rsets for event in rs['events']]
        
        # create a dataframe with the JSON events
        edf = json_normalize(events)
        edf['event._id'] = edf['event._id'].apply(lambda x: str(x)) # convert the bson to string
        edf['request_id'] = edf['request_id'].apply(lambda x: str(x)) # convert the bson to string
        
        events = edf.copy()
        events.columns = [col.replace('event.','') for col in events.columns]
        
        print("Generating summary features")
        events_summary = event_processing.generate_deposit_request_summary_df(events, 'request_id')
        
        return events_summary

    
    def get_summarized_clean_request_sets(self, rsets):
        '''Gets the summarized request set information. This condenses all of the events in a request set into a single record, which can be used
        for machine learning, or analytics. This either reads the data from the database or works from a user supplied list of request sets.

        Keyword Arguments:
            rsets {list} -- The list of request sets in clean format. (default: {None})
        
        Returns:
            list[json] -- The summarized request sets.
        '''

        # separate out the events from the requests
        events = [{
            'request_id': rs['request']['_id'], 
            'event': event
        } for rs in rsets for event in rs['events']]
        
        # create a dataframe with the JSON events
        edf = json_normalize(events)
        edf['event._id'] = edf['event._id'].apply(lambda x: str(x)) # convert the bson to string
        edf['request_id'] = edf['request_id'].apply(lambda x: str(x)) # convert the bson to string
        
        events = edf.copy()
        events.columns = [col.replace('event.','') for col in events.columns]
        
        events_summary = event_processing.generate_deposit_request_summary_df(events, 'request_id')
        
        return events_summary