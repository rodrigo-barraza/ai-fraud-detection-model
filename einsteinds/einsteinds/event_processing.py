"""event_processing.py

These methods all relate to cleaning and pre-processing the event data for data science needs.
This includes cleaning the events to make them consistent and extracting certain features.
It also includes created summaries that can be used for machine learning and analytics.
"""

import datetime
import pandas as pd 
from pandas.io.json import json_normalize
import numpy as np

from einsteinds import utils

# THIS ISN'T WORKING SO COMMENTED IT OUT
# import geopy
# from geopy.geocoders import Nominatim


# dictionary to map product codes to currency types
product_map = {
    1: 'BTC',
    2: 'ETH',
    3: 'LTC',
    4: 'XRP',
    5: 'DASH',
    6: 'BCC',
    7: 'USD',
    8: 'CAD',
    9: 'EOS',
    10: 'OMG',
    11: 'ZEC',
    12: 'IOT',
    13: 'GNT',
    14: 'TRX',
    15: 'ZRX',
    16: 'UBQ',
    17: 'FLASH'
}

# dictionary to convert countries
countries = {
    'canada': 'ca',
    'united states': 'us',
    'india': 'in'
}

# dictionary to convert provinces
prov_terr = {
    'alberta': 'ab',
    'british columbia': 'bc',
    'manitoba': 'mb',
    'new brunswick': 'nb',
    'newfoundland and labrador': 'nl',
    'northwest territories': 'nt',
    'nova scotia': 'ns',
    'nunavut': 'nu',
    'ontario': 'on',
    'prince edward island': 'pe',
    'quebec': 'qc',
    'saskatchewan': 'sk',
    'yukon': 'yt'
 }

# dictionary to convert states
states = {
    'alaska': 'ak',
    'alabama': 'al',
    'arkansas': 'ar',
    'american samoa': 'as',
    'arizona': 'az',
    'california': 'ca',
    'colorado': 'co',
    'connecticut': 'ct',
    'district of columbia': 'dc',
    'delaware': 'de',
    'florida': 'fl',
    'georgia': 'ga',
    'guam': 'gu',
    'hawaii': 'hi',
    'iowa': 'ia',
    'idaho': 'id',
    'illinois': 'il',
    'indiana': 'in',
    'kansas': 'ks',
    'kentucky': 'ky',
    'louisiana': 'la',
    'massachusetts': 'ma',
    'maryland': 'md',
    'maine': 'me',
    'michigan': 'mi',
    'minnesota': 'mn',
    'missouri': 'mo',
    'northern mariana islands': 'mp',
    'mississippi': 'ms',
    'montana': 'mt',
    'national': 'na',
    'north carolina': 'nc',
    'north dakota': 'nd',
    'nebraska': 'ne',
    'new hampshire': 'nh',
    'new jersey': 'nj',
    'new mexico': 'nm',
    'nevada': 'nv',
    'new york': 'ny',
    'ohio': 'oh',
    'oklahoma': 'ok',
    'oregon': 'or',
    'pennsylvania': 'pa',
    'puerto rico': 'pr',
    'rhode island': 'ri',
    'south carolina': 'sc',
    'south dakota': 'sd',
    'tennessee': 'tn',
    'texas': 'tx',
    'utah': 'ut',
    'virginia': 'va',
    'virgin islands': 'vi',
    'vermont': 'vt',
    'washington': 'wa',
    'wisconsin': 'wi',
    'west virginia': 'wv',
    'wyoming': 'wy'
}

# dictionary that defines how fields get processed for the creation of summaries 
# used in a recurrent neural network model
recurrent_mappings = {
    '_id': ['drop'],
    'created': ['time'],
    'session_id': ['cumunique'],
    'category_action_label': ['binary','cumunique'],
    'category_action': ['binary','cumunique'],
    'category_label': ['binary','cumunique'],
    'event_category': ['binary','cumunique'],
    'event_action': ['binary','cumunique'],
    'event_label': ['binary','cumunique'],
    'user_first_name': ['cumunique'],
    'user_last_name': ['cumunique'],
    'user_full_name': ['cumunique'],
    'user_email': ['cumunique'],
    'user_city': ['binary','cumunique'],
    'user_province_state_territory': ['binary','cumunique'],
    'user_country': ['binary','cumunique'],
    'user_postal_code_zip': ['cumunique'], 
    'user_street': ['cumunique'], 
    'user_ip': ['cumunique'],
    'transaction_type': ['binary','cumunique'],
    'billing_first_name': ['cumunique'],
    'billing_last_name': ['cumunique'],
    'billing_name': ['cumunique'],
    'billing_email': ['cumunique'],
    'billing_city': ['binary','cumunique'],
    'billing_province_state_territory': ['binary','cumunique'],
    'billing_country': ['binary','cumunique'],
    'billing_postal_code_zip': ['cumunique'],
    'billing_street': ['cumunique'],
    'card_last_digits': ['cumunique'],
    'card_expiry_month': ['cumunique'],
    'card_expiry_year': ['cumunique'],
    'card_type': ['cumunique'],
    'card_cvv': ['cumunique'],
    'cryptocurrency': ['binary','cumunique'],
    'fiat_currency': ['binary','cumunique'],
    'cryptocurrency_amount': ['value'],
    'fiat_currency_value': ['value'],
    'fiat_rate': ['value'],
    'crypto_wallet': ['cumunique'],
    'device_info': ['cumunique'],
    'password_length': ['cumunique'],
    'trade_side': ['binary','cumunique'],
    'trade_order_type': ['binary','cumunique'],
    'trade_result': ['binary','cumunique'],
    'trade_latest_price': ['value'],
    'trade_instrument': ['binary','cumunique']
}


def clean_event(event):
    '''Cleans a single event.
    
    Arguments:
        event {json} -- The event to be cleaned.
    
    Returns:
        json -- The cleaned event
    '''

    clean_event = {}

    event = json_normalize([event]).to_dict(orient='records')[0]

    for field in column_mappings.keys():

        clean_event = add_field(event=clean_event, key=field, value = column_mappings[field](event))


    return clean_event


def clean_event_minimal(event):
    '''An alternate event cleaning method that is a bit more efficient, since it only creates fields that relate to the raw record's source fields.
    
    Arguments:
        event {json} -- The event to be cleaned.
    
    Returns:
        json -- The cleaned event
    '''

    clean_event = {}

    event = json_normalize([event]).to_dict(orient='records')[0]

    cleaning_functions = []

    for field in event.keys():

        if col_to_function_map.get(field) != None:

            cleaning_functions += col_to_function_map[field]
        

    cleaning_functions = list(set(cleaning_functions))
    
    clean_event = {}

    for func in cleaning_functions:

        clean_event = add_field(event=clean_event, key=func_to_field_mapper[func.__name__], value = func(event))

    return clean_event


def clean_events(events):
    '''Cleans a set of events
    
    Arguments:
        events {list[json]} -- The list of json events to be cleaned.
    
    Returns:
        list[json] -- The cleaned list of events.
    '''

    return [clean_event(event) for event in events]


def clean_request_set(request_set):
    '''Cleans the request and all of the events in a request set.
    
    Arguments:
        request_set {json} -- The request set to be cleaned
    
    Returns:
        json -- The cleaned request set.
    '''
    
    # need to copy to avoid changing the original
    rset = request_set.copy()

    rset['request'] = clean_event(rset['request'])
    rset['events'] = clean_events(rset['events'])
    
    return rset


def clean_request_sets(request_sets):
    '''Cleans a list of request sets.
    
    Arguments:
        request_sets {list[json]} -- The list of request sets to be cleaned.
    
    Returns:
        list[json] -- The cleaned list of request sets.
    '''

    return [clean_request_set(request_set.copy()) for request_set in request_sets]


def cum_unique(array):
    '''Calculates the cumulative count of the unique values in a list.
    
    Arguments:
        array {list like object} -- The list to calculate the cumulative unique count on.

    Returns:
        list -- The cumulative counts of the unique items in a list.
    '''
    
    so_far = []
    unique = []
    
    for i in array:
        
        if i != None and pd.isnull([i])[0] == False:
            so_far.append(i)
            
        unique.append(len(list(set(so_far))))
        
    return unique


def seconds_since(series):
    '''Calculates the seconds between events in a pandas Series.
    
    Arguments:
        series {pandas.Series} -- The pandas Series which must be a datetime series.
    
    Returns:
        pandas.Series -- a series containing the time in seconds since the last event.
    '''
    
    diff = series - series.shift(1)
    
    diff = diff.apply(lambda delta: delta.total_seconds())
    
    diff = diff.replace(np.nan, 0)
    return diff


def cummean_seconds_since(series):
    '''Returns the cumulative mean seconds since the last event. 
    
    Arguments:
        series {pandas.Series} -- The pandas series contain the datetime of each event.
    
    Returns:
        pandas.Series -- The pandas series containing the cumulative mean seconds since the last event.
    '''
    
    return seconds_since(series).expanding().mean()


def create_recurrent_request_summary(rset):
    '''Creates a version of a request set that only contains numerical information, including cumulative summary features which 
    can be used in a recurrent neural network.
    
    Arguments:
        rset {json} -- The request set to be summarized.
    
    Returns:
        json -- The converted request set.
    '''

    request = rset['request']
    events = rset['events']
    events.append(request)

    df = pd.DataFrame(events)

    for col in df.columns:

        if col in recurrent_mappings.keys():

            this_col = df[col]

            transforms = recurrent_mappings[col]

            if 'binary' in transforms:
                df = pd.get_dummies(df, columns=[col])

            if 'cumunique' in transforms:

                df[col+'_cumunique'] = cum_unique(this_col)

            if 'time' in transforms:

                df[col+'_seconds_since_last'] = seconds_since(this_col)
                df[col+'_cummean_seconds_since'] = cummean_seconds_since(this_col)

            if 'value' not in transforms and 'binary' not in transforms:
                df.drop(col, axis=1, inplace=True)
    
    df['user_email'] = request['user_email']
    df['created'] = request['created']

    return df.to_dict('records')


def create_recurrent_request_summaries(rsets):
    '''Create a list of recurrent summarized records.
    
    Arguments:
        rsets {list[json]} -- The list of request sets. 
    
    Returns:
        list[json] -- The list of summarized/converted request sets.
    '''

    return [create_recurrent_request_summary(rset) for rset in rsets]


def add_field(event, key, value):
    '''Add a field to an event.
    
    Arguments:
        event {json} -- The event to add a field to.
        key {string} -- The name of the field to add.
        value {object} -- The value to place in the new field.
    
    Returns:
        json -- The 
    '''


    if value == None:
        return event
    else:
        event[key] = value

    return event


# CODE FOR LAT LONG - CURRENTLY NOT WORKING
# def add_user_lat_long(event):

#     street = get_if_exists(event, 'user_street')
#     city = get_if_exists(event, 'user_city')
#     prov_state = get_if_exists(event, 'user_province_state_territory')
#     country = get_if_exists(event, 'user_country')
#     postal_zip = get_if_exists(event, 'user_postal_code_zip')

#     lat_long = get_lat_long(street, city, prov_state, country, postal_zip)

#     if lat_long != None:

#         event['user_latitude'] = lat_long[0]
#         event['user_longitude'] = lat_long[1]

#     return event


# def add_billing_lat_long(event):

#     street = get_if_exists(event, 'billing_street')
#     city = get_if_exists(event, 'billing_city')
#     prov_state = get_if_exists(event, 'billing_province_state_territory')
#     country = get_if_exists(event, 'billing_country')
#     postal_zip = get_if_exists(event, 'billing_postal_code_zip')

#     lat_long = get_lat_long(street, city, prov_state, country, postal_zip)

#     if lat_long != None:

#         event['billing_latitude'] = lat_long[0]
#         event['billing_longitude'] = lat_long[1]

#     return event


# # note that this isn't working yet. Timing out for some reason
# def get_lat_long(street=None, city=None, prov_state=None, country=None, postal_zip=None):

#     address_string = ''

#     part_list = [street, city, prov_state, country, postal_zip]

#     for piece in part_list:
#         if piece != None:
#             if address_string == '':
#                 address_string = piece
#             else:
#                 address_string += ', '+piece

#     if address_string != '':    

#         geolocator = Nominatim()
#         location = geolocator.geocode(address_string)

#         if location != None:
#             lat = location.lattitude
#             lon = location.longitude

#             return (lat, lon)

#     return None


def strip_double_string(string):
    '''Checks if a string is two of the same string and if so returns the singular string.
    
    Arguments:
        string {string} -- the string to check
    
    Returns:
        string -- the singular string
    '''


    if string == None:
        return None

    length = len(string)//2

    if string[0:length] == string[length+1:]:
        return string[0:length]

    return string
    

def get_if_exists(event, key):
    '''Returns the value at a given key if the key exists in the dictionary.
    
    Arguments:
        event {dictionary} -- The dictionary containing the event data
        key {string} -- the string representing the key to look for in the event
    
    Returns:
        value -- returns the value at a given key
    '''


    return event.get(key)


def get_field_as_type(event, key, dtype):
    '''Attempts to return the value at a given key in the event as a specific type.
    
    Arguments:
        event {json} -- The event to get the field from
        key {string} -- The field name to retrieve the value from
        dtype {string} -- The string name of the desired datatype
    
    Returns:
        value of dtype or None -- Returns the value in the target type or None if the conversion is not possible.
    '''


    try:

        result = get_if_exists(event, key)

        if result != None:
            if dtype == 'string':

                result = str(result).lower()

                if result == '':
                    result = None

            elif dtype == 'int':

                result = int(result)

            elif dtype == 'datetime':

                if not isinstance(result, datetime.datetime):
                    result = None
            
            elif dtype == 'float':

                result = float(result)

    except:

        result = None

    return result


def get_mode_from_fields(event, dtype, fields):
    '''Extract the most common value from a set of fields that often contain the same value.
    This is a technique to combine multiple fields together that contain duplicate info.

    This might not be the best way to do this.
    
    Arguments:
        event {json} -- The event to get the field from
        dtype {string} -- The string name of the desired datatype
        fields {list[string]} -- The list of fields to search for the value in.
    
    Returns:
        object or None -- The most common value of type dtype found in the list of fields.
    '''


    # get the values of all the keys
    values = [get_field_as_type(event=event, dtype=dtype, key=field) for field in fields]

    # get rid of any none values
    values = list(set(values) - set([None, np.nan]))

    if len(values) == 0:
        return None

    return max(set(values), key=values.count) # calculates the mode and returns it


def get_created(event):
    """Extracts the created field from the MongoDB event
    
    Arguments:
        event {dict} -- json event description
    
    Returns:
        datetime or None -- the datetime the event was created
    """

    return get_field_as_type(event=event, key='created', dtype='datetime')


def get_event_category(event):
    '''Gets the event category from the event as a string.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The event category
    '''

    return get_field_as_type(event=event, key='eventCategory', dtype='string')


def get_event_action(event):
    '''Gets the event action from the event as a string.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The event action
    '''

    return get_field_as_type(event=event, key='eventAction', dtype='string')


def get_event_label(event):
    '''Gets the event label from the event as a string.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The event label
    '''

    if get_event_category(event) == 'session':
        return 'marker' # fix the issue of the session id being in the label

    label = get_field_as_type(event=event, key='eventLabel', dtype='string')

    if label != None and label.lower() == 'bitcoin':
        return 'btc'
        
    return label


def get_category_action_label(event):
    '''Gets the combined category action label which serves as a kind of unique event type identifier.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string -- The combined category_action_label
    '''

    return "_".join([str(get_event_category(event)),str(get_event_action(event)),str(get_event_label(event))])


def get_category_action(event):
    '''Gets the combined category action which serves as a kind of unique event type identifier.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string -- The combined category_action
    '''

    return "_".join([str(get_event_category(event)),str(get_event_action(event))])


def get_category_label(event):
    '''Gets the combined category label which serves as a kind of unique event type identifier.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string -- The combined category_label
    '''

    return "_".join([str(get_event_category(event)), str(get_event_label(event))])


def get_id(event):
    '''Gets the mongo id of the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        bson.ObjectId The mongo id of the event.
    '''

    return event['_id']


def get_user_first_name(event):
    '''Gets the first name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The first name of the user.
    '''

    return strip_double_string(get_field_as_type(event=event, dtype='string', key='metadata.firstName'))


def get_user_last_name(event):
    '''Gets the last name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The last name of the user.
    '''

    return strip_double_string(get_field_as_type(event=event, dtype='string', key='metadata.lastName'))


def get_user_full_name(event):
    '''Gets the full name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The full name of the user.
    '''

    return strip_double_string(get_mode_from_fields(event=event, dtype='string', fields=['metadata.name', 'metadata.fullName']))


def get_user_email(event):
    '''Gets the email of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The email of the user.
    '''

    return get_field_as_type(event=event, dtype='string', key='metadata.email')


def get_user_city(event):
    '''Gets the city of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The city of the user.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=['metadata.addressCity', 'metadata.city'])


def get_user_province_state_territory(event):
    '''Gets the province state or territory of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The province state or territory of the user.
    '''

    country = get_user_country(event)
    prov_state = get_mode_from_fields(event=event, dtype='string', fields=['metadata.addressProvince','metadata.province'])

    if country != None and prov_state != None:

        if country == 'us' and prov_state in states.keys():
            prov_state = states[prov_state]
        elif country == 'ca' and prov_state in prov_terr:
            prov_state = prov_terr[prov_state]

    return prov_state

def get_user_country(event):
    '''Gets the country of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The country of the user.
    '''

    country = get_mode_from_fields(event=event, dtype='string', fields=['metadata.addressCountry', 'metadata.country'])

    if country != None and country in countries.keys():
        country = countries[country]

    return country


def get_user_postal_zip(event):
    '''Gets the postal code or zip code of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The postal code or zip code of the user.
    '''

    postal =  get_mode_from_fields(event=event, dtype='string', fields=['metadata.addressPostal','metadata.postal'])

    if postal != None:
        postal = postal.replace(" ",'')

    return postal


def get_user_street(event):
    '''Gets the user's street from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The user's street.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=['metadata.addressStreet','metadata.street'])


def get_user_ip(event):
    '''Gets the user's ip address.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The user's ip address.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.ip',
        'metadata.requestIp',
        'metadata.prossessorResponse.customerIp',
        'metadata.prossessorResponse.request_ip'
        ])


def get_billing_first_name(event):
    '''Gets the first name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing first name of the user.
    '''

    return strip_double_string(get_field_as_type(event=event, dtype='string', key='metadata.prossessorResponse.profile.firstName'))


def get_billing_last_name(event):
    '''Gets the first name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing last name of the user.
    '''

    return strip_double_string(get_field_as_type(event=event, dtype='string', key='metadata.prossessorResponse.profile.lastName'))


def get_billing_name(event):
    '''Gets the first name of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing full name of the user.
    '''

    return strip_double_string(get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.cardName',
        'metadata.prossessorResponse.holderName',
        'metadata.cardHolder'
        ]))


def get_billing_email(event):
    '''Gets the billing email of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing full name of the user.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.email',
        'metadata.prossessorResponse.profile.email',
        'metadata.mongoResponse.email',
        'metadata.requestParams.email'
     ])


def get_billing_city(event):
    '''Gets the billing city of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing city of the user.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.billingDetails.city', 
        'metadata.processorResponse.billingDetails.city',
        'metadata.prossessorError.billingDetails.city'
     ])


def get_billing_province_state_territory(event):
    '''Gets the billing province state or territory of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing province state or territory of the user.
    '''

    country = get_billing_country(event)

    prov_state =  get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.billingDetails.state', 
        'metadata.processorResponse.billingDetails.state',
        'metadata.prossessorError.billingDetails.state'
    ])

    if country != None and prov_state != None:

        if country == 'us' and prov_state in states.keys():
            prov_state = states[prov_state]

        elif country == 'ca' and prov_state in prov_terr:
            prov_state = prov_terr[prov_state]

    return prov_state

    

def get_billing_country(event):
    '''Gets the billing country of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing country of the user.
    '''

    country = get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.billingDetails.country', 
        'metadata.processorResponse.billingDetails.country',
        'metadata.prossessorError.billingDetails.country'
    ])

    if country != None and country in countries.keys():
        country = countries[country]

    return country


def get_billing_postal_code_zip(event):
    '''Gets the billing postal code or zip code of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing postal code or zip code of the user.
    '''

    postal = get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.billingDetails.zip',
        'metadata.processorResponse.billingDetails.zip',
        'metadata.prossessorError.billingDetails.zip'
    ])
    
    if postal != None:
        postal = postal.replace(" ",'')

    return postal

def get_billing_street(event):
    '''Gets the billing street of the user from the event.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string or None -- The billing street of the user.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.billingDetails.street', 
        'metadata.processorResponse.billingDetails.street',
        'metadata.prossessorError.billingDetails.street'
    ])


def get_card_last_digits(event):
    '''Gets the last 4 digits of the user's credit card.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        int -- The last 4 digits of the user's credit card number as an int, or None.
    '''

    return get_mode_from_fields(event=event, dtype='int', fields=[
        'metadata.cardNumberLastFour', 
        'metadata.prossessorResponse.card.lastDigits',
        'metadata.prossessorResponse.lastDigits',
        'metadata.processorResponse.card.lastDigits'
        'metadata.cardSuffix',
        'metadata.prossessorError.card.lastDigits',
        'metadata.prossessorResponse.card_suffix'
    ])

def get_card_expiry_month(event):
    '''Gets the card expiry month of the user's credit card.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        int -- The two digit expiry month of the user's credit card number as an int, or None.
    '''

    return get_mode_from_fields(event=event, dtype='int', fields=[
        'metadata.prossessorResponse.card.cardExpiry.month', 
        'metadata.prossessorResponse.cardExpiry.month',
        'metadata.prossessorResponse.card_expiry_month',
        'metadata.prossessorResponse.card_expiry_year',
        'metadata.processorResponse.card.cardExpiry.month',
        'metadata.prossessorError.card.cardExpiry.month'
    ])

def get_card_expiry_year(event):
    '''Gets the card expiry yeat of the user's credit card.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        int -- The two digit expiry year[] of the user's credit card number as an int, or None.
    '''

    return get_mode_from_fields(event=event, dtype='int', fields=[
        'metadata.prossessorResponse.card.cardExpiry.year',
        'metadata.prossessorResponse.cardExpiry.year',
        'metadata.processorResponse.card.cardExpiry.year',
        'metadata.prossessorError.card.cardExpiry.year'
    ])

def get_card_type(event):
    '''Gets the type of credit card used by the user.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string -- The type of credit card used by the user, or None.
    '''

    return get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.prossessorResponse.card.type',
        'metadata.prossessorResponse.card.cardType',
        'metadata.prossessorResponse.cardType',
        'metadata.processorResponse.card.type',
        'metadata.prossessorError.card.cardType',
        'metadata.prossessorError.card.type',
        'metadata.prossessorResponse.card_type'
    ])

def get_card_cvv(event):
    '''Gets the cvv the user's credit card.
    
    Arguments:
        event {json} -- The event
    
    Returns:
        string -- The 3 digit cvv of the user's credit card number as a string, or None.
    '''

    return get_field_as_type(event=event, dtype='string', key=['metadata.prossessorResponse.cvv'])


def get_transaction_type(event):
    '''Gets the transaction type (either interac or credit card) from the event as a string.

    Arguments:
        event {event} -- The event

    Returns:
        string -- The transaction type of the event or None.
    '''

    event_category = get_event_category(event)
    
    if event_category == 'buy':

        return 'credit_card'
    
    if event_category == 'interac':

        return 'interac'

    return None


def get_fiat_currency_value(event):
    '''Gets the value in fiat currency of the event.
    
    Arguments:
        event {event} -- The event
    
    Returns:
        float -- The value in fiat currency.
    '''

    event_category = get_event_category(event)
    
    if event_category != 'trade':

        value = get_field_as_type(event=event, key='value', dtype='float')
        
        if value != None:
            return value/100.0
        
        else:
            value = get_field_as_type(event=event, key='metadata.cents', dtype='float')

            if value != None:
                return value/100.0
            else:
                return get_mode_from_fields(event=event, dtype='float', fields=[
                    'metadata.price',
                    'metadata.requestParams.price',
                    'metadata.mongoResponse.price', 
                    'metadata.processorResponse.chargeAmount',
                    'metadata.prossessorResponse.charge_amount',
                    'metadata.prossessorResponse.amount',
                    'metadata.requestParams.charge_amount'
                ])
    
    if event_category == 'trade':

            amount = get_cryptocurrency_amount(event)
            price = get_fiat_currency_rate(event)

            if amount != None and price != None and price != 0.0:
                return amount*price


    return None

def get_fiat_currency_rate(event):
    '''Gets the rate in fiat currency for one unit of cryptocurrency. This value is dependant on both the cryptocurrency and which fiat currency is selected.
    
    Arguments:
        event {event} -- The event
    
    Returns:
        float -- The fiat currency rate for one unit of cryptocurrency.
    '''

    event_category = get_event_category(event)
    
    if event_category == 'trade':

        price = get_field_as_type(event=event, key='metadata.price', dtype='float')
        
        if price == None or price == 0.0:
            price = get_field_as_type(event=event, key='metadata.lastTradedPx', dtype='float')
        
        if price != None:
            return price

    if event_category != 'trade':

        rate = get_field_as_type(event=event, key='metadata.rate', dtype='float')

        if rate != None:

            return rate
        
        else:

            amount = get_cryptocurrency_amount(event)
            value = get_fiat_currency_value(event)

            if amount != None and amount != 0.0 and value != None and value != 0.0:
                return value/amount

    return None

def get_cryptocurrency(event):
    '''Get the cryptocurrency of the transaction.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The cryptocurrency being used, purchased or traded in the event.
    '''

    event_category = get_event_category(event)
    event_label = get_event_label(event)

    if event_category == 'trade':

        result = get_field_as_type(event=event, key='metadata.instrument', dtype='string')

        if result != None:
            return result[0:-3] # strip off the fiat from the end eg BTCUSD
        
        return None

    if event_label not in  ['interac-confirm', None] and event_category in ['buy', 'enrollment-check', 'transaction', 'interac', 'withdraw']:

        return event_label

    product_id = get_field_as_type(event=event, key='metadata.productId', dtype='int')

    if product_id != None:
        return product_map[product_id].lower()

    return get_mode_from_fields(event=event, dtype='string', fields=[
                'metadata.product',
                'metadata.mongoResponse.product',
                'metadata.processorResponse.product'
                'metadata.requestParams.product'
        ])

def get_fiat_currency(event):
    '''The fiat currency being used in the event. Currently this would be either USD or CAD.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The 3 letter currency code of the fiat currency being used.
    '''

    currency = get_mode_from_fields(event=event, dtype='string', fields=[
        'metadata.processorResponse.currency', 
        'metadata.prossessorResponse.currencyCode',
        'metadata.processorResponse.currency',
        'metadata.against',
        'metadata.currency',
        'metadata.prossessorError.currency',
        'metadata.prossessorError.currencyCode',
        'metadata.requestParams.currency'
    ])

    event_label = get_event_label(event)
    event_category = get_event_category(event)

    if currency == None:

        if event_label == 'canadian dollar':
            return 'cad'
        if event_label == 'us dollar':
            return 'usd'
    
    if event_category == 'trade':

        return get_field_as_type(event=event, key='metadata.instrument', dtype='string')[-3:]
    
    return currency


def get_cryptocurrency_amount(event):
    '''The amount in cryptocurrency of the event.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        float -- The amount of cryptocurrency.
    '''


    amount = get_field_as_type(event=event, dtype='float', key='metadata.amount')

    if amount != None:
        return amount

    return get_mode_from_fields(event=event, dtype='float', fields=[
        'metadata.amount',
        'metadata.requestParams.amount',
        'metadata.mongoResponse.amount',
        'metadata.processorResponse.productAmount',
        'metadata.blockioResponse.data.amount_sent',
        'metadata.requestParams.product_amount'
    ])


def get_trade_order_type(event):
    '''The trade order type (MARKET, LIMIT, STOP)
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The trade order type or None.
    '''

    event_category = get_event_category(event)

    if event_category == 'trade':
        return get_field_as_type(event=event, key='metadata.type', dtype='string')

    return None

def get_trade_instrument(event):
    '''The trade instrument being used. For example. BTCUSD which is trading Bitcoin for US Dollars.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The trade instrument being used.
    '''

    event_category = get_event_category(event)

    if event_category == 'trade':
        return get_field_as_type(event=event, key='metadata.instrument', dtype='string')

    return None

def get_trade_side(event):
    '''The trade order side (BUY or SELL)
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The trade order side, or None.
    '''

    event_category = get_event_category(event)

    if event_category == 'trade':
        return get_field_as_type(event=event, key='metadata.side', dtype='string')

    return None

def get_trade_status(event):
    '''Gets the trade order status which is whether or not the trade was accepted by the Alpha Point system.
    This doesn't currently tell if the trade was fulfilled, only if it fits the parameters of a valid trade.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
         -- [description]
    '''

    event_category = get_event_category(event)

    if event_category == 'trade':
        return get_field_as_type(event=event, key='metadata.tradesResponse', dtype='string')

    return None


def get_session_id(event):
    '''Gets the session id of the event.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The session id of the event.
    '''

    return get_field_as_type(event=event, key='metadata.sessionId', dtype='string')

def get_crypto_wallet(event):
    '''Gets the crypto wallet of the event.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The crypto wallet of the event.
    '''

    return get_field_as_type(event=event, key='metadata.wallet', dtype='string')

def get_device_info(event):
    '''Gets the browser/device being used to generate the event.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        string -- The browser/device being used.
    '''

    return get_field_as_type(event=event, key='metadata.userAgent', dtype='string')

def get_password_length(event):
    '''Gets the length of the user's password.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        int -- The length of the user's password.
    '''

    return get_field_as_type(event=event, key='metadata.passwordLength', dtype='int')

def get_trade_latest_price(event):
    '''Gets the latest trade price for the given combination of cryptocurrency and fiat currency.
    
    Arguments:
        event {json} -- The event.
    
    Returns:
        float -- The latest trade price for the specific cryptocurrency/fiat currency combination.
    '''


    return get_field_as_type(event=event, key='metadata.lastTradedPx', dtype='float')


# mapping of the output fields for clean events, and the functions used to generate the values for those fields.
column_mappings = {
    '_id': get_id,
    'created': get_created,
    'session_id': get_session_id,
    'category_action_label': get_category_action_label,
    'category_action': get_category_action,
    'category_label': get_category_label,
    'event_category': get_event_category,
    'event_action': get_event_action,
    'event_label': get_event_label,
    'user_first_name': get_user_first_name,
    'user_last_name': get_user_last_name,
    'user_full_name': get_user_full_name,
    'user_email': get_user_email,
    'user_city': get_user_city,
    'user_province_state_territory': get_user_province_state_territory,
    'user_country': get_user_country,
    'user_postal_code_zip': get_user_postal_zip, 
    'user_street': get_user_street, 
    'user_ip': get_user_ip,
    'transaction_type': get_transaction_type,
    'billing_first_name': get_billing_first_name,
    'billing_last_name': get_billing_last_name,
    'billing_name': get_billing_name,
    'billing_email': get_billing_email,
    'billing_city': get_billing_city,
    'billing_province_state_territory': get_billing_province_state_territory,
    'billing_country': get_billing_country,
    'billing_postal_code_zip': get_billing_postal_code_zip,
    'billing_street': get_billing_street,
    'card_last_digits': get_card_last_digits,
    'card_expiry_month': get_card_expiry_month,
    'card_expiry_year': get_card_expiry_year,
    'card_type': get_card_type,
    'card_cvv': get_card_cvv,
    'cryptocurrency': get_cryptocurrency,
    'fiat_currency': get_fiat_currency, 
    'cryptocurrency_amount': get_cryptocurrency_amount,
    'fiat_currency_value': get_fiat_currency_value,
    'fiat_rate': get_fiat_currency_rate,
    'crypto_wallet': get_crypto_wallet,
    'device_info': get_device_info,
    'password_length': get_password_length,
    'trade_side': get_trade_side,
    'trade_order_type': get_trade_order_type,
    'trade_result': get_trade_status,
    'trade_latest_price': get_trade_latest_price,
    'trade_instrument': get_trade_instrument
}


def summarize_numerical_by_group(df, column, index, group):
    '''Summarizes a numerical pandas column to produce statistics about the column by group.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        column {string} -- The string name of the pandas column in the df.
        index {string} -- The record index column in the df.
        group {string} -- The string name of the column to use for grouping.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''

    
    if not all(col in df.columns for col in [index, column, group]):
        return pd.DataFrame()

    def size(gdf):

        return gdf.size
    
    summary = (df
               .groupby([index, group])[column]
               .aggregate(['mean', 'median', 'sum', 'max','min', np.nanstd, pd.Series.nunique, pd.Series.count, size])
               .reset_index())
    
    summary.columns = [col if col in [index,group] else column+"_"+col for col in summary.columns]
    
    summary = pd.melt(summary, id_vars=[index, group])
    
    
    summary[group] = summary[group].astype(str)+'_'+summary['variable'].astype(str)
    
    summary = summary.drop(['variable'], axis=1)
    
    summary = summary.pivot(index=index, columns=group, values='value')
    
    summary.columns = [group+"_"+col for col in summary.columns]
    
    return summary.reset_index().sort_values('request_id').set_index('request_id')


def summarize_numerical(df, column, index):
    '''Summarizes a numerical pandas column to produce statistics about the column.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        column {string} -- The string name of the pandas column in the df.
        index {string} -- The record index column in the df.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''

    if not all(col in df.columns for col in [index, column]):
        return pd.DataFrame()

    def size(gdf):

        return gdf.size
    
    summary = (df
               .groupby([index])[column]
               .aggregate(['mean', 'median', 'sum', 'max','min', np.nanstd, pd.Series.nunique, pd.Series.count, size])
               .reset_index())
    
    summary.columns = [col if col in [index] else column+"_"+col for col in summary.columns]
    
    return summary.sort_values('request_id').set_index('request_id')


def summarize_categorical(df, column, group):
    '''Summarizes a categorical pandas column to produce statistics about the column.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        column {string} -- The string name of the pandas column in the df.
        index {string} -- The record index column in the df.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''

    if not all(col in df.columns for col in [column, group]):
        return pd.DataFrame()

    def size(gdf):

        return gdf.size
    
    summary =  df.groupby(group)[column].aggregate([pd.Series.nunique, pd.Series.count, size]).reset_index()
    
    summary.columns = [col if col == group else column+"_"+col for col in summary.columns]
    
    return summary.sort_values('request_id').set_index('request_id')


def summarize_categorical_by_group(df, column, index, group):
    '''Summarizes a categorical pandas column to produce statistics about the column by group.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        column {string} -- The string name of the pandas column in the df.
        index {string} -- The record index column in the df.
        group {string} -- The string name of the column to use for grouping.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''
    
    if not all(col in df.columns for col in [index, column, group]):
        return pd.DataFrame()

    def size(gdf):

        return gdf.size
    
    summary = (df
               .groupby([index, group])[column]
               .aggregate([pd.Series.nunique, pd.Series.count, size])
               .reset_index())
    
    summary.columns = [col if col in [index,group] else column+"_"+col for col in summary.columns]
    
    summary = pd.melt(summary, id_vars=[index, group])
    
    summary[group] = summary[group].astype(str)+'_'+summary['variable'].astype(str)
    
    summary = summary.drop(['variable'], axis=1)
    
    summary = summary.pivot(index=index, columns=group, values='value')
    
    summary.columns = [group+"_"+col for col in summary.columns]
    
    return summary.reset_index().sort_values('request_id').set_index('request_id')


def summarize_timestamp(df, index, column):
    '''Summarizes a timestamp pandas column to produce statistics about the column.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        column {string} -- The string name of the pandas column in the df.
        index {string} -- The record index column in the df.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''
    if not all(col in df.columns for col in [index, column]):
        return pd.DataFrame()

    df = df.copy().sort_values([index,column])
    
    df['time_since_last'] = df['created'] - df.groupby(index)[column].shift(1)
    
    def size(gdf):

        return gdf.size
    
    def to_seconds(timed):
        
        return timed.total_seconds()
        
    df['time_since_last'] = df['time_since_last'].apply(to_seconds)
    
    summary = df.groupby(index)['time_since_last'].aggregate(['mean', 'median', 'sum', 'max','min', np.nanstd, pd.Series.count, size])
    
    summary.columns = [col if col == index else column+"_time_since_last_"+col for col in summary.columns]
    
    return summary.reset_index().sort_values('request_id').set_index('request_id')


def summarize_binary(df, index, columns):
    '''Summarizes a list of categorical columns as binary sums. 
    Converts each categorical column into a set of binary columns, one for each categorical value and,
    summarizes each of the binary columns.
    
    Arguments:
        df {pandas.DataFrame} -- The data frame containing the column, index and group.
        columns {list[string]} -- The list of string names of the pandas columns in the df.
        index {string} -- The record index column in the df.
    
    Returns:
        pandas.DataFrame -- The summary results in a dataframe with the index as the dataframe's index.
    '''

    columns = [col for col in columns if col in df.columns]

    if len(columns) == 0:
        return pd.DataFrame()

    df = df.copy()[[index]+columns]
    
    df = pd.get_dummies(df, prefix=columns, columns=columns)
    
    summary = df.groupby([index]).aggregate(['sum'])
    
    summary.columns = ["_".join(col) if col[0] != index else col for col in summary.columns.ravel()]
    
    return summary

def generate_deposit_request_summary_df(events, request_id):
    '''Creates a numerical summary of the set of events in the hour prior to the user
    making a credit card or interac deposit request.
    
    Arguments:
        events {list[json]} -- The list of events to summarize.
        request_id {string} -- The mongo id of the event where the request is made.
    
    Returns:
        pandas.DataFrame -- The summary of the deposit request events.
    '''

    # This doesn't yet use all the available clean columns and could be further improved to do so.

    summaries = [
        summarize_categorical(events, 'billing_city', request_id),
        summarize_categorical(events, 'billing_country', request_id),
        summarize_categorical(events, 'billing_email', request_id),
        summarize_categorical(events, 'billing_first_name', request_id),
        summarize_categorical(events, 'billing_last_name', request_id),
        summarize_categorical(events, 'billing_name', request_id),
        summarize_categorical(events, 'billing_postal_code_zip', request_id),
        summarize_categorical(events, 'billing_province_state_territory', request_id),
        summarize_categorical(events, 'billing_street', request_id),
        summarize_categorical(events, 'card_expiry_month', request_id),
        summarize_categorical(events, 'card_expiry_year', request_id),
        summarize_categorical(events, 'card_last_digits', request_id),
        summarize_categorical(events, 'card_type', request_id),
        summarize_categorical(events, 'category_action', request_id),
        summarize_categorical(events, 'category_action_label', request_id),
        summarize_categorical(events, 'category_label', request_id),
        summarize_categorical(events, 'crypto_wallet', request_id),
        summarize_categorical(events, 'cryptocurrency', request_id),
        summarize_categorical(events, 'device_info', request_id),
        summarize_categorical(events, 'event_action', request_id),
        summarize_categorical(events, 'event_category', request_id),
        summarize_categorical(events, 'event_label', request_id),
        summarize_categorical(events, 'fiat_currency', request_id),
        summarize_categorical(events, 'transaction_type', request_id),
        summarize_categorical(events, 'user_city', request_id),
        summarize_categorical(events, 'user_country', request_id),
        summarize_categorical(events, 'user_email', request_id),
        summarize_categorical(events, 'user_first_name', request_id),
        summarize_categorical(events, 'user_ip', request_id),
        summarize_categorical(events, 'user_last_name', request_id),
        summarize_categorical(events, 'user_postal_code_zip', request_id),
        summarize_categorical(events, 'user_province_state_territory', request_id),
        summarize_categorical(events, 'user_street', request_id),
        summarize_categorical(events, 'password_length', request_id),
        summarize_timestamp(events, request_id, 'created'),
        summarize_numerical_by_group(events, 'cryptocurrency_amount', request_id, 'cryptocurrency'),
        summarize_numerical_by_group(events, 'fiat_rate', request_id, 'cryptocurrency'),
        summarize_numerical_by_group(events, 'fiat_currency_value', request_id, 'fiat_currency'),
        summarize_numerical(events, 'fiat_currency_value', 'request_id'),
        summarize_numerical_by_group(events, 'fiat_currency_value', request_id, 'category_action'),
        summarize_binary(events, request_id, ['event_category', 'event_action', 'event_label', 'category_action_label','category_action','category_label'])
    ]

    summarydf = events.groupby(['user_email',request_id])['_id'].count().reset_index().drop('_id', axis=1).set_index(request_id)

    for summary in summaries:
        summarydf = summarydf.join(summary)
        
    return summarydf

'''A dictionary to map from the individual fields in the original events, to the functions needed to clean the events
to minimize processing required. Only the functions that need to run are run.
'''
col_to_function_map = {
    'metadata.firstName': [get_user_first_name],
    'metadata.lastName': [get_user_last_name],
    'metadata.name': [get_user_full_name], 
    'metadata.fullName': [get_user_full_name],
    'metadata.email': [get_user_email],
    'metadata.addressCity': [get_user_city],
    'metadata.city': [get_user_city],
    'metadata.addressProvince': [get_user_province_state_territory],
    'metadata.province': [get_user_province_state_territory],
    'metadata.addressCountry': [get_user_country], 
    'metadata.country': [get_user_country],
    'metadata.addressPostal': [get_user_postal_zip],
    'metadata.postal': [get_user_postal_zip],
    'metadata.addressStreet': [get_user_street],
    'metadata.street': [get_user_street],
    'metadata.ip': [get_user_ip],
    'metadata.requestIp': [get_user_ip],
    'metadata.prossessorResponse.customerIp': [get_user_ip],
    'metadata.prossessorResponse.request_ip': [get_user_ip],
    'metadata.prossessorResponse.profile.firstName': [get_billing_first_name],
    'metadata.prossessorResponse.profile.lastName': [get_billing_last_name],
    'metadata.cardName': [get_billing_name],
    'metadata.prossessorResponse.holderName': [get_billing_name],
    'metadata.cardHolder': [get_billing_name],
    'metadata.prossessorResponse.email': [get_billing_email],
    'metadata.prossessorResponse.profile.email': [get_billing_email],
    'metadata.mongoResponse.email': [get_billing_email],
    'metadata.requestParams.email': [get_billing_email],
    'metadata.prossessorResponse.billingDetails.city': [get_billing_city], 
    'metadata.processorResponse.billingDetails.city': [get_billing_city],
    'metadata.prossessorError.billingDetails.city': [get_billing_city],
    'metadata.prossessorResponse.billingDetails.state': [get_billing_province_state_territory], 
    'metadata.processorResponse.billingDetails.state': [get_billing_province_state_territory],
    'metadata.prossessorError.billingDetails.state': [get_billing_province_state_territory],
    'metadata.prossessorResponse.billingDetails.country': [get_billing_country], 
    'metadata.processorResponse.billingDetails.country': [get_billing_country],
    'metadata.prossessorError.billingDetails.country': [get_billing_country],
    'metadata.prossessorResponse.billingDetails.zip': [get_billing_postal_code_zip],
    'metadata.processorResponse.billingDetails.zip': [get_billing_postal_code_zip],
    'metadata.prossessorError.billingDetails.zip': [get_billing_postal_code_zip],
    'metadata.prossessorResponse.billingDetails.street': [get_billing_street], 
    'metadata.processorResponse.billingDetails.street': [get_billing_street],
    'metadata.prossessorError.billingDetails.street': [get_billing_street],
    'metadata.cardNumberLastFour': [get_card_last_digits], 
    'metadata.prossessorResponse.card.lastDigits': [get_card_last_digits],
    'metadata.prossessorResponse.lastDigits': [get_card_last_digits],
    'metadata.processorResponse.card.lastDigits': [get_card_last_digits],
    'metadata.cardSuffix': [get_card_last_digits],
    'metadata.prossessorError.card.lastDigits': [get_card_last_digits],
    'metadata.prossessorResponse.card_suffix': [get_card_last_digits],
    'metadata.prossessorResponse.card.cardExpiry.month': [get_card_expiry_month], 
    'metadata.prossessorResponse.cardExpiry.month': [get_card_expiry_month],
    'metadata.prossessorResponse.card_expiry_month': [get_card_expiry_month],
    'metadata.prossessorResponse.card_expiry_year': [get_card_expiry_month],
    'metadata.processorResponse.card.cardExpiry.month': [get_card_expiry_month],
    'metadata.prossessorError.card.cardExpiry.month': [get_card_expiry_month],
    'metadata.prossessorResponse.card.cardExpiry.year': [get_card_expiry_year],
    'metadata.prossessorResponse.cardExpiry.year': [get_card_expiry_year],
    'metadata.processorResponse.card.cardExpiry.year': [get_card_expiry_year],
    'metadata.prossessorError.card.cardExpiry.year': [get_card_expiry_year],
    'metadata.prossessorResponse.card.type': [get_card_type],
    'metadata.prossessorResponse.card.cardType': [get_card_type],
    'metadata.prossessorResponse.cardType': [get_card_type],
    'metadata.processorResponse.card.type': [get_card_type],
    'metadata.prossessorError.card.cardType': [get_card_type],
    'metadata.prossessorError.card.type': [get_card_type],
    'metadata.prossessorResponse.card_type': [get_card_type],
    'metadata.prossessorResponse.cvv': [get_card_cvv],
    'metadata.cents': [get_fiat_currency_value],
    'value': [get_fiat_currency_value, get_fiat_currency_rate],
    'metadata.price': [get_fiat_currency_value, get_fiat_currency_rate],
    'metadata.requestParams.price': [get_fiat_currency_value],
    'metadata.mongoResponse.price': [get_fiat_currency_value], 
    'metadata.processorResponse.chargeAmount': [get_fiat_currency_value],
    'metadata.prossessorResponse.charge_amount': [get_fiat_currency_value],
    'metadata.prossessorResponse.amount': [get_fiat_currency_value],
    'metadata.requestParams.charge_amount': [get_fiat_currency_value],
    'metadata.lastTradedPx': [get_fiat_currency_rate, get_trade_latest_price],
    'metadata.rate': [get_fiat_currency_rate],
    'metadata.instrument': [get_cryptocurrency, get_fiat_currency],
    'metadata.productId': [get_cryptocurrency],
    'metadata.product': [get_cryptocurrency],
    'metadata.mongoResponse.product': [get_cryptocurrency],
    'metadata.processorResponse.product': [get_cryptocurrency],
    'metadata.requestParams.product': [get_cryptocurrency],
    'metadata.processorResponse.currency': [get_fiat_currency], 
    'metadata.prossessorResponse.currencyCode': [get_fiat_currency],
    'metadata.processorResponse.currency': [get_fiat_currency],
    'metadata.against': [get_fiat_currency],
    'metadata.currency': [get_fiat_currency],
    'metadata.prossessorError.currency': [get_fiat_currency],
    'metadata.prossessorError.currencyCode': [get_fiat_currency],
    'metadata.requestParams.currency': [get_fiat_currency],
    'metadata.amount': [get_cryptocurrency_amount],
    'metadata.requestParams.amount': [get_cryptocurrency_amount],
    'metadata.mongoResponse.amount': [get_cryptocurrency_amount],
    'metadata.processorResponse.productAmount': [get_cryptocurrency_amount],
    'metadata.blockioResponse.data.amount_sent': [get_cryptocurrency_amount],
    'metadata.requestParams.product_amount': [get_cryptocurrency_amount],
    'metadata.type': [get_trade_order_type],
    'metadata.instrument': [get_trade_instrument],
    'metadata.side': [get_trade_side],
    'metadata.tradesResponse': [get_trade_status],
    'metadata.sessionId': [get_session_id],
    'metadata.wallet': [get_crypto_wallet],
    'metadata.userAgent': [get_device_info],
    'metadata.passwordLength': [get_password_length],
}

'''A dictionary to map the function names to the resulting field names in the cleaned events.
'''
func_to_field_mapper = {'get_id': '_id',
 'get_created': 'created',
 'get_session_id': 'session_id',
 'get_category_action_label': 'category_action_label',
 'get_category_action': 'category_action',
 'get_category_label': 'category_label',
 'get_event_category': 'event_category',
 'get_event_action': 'event_action',
 'get_event_label': 'event_label',
 'get_user_first_name': 'user_first_name',
 'get_user_last_name': 'user_last_name',
 'get_user_full_name': 'user_full_name',
 'get_user_email': 'user_email',
 'get_user_city': 'user_city',
 'get_user_province_state_territory': 'user_province_state_territory',
 'get_user_country': 'user_country',
 'get_user_postal_zip': 'user_postal_code_zip',
 'get_user_street': 'user_street',
 'get_user_ip': 'user_ip',
 'get_transaction_type': 'transaction_type',
 'get_billing_first_name': 'billing_first_name',
 'get_billing_last_name': 'billing_last_name',
 'get_billing_name': 'billing_name',
 'get_billing_email': 'billing_email',
 'get_billing_city': 'billing_city',
 'get_billing_province_state_territory': 'billing_province_state_territory',
 'get_billing_country': 'billing_country',
 'get_billing_postal_code_zip': 'billing_postal_code_zip',
 'get_billing_street': 'billing_street',
 'get_card_last_digits': 'card_last_digits',
 'get_card_expiry_month': 'card_expiry_month',
 'get_card_expiry_year': 'card_expiry_year',
 'get_card_type': 'card_type',
 'get_card_cvv': 'card_cvv',
 'get_cryptocurrency': 'cryptocurrency',
 'get_fiat_currency': 'fiat_currency',
 'get_cryptocurrency_amount': 'cryptocurrency_amount',
 'get_fiat_currency_value': 'fiat_currency_value',
 'get_fiat_currency_rate': 'fiat_rate',
 'get_crypto_wallet': 'crypto_wallet',
 'get_device_info': 'device_info',
 'get_password_length': 'password_length',
 'get_trade_side': 'trade_side',
 'get_trade_order_type': 'trade_order_type',
 'get_trade_status': 'trade_result',
 'get_trade_latest_price': 'trade_latest_price',
 'get_trade_instrument': 'trade_instrument'
 }