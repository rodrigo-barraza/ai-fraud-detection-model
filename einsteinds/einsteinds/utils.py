# data analysis stuff
from difflib import SequenceMatcher
from collections import Counter
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import datetime
import time
import pickle
import math
import json
import hashlib

# machine learning stuff
import sklearn
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)

from sklearn import cluster
from einsteinds import event_processing

# constant
MIN_SEARCH_DAYS = 7

###########

# columns to expand when processing the dataframe
expand_columns = ['eventAction', 
                  'eventCategory', 
                  'eventLabel',
                  'category_label_action',
                  'category_action',
                  'metadata.fraudulent']

##########


def getUserEventsDict(event_df):
    '''Takes in a pandas dataframe of events and creates a dictionary of events by user email'''

    user_dict = {}

    emails = event_df['metadata.email'].unique()

    for email in emails:

        user_dict[email] = event_df[event_df['metadata.email'] == email]

    return user_dict


def flatten_objects(objectList):
    '''Converts a list containing nested JSON objects to a flattened pandas dataframe'''

    return json_normalize(objectList)

def similarityRatio(a, b):
    '''
    Calculates a similarity score between two text strings.
    '''

    return SequenceMatcher(None, a, b).ratio()

def mode(array):
    '''
    Finds the statistical mode of an array aka the most common item.
    '''
    
    counts = Counter(array)
    return counts.most_common(1)[0][0]

def isInt(row):
    '''
    Function to map over a dataframe column to determin if an individual element in the column is an int.
    '''
    
    return row.is_integer()

def convertToNumeric(df):
    '''
    Attempt to convert all the columns in a dataframe to numeric.
    '''

    for col in df.columns:

        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
        
    return df

def getUserEvents(email, events_df):
    '''
    Get a pandas dataframe of all events for the user with that email.
    '''
    
    return events_df[events_df['metadata.email'] == email]


def totalUserEvents(user_df):
    '''Calculate the number of events for an individual user.'''
    
    return len(user_df.index) 


def totalUserRequests(user_df):
    '''
    Get the total number of buy requests for that user
    '''
    request_df = user_df[user_df['eventAction'] == 'request']
    
    return len(request_df.index)


def uniqueCards(user_df):
    '''
    Calculate the number of unique cards each user has
    '''
    
    card_digits = user_df[['metadata.cardNumberLastFour','metadata.prossessorResponse.lastDigits','metadata.prossessorResponse.card.lastDigits']]
    
    nums = []
    
    for col in card_digits.columns:
        
        df = card_digits[[col]].dropna()
        nums += list(df[col].values)
    
    return list(set(nums))    


def summarizeUserEvents(user_df, expand_cols=expand_columns):
    '''Summarizes the events for a given user.'''
    
    full_summary = {}
    
    for col in user_df.columns:
        
        series = user_df[col]
        
        summary = summarizeColumn(col, series, expand_cols)
        
        full_summary = {**full_summary, **summary}
        
    return full_summary


def summarizeAllUserEvents(user_dict):

    summaries = []

    for user in user_dict.keys(): # go through each email address
        
        summary = summarizeUserEvents(user_dict[user])
        summary['user_email'] = user
        
        summaries.append(summary)

    flat = flatten_objects(summaries)
    simple = simplifyAgg(flat)

    return simple
        
        
def summarizeColumn(name, series, expand_cols=expand_columns):
    '''Calculates summary statistics on a column of a data frame'''
    
    summary = {}
    
    # remove nas
    values = series.dropna().values
    
    excluded_substrings = ['id','created','address']
    
    # check that the column has values and that it's not a list type
    if (len(values) > 0) and (not isinstance(values[0],(list,))) and not any(sub in name.lower() for sub in excluded_substrings):
        
        nas = len(series.values) - len(values)
        summary[name+'_na'] = nas
        summary[name+'_notna'] = len(values)
        
        unique_values = series.dropna().unique()
        
        summary[name+'_unique'] = len(unique_values)
        
        # if the datatype is the obje
        if series.dtype == 'object':
            
            if name in expand_cols:
            
                for val in unique_values:

                    summary[name+'_n_'+str(val)] = np.sum(values == val)
                    #summary[name+'_mode'] = mode(values)

        elif series.dtype == 'float64':
            
            if name in expand_cols:
            
                for val in unique_values:

                    summary[name+'_n_'+str(val)] = np.sum(values == val)

            summary[name+'_mean'] = np.mean(values)
            summary[name+'_median'] = np.median(values.astype(int))
            #summary[name+'_mode'] = mode(values)
            summary[name+'_std'] = np.std(values)

            minval = np.min(values)
            maxval = np.max(values)

            summary[name+'_min'] = minval
            summary[name+'_max'] = maxval
            summary[name+'_range'] = maxval-minval

    return summary   

def simplifyAgg(df): 
    '''
    This whole chunk here is about managing missing data in the aggregates by removing some columns that don't need to be there
    and replacing NA values with 0s where it makes sense to allow for machine learning analysis.
    '''

    for col in df.columns:
        
        col_vals = df[col]
        
        nacount = len(col_vals) - len(col_vals.dropna())
        
        # if the column only relates to one user aka it's probably an id column or something like that
        if nacount == len(col_vals) - 1 or len(col_vals.dropna().unique()) == 1:
            df.drop(labels=[col], axis=1,inplace = True)
        
        else:
            # for all the columns that are counts of a certain value replace NaN with 0
            if '_n_' in col:
                df[col].fillna(0, inplace=True)

            # for all the columns that are modes
            if '_mode' in col:
                df[col].fillna('not_applicable', inplace=True)

            if '_unique' in col or '_notna' in col or '_na' in col:
                df[col].fillna(0, inplace=True)

            if '_mean' in col or '_median' in col or '_max' in col or '_min' in col or '_range' in col or '_std' in col:
                df[col].fillna(0, inplace=True)

    return df


def getNaCount(df):
    '''Calculates the number of NA values by column and returns a dataframe summarizing the info'''

    na_counts = []

    for col in df:
        
        vals = df[col]
        
        nas = len(vals) - len(vals.dropna())
        
        na_counts.append((col, nas))
        
    na_df = pd.DataFrame(data=na_counts, columns=['column','n_nas'])

    na_df.sort_values(ascending=False, by='n_nas', inplace=True)

    return na_df


def scaleValues(values):

    return sklearn.preprocessing.scale(X=values, with_mean=True, with_std=True)

def scaleDf(df):

    thisdf = df

    for col in thisdf.columns:
        thisdf[col] = scaleValues(thisdf[col].values)

    return thisdf


def calculatetSNEEmbeddings(df, pca=False):

    # get the values from the dataframe and scale them
    X = scaleDf(df).values

    if pca == True:
        pca = decomposition.PCA(n_components=3)
        X = pca.fit_transform(X)

    tsne = manifold.TSNE(n_components=2)
 
    X_tsne = tsne.fit_transform(X)

    return X_tsne

def calculatetKMeans(df, pca=True):

    # get the values from the dataframe and scale them
    X = scaleDf(df).values

    if pca == True:
        # create principal components 
        pca = decomposition.PCA(n_components=3)
        X = pca.fit_transform(X)

    kmn = cluster.KMeans()
    kmn.fit(X)
 
    cluster_labels = kmn.predict(X)

    return cluster_labels


######

# functions for processing and normalizing datafraem

def processEvents(df):
    '''Process the events dataframe to normalize columns'''
    
    subsetdf = None
    
    if df.shape[0] > 0:
        subsetdf = df
        subsetdf['category_label_action'] = subsetdf.eventCategory+'_'+subsetdf.eventLabel+'_'+subsetdf.eventAction
        subsetdf['category_action'] = subsetdf.eventCategory+'_'+subsetdf.eventAction
    
    return subsetdf


def combineEmails(row):
    
    if not pd.isnull(row['metadata.email']):
        
        return row['metadata.email']
    
    elif not pd.isnull(row['metadata.requestParams.email']):
        
        return row['metadata.requestParams.email']
        
    elif not pd.isnull(row['metadata.prossessorResponse.email']):
        
        return row['metadata.prossessorResponse.email']
    
    elif not pd.isnull(row['metadata.prossessorResponse.profile.email']):
        
        return row['metadata.prossessorResponse.profile.email']
    else:
        return np.nan
    
def combineNames(row):
    
    if not pd.isnull(row['metadata.firstName']):
        if not pd.isnull(row['metadata.lastName']):
            if row['metadata.firstName'] == row['metadata.lastName']:
                return row['metadata.firstName']
            else:
                return row['metadata.firstName'] + " " + row['metadata.lastName']
        else:
            return row['metadata.firstName']
                                                               
    if not pd.isnull(row['metadata.lastName']):
        if not pd.isnull(row['metadata.firstName']):
            if row['metadata.firstName'] == row['metadata.lastName']:
                return row['metadata.firstName']
            else:
                return row['metadata.firstName'] + " " + row['metadata.lastName']
        else:
            return row['metadata.lastName']
                                                               
    if not pd.isnull(row['metadata.prossessorResponse.profile.firstName']):
        if not pd.isnull(row['metadata.prossessorResponse.profile.lastName']):
            if row['metadata.prossessorResponse.profile.firstName'] == row['metadata.prossessorResponse.profile.lastName']:
                return row['metadata.prossessorResponse.profile.firstName']
            else:
                return row['metadata.prossessorResponse.profile.firstName'] + " " + row['metadata.prossessorResponse.profile.lastName']
        else:
            return row['metadata.prossessorResponse.profile.firstName']
    if not pd.isnull(row['metadata.prossessorResponse.profile.lastName']):
        if not pd.isnull(row['metadata.prossessorResponse.profile.firstName']):
            if row['metadata.prossessorResponse.profile.firstName'] == row['metadata.prossessorResponse.profile.lastName']:
                return row['metadata.prossessorResponse.profile.firstName']
            else:
                return row['metadata.prossessorResponse.profile.firstName'] + " " + row['metadata.prossessorResponse.profile.lastName']
        else:
            return row['metadata.prossessorResponse.profile.lastName']
    
    return np.nan
                                                                                          
    
def combineCardDigits(row):
    
    if not pd.isnull(row['metadata.cardNumberLastFour']):
        
        return row['metadata.cardNumberLastFour']
    
    elif not pd.isnull(row['metadata.prossessorResponse.lastDigits']):
        
        return row['metadata.prossessorResponse.lastDigits']
        
    elif not pd.isnull(row['metadata.prossessorResponse.card.lastDigits']):
        
        return row['metadata.prossessorResponse.card.lastDigits']

    else:
        return np.nan
    
    
def combineCardTypes(row):
    
    if not pd.isnull(row['metadata.prossessorResponse.card.cardType']):
        
        return row['metadata.prossessorResponse.card.cardType']
    
    elif not pd.isnull(row['metadata.prossessorResponse.card.type']):
        
        return row['metadata.prossessorResponse.card.type']
        
    elif not pd.isnull(row['metadata.prossessorResponse.cardType']):
        
        return row['metadata.prossessorResponse.cardType']

    else:
        return np.nan
    
def combineAmounts(row):
    
    if not pd.isnull(row['metadata.amount']):
        
        return row['metadata.amount']
    
    elif not pd.isnull(row['metadata.requestParams.amount']):
        
        return row['metadata.requestParams.amount']
        
    elif not pd.isnull(row['metadata.prossessorResponse.amount']):
        
        return row['metadata.prossessorResponse.amount']

    else:
        return np.nan
    
def combinePrices(row):
    
    if not pd.isnull(row['metadata.rate']):
        
        return row['metadata.rate']
    
    elif not pd.isnull(row['metadata.requestParams.price']):
        
        return row['metadata.requestParams.price']
        
    else:
        return np.nan
    
def combineCardholderName(row):
    
    if not pd.isnull(row['metadata.cardName']):
        
        return row['metadata.cardName']
    
    elif not pd.isnull(row['metadata.prossessorResponse.holderName']):
        
        return row['metadata.prossessorResponse.holderName']
        
    else:
        return np.nan
    
def compareNameCardName(row):
    
    if not pd.isnull(row['name']) and not pd.isnull(row['cardholdername']):
    
        return similarityRatio(row['name'],row['cardholdername'])
    
    return np.nan

def catColumnMeanSimilarity(series):
    '''Calculates the within column similarity for a pandas series of text values'''
    
    unique = series[series.notnull()].unique()
    
    return ([[similarityRatio(x,y) for x in unique] for y in unique])

def cleanUpEvents(df):
    '''Takes in the base flattened dataframe and creates columns which summarize data from disperate source columns'''

    subsetdf = df
    subsetdf['email'] = subsetdf.apply(combineEmails, axis=1)
    subsetdf['name'] = subsetdf.apply(combineNames, axis=1)
    subsetdf['cardnumbers'] = subsetdf.apply(combineCardDigits, axis=1)
    subsetdf['cardtypes'] = subsetdf.apply(combineCardTypes, axis=1)
    subsetdf['amounts'] = subsetdf.apply(combineAmounts, axis=1)
    subsetdf['prices'] = subsetdf.apply(combinePrices, axis=1)
    subsetdf['cardholdername'] = subsetdf.apply(combineCardholderName, axis=1)
    subsetdf['namesimilarity'] = subsetdf.apply(compareNameCardName, axis=1)
    subsetdf['category_label_action'] = subsetdf.eventCategory+'_'+subsetdf.eventLabel+'_'+subsetdf.eventAction
    subsetdf['category_action'] = subsetdf.eventCategory+'_'+subsetdf.eventAction

    return subsetdf

def createUserEventSummaries(clean_df, expand_cols=expand_columns):
    '''
    Takes in a clean dataframe and creates event summaries for each user
    Returns a consolidated dataframe and
    '''
    
    # summarize the whole dataframe first
    # need to do this to get a list of all possible columns before summarizing each user
    wholedf_summary = summarizeUserEvents(clean_df)

    columns = list(wholedf_summary.keys())

    data_dict = {}
    data_dict['email'] = []

    # initialize the data dictionary
    for col in columns:

        data_dict[col] = []

    emails = clean_df.email.unique()

    # variables to store summaries and user dictionaries
    summaries = []
    user_dict = {}

    # for each user
    for email in emails:
        
        # get the user's events
        user_df = clean_df[clean_df.email == email]
        
        # get the user summary
        summary = summarizeUserEvents(user_df=user_df, expand_cols=expand_cols)
        
        summaries.append(summary)
        
        # create the user dict
        user_dict[email] = {'summary': summary, 'dataframe': user_df}

        data_dict['email'].append(email)

        summary_fields = summary.keys()

        # populate the data dict with values from the user summary
        for key in data_dict.keys():
            if key != 'email':
                if key in summary_fields:
                    data_dict[key].append(summary[key])
                else:
                    data_dict[key].append(0)
                    
    summary_df = pd.DataFrame(data_dict)
    summary_df = simplifyAgg(summary_df)

    return {'summary_df': summary_df, 'summary_dict': user_dict, 'all_users_summary': wholedf_summary}


def distance(user_df, all_df, top=None):
    '''Calculates the distance in standard deviations from mean of other users.
    If top is specified it returns the top metrics in ascending order.
    '''

    col = []
    val = []

    print(user_df)
    
    for metric in all_df.columns:

        if metric != 'email':
            print(metric)
            mean = np.mean(all_df[metric].values)
            std = np.std(all_df[metric].values)
            distance = abs(user_df[metric].values[0]-mean)
            distance_std = distance/std

            col.append(metric)
            val.append(distance_std)
    
    result = pd.DataFrame({'metric': col, 'distance': val})
    result = result.dropna()

    if top == None:
        return result.sort_values(by='distance')
    else:
        return result.sort_values(by='distance').tail(top)


def allUserAvgDistance(all_df):
    '''Calculates the distance in standard deviations from mean of other users.
    If top is specified it returns the top metrics in ascending order.
    '''

    lookup = {}
    
    for metric in all_df.columns:

        if metric != 'email':
            
            mean = np.mean(all_df[metric].values)
            std = np.std(all_df[metric].values)

            lookup[metric] = {'mean': mean, 'std': std}

    users = []
    avg_distances = []

    for user in all_df.email.unique():
        if pd.isnull(user) == False:

            users.append(user)
            udf = all_df[all_df.email == user]
            distances = []

            for metric in all_df.columns:

                if metric != 'email':

                    udf = all_df[all_df.email == user]

                    distance = abs(udf[metric].values[0]-lookup[metric]['mean'])
                    distances.append(distance/lookup[metric]['std'])

            avg_distances.append(np.nanmean(distances))

    return pd.DataFrame({'email': users, 'avg_distance_z': avg_distances})      


def savePickle(event_dict, name):

    with open(name, 'wb') as handle:
        pickle.dump(event_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def openPickle(name):

    data = {}

    with open(name, 'rb') as handle:
    
        data = pickle.load(handle)

    return data

#####


def prefixColumns(prefixlist, columnlist, allin=False):
    '''
    Allows the user to select pandas columns by a list of prefixes.
    If allin=False, columns with any of the prefixes will be selected. If allin=True, only columns containing all the prefixes are selected.
    '''
    sublist = []
    
    for col in columnlist:
        if allin == True:
            if all(prefix in col.lower() for prefix in prefixlist):
                sublist.append(col)
        else:
            if any(prefix in col.lower() for prefix in prefixlist):
                sublist.append(col)
            
    return sublist


def combineColumnsByPrefix(prefixlist, df, allin=True):
    '''Given a list of prefixes combines the values in multiple columns by taking the max value'''

    return np.nanmax(df[prefixColumns(prefixlist=prefixlist, columnlist=df.columns, allin=allin)].fillna('').astype(str).values, axis=1)


def summarizeRecordsSmall(df):
    '''Creates a smaller subset of the data summary that is focused on the most important columns.'''
    
    numeric_columns = ['currency_price']
    general_columns = ['card_name', 'card_type', 'card_last_digits','first_name','last_name','country','state','city','currency','card_expiry_month','card_expiry_year']
    exclude_columns = ['created','email','currency_amount']
    
    users = df.email.unique()
    
    summary_dict = {}
    summary_dict['email'] = []
    
    # initialize dictionary
    for column in df.columns:
        if column not in exclude_columns:
            summary_dict['n_'+column] = []

            if column in general_columns:
                summary_dict['unique_'+column] = []
                summary_dict['na_'+column] = []

            if column in numeric_columns:
                summary_dict['na_'+column] = []
                summary_dict['mean_'+column] = []
                summary_dict['median_'+column] = []
                summary_dict['std_'+column] = []
    
    # for each user in the database with events
    for user in users:
        
        userdf = df[df.email == user]
        summary_dict['email'].append(user) 
        
        for column in df.columns:
            if column not in exclude_columns:

                if column in general_columns:
                    summary_dict['n_'+column].append(np.nansum(userdf[column].isnull() == False))
                    summary_dict['unique_'+column].append(len(userdf[column].unique()))
                    summary_dict['na_'+column].append(np.sum(userdf[column].isnull() == True))

                elif column in numeric_columns:
                    col = pd.to_numeric(userdf[column])
                    summary_dict['n_'+column].append(np.nansum(col.isnull() == False))
                    summary_dict['na_'+column].append(np.sum(col.isnull() == True))
                    summary_dict['mean_'+column].append(np.nanmean(col))
                    summary_dict['median_'+column].append(np.nanmedian(col))
                    summary_dict['std_'+column].append(np.nanstd(col))
                
                else:
                    summary_dict['n_'+column].append(np.nansum(userdf[column]))
                    
    # convert to a dataframe and return                
    return pd.DataFrame(summary_dict)


def similarityMax(email, email_list, threshold=0.9):
    '''Returns True if the email is similar to an email in the list and false otherwise'''
    
    similarity = np.array([similarityRatio(email, fe) for fe in email_list]) 
    
    max_index = np.argmax(similarity)
    
    if similarity[max_index] >= 0.9 and similarity[max_index] < 1:
        return True
    else:
        return False


def cleanEventsDF(edf_raw, expand=True):
    '''Cleans the raw events combining columns together'''

    cols = prefixColumns(['created','email','name','card','price','value','amount','cents','.rate','event','fraud','.ip','product','country','city','state','interac'], edf_raw.columns)

    subset = edf_raw[cols]
    subset = subset.dropna(how='all', subset=prefixColumns(['email'],edf_raw.columns))
    subset['eventLabel'][subset.eventLabel == 'bitcoin'] = 'BTC'
    subset['cl'] = subset.eventCategory +'_'+subset.eventLabel
    subset['ca'] = subset.eventCategory +'_'+subset.eventAction
    subset['cla'] = subset.eventCategory+'_'+subset.eventLabel+'_'+subset.eventAction
    subset['email'] = combineColumnsByPrefix(['email'],subset)
    subset['card_name'] = np.nanmax(subset[['metadata.prossessorResponse.holderName','metadata.cardName']].fillna('').astype(str).values, axis=1)
    subset['card_type'] = combineColumnsByPrefix(['card','type'],subset)
    subset['card_last_digits'] = combineColumnsByPrefix(['card','last'],subset)
    subset['first_name'] = combineColumnsByPrefix(['first','name'],subset)
    subset['last_name'] = combineColumnsByPrefix(['last','name'],subset)
    subset['country'] = combineColumnsByPrefix(['country'],subset)
    subset['state'] = combineColumnsByPrefix(['state'],subset)
    subset['city'] = combineColumnsByPrefix(['city'],subset)
    subset['card_expiry_month'] = combineColumnsByPrefix(['card','expiry','month'],subset)
    subset['card_expiry_year'] = combineColumnsByPrefix(['card','expiry','year'],subset)
    subset['currency'] = combineColumnsByPrefix(['product'],subset)
    subset['currency_price'] = combineColumnsByPrefix(['.price'],subset)
    subset['currency_amount'] = np.nanmax(subset[['metadata.amount','metadata.mongoResponse.amount','metadata.requestParams.amount']].fillna('').astype(str).values, axis=1)
    subset.replace(to_replace='',value=np.nan, inplace=True)

    summary_columns = ['created','email','eventCategory','eventLabel','eventAction','cl','cla','ca','card_name','card_type','card_last_digits','first_name','last_name','country','state','city','card_expiry_month','card_expiry_year','currency','currency_price','currency_amount','metadata.fraudulent']
    expand_columns = ['eventCategory','eventLabel','eventAction','cl','ca','cla','metadata.fraudulent']

    base_summary = subset[summary_columns]

    if expand == True:
        base_summary = pd.get_dummies(columns=expand_columns, data=base_summary, dummy_na=True)

    return base_summary


def user_node_json(email, df):
    
    df = df[df.email == email].drop('email', axis=1)
    
    dict_rec = df.to_dict(orient='records')
    
    return dict_rec
    
def user_link_json(email, df):
    
    df = df[df.email == email].drop('email', axis=1)
    
    dict_rec = df.to_dict(orient='records')
    
    return dict_rec

def transition_summary(df):
    
    # get rid of login auth as a destination because it doesn't make sense. Login-auth should be the entry point to a session
    # df = df[(df['destination'].str.contains('login') & (df['destination'].str.contains('auth'))) == False]
    
    gb = df.groupby(['email','origin','destination'], as_index=False).count()
    gb['user_count'] = gb['created']
    gb = gb[['email','origin','destination','user_count']]
    gb = gb.dropna()

    total_user_activity_counts = pd.DataFrame(gb.groupby(['email'])['user_count'].sum())
    total_user_activity_counts = total_user_activity_counts.reset_index()
    total_user_activity_counts['total_user_count'] = total_user_activity_counts['user_count']
    total_user_activity_counts = total_user_activity_counts[['email','total_user_count']]
    total_user_activity_counts = total_user_activity_counts.dropna()

    transition_data = gb.set_index('email').join(total_user_activity_counts.set_index('email'))

    gb = df.groupby(['origin','destination'], as_index=False).count()
    gb['alluser_transition_count'] = gb['created']
    gb = gb[['origin','destination','alluser_transition_count']]
    gb = gb.dropna()
    gb.set_index(['origin','destination'])

    transition_summary = transition_data.reset_index().set_index(['origin','destination']).join(gb.set_index(['origin','destination'])).reset_index()
    transition_summary['all_user_count'] = df.shape[0]
    transition_summary['user_proportion'] = transition_summary['user_count']/transition_summary['total_user_count']
    transition_summary['alluser_proportion'] = transition_summary['alluser_transition_count']/transition_summary['all_user_count']

    transition_summary['relative_proportion'] = transition_summary['user_proportion']/transition_summary['alluser_proportion']

    return transition_summary

def state_summary(df):
    
    all_users = pd.DataFrame(df.groupby(['destination'])['created'].count()).reset_index()
    all_users['alluser_state_count'] = all_users['created']
    all_users = all_users.drop('created', axis=1)
    all_users['alluser_state_proportion'] = all_users['alluser_state_count']/np.sum(all_users['alluser_state_count'])
        
    users = pd.DataFrame(df.groupby(['email','destination'])['created'].count()).reset_index()
    users['user_state_count'] = users['created']
    users = users.drop('created', axis=1)
    
    ugb  = pd.DataFrame(users.groupby(['email'])['user_state_count'].sum()).reset_index()
    ugb['user_allstate_count'] = ugb['user_state_count']
    ugb = ugb.drop('user_state_count', axis=1)
    
    summary = users.set_index('email').join(ugb.set_index('email'))
    
    summary = summary.reset_index().set_index('destination').join(all_users.set_index('destination')).reset_index()
    
    summary['user_state_proportion'] = summary['user_state_count']/summary['user_allstate_count']
    summary['relative_proportion'] = summary['user_state_proportion']/summary['alluser_state_proportion']
    summary['state'] = summary['destination']
    summary = summary.drop('destination', axis=1)
    
    return summary

def get_search_dates(start_date, end_date):
    '''Converts the PST datetimes into GMT datetime, which is what is stored in MongoDB. This should be improved to
    be timezone aware.
    
    Arguments:
        start_date {datetime.datetime or string} -- The start date
        end_date {datetime.datetime or string} -- The end date

    Returns:
        tuple(datetime.datetime, datetime.datetime) -- The datetimes converted to GMT
    '''

    
    if end_date == None:

        end_date = datetime.datetime.now()

    if start_date == None:

        start_date = end_date - datetime.timedelta(days=MIN_SEARCH_DAYS)

    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date[0:10], '%Y-%m-%d')
        
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date[0:10], '%Y-%m-%d')
    
    # adjust search query because Mongo is in GMT
    start_date = start_date + datetime.timedelta(hours=7)
    end_date = end_date + datetime.timedelta(hours=7)
    
    return (start_date, end_date)


def summarize_request_set(request_set):
    
    if len(request_set['events']) > 0:

        # convert the events to a pandas dataframe
        events = json_normalize(event_processing.clean_events(request_set['events']))
    
        # count events by category
        category_action_label_groups = events.groupby('category_action_label')['_id'].count().reset_index().rename(columns={'category_action_label': 'field', '_id': 'value'})
        category_action_groups = events.groupby('category_action')['_id'].count().reset_index().rename(columns={'category_action': 'field', '_id': 'value'})
        category_label_groups = events.groupby('category_label')['_id'].count().reset_index().rename(columns={'category_label': 'field', '_id': 'value'})
        category_action_label_groups['field'] = category_action_label_groups['field'].apply(lambda x: 'n_'+x)
        category_action_groups['field'] = category_action_groups['field'].apply(lambda x: 'n_'+x)
        category_label_groups['field'] = category_label_groups['field'].apply(lambda x: 'n_'+x)
        
        has_amount = None
        has_value = None

        result = pd.concat([category_action_label_groups, category_action_groups, category_label_groups]).reset_index(drop=True)

        if 'cryptocurrency_amount' in events.columns:
            # convert amount and value columns to numeric
            events['cryptocurrency_amount'] = pd.to_numeric(events['cryptocurrency_amount'])

             # get only the not null values
            has_amount = events[events['cryptocurrency_amount'].isnull() == False]

            # category_action_labelculate statistics on amount data
            count_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].count().reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            mean_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].mean().reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            min_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].min().reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            max_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].max().reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            median_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].median().reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            std_amount = has_amount.groupby('category_action_label')['cryptocurrency_amount'].std(ddof=0).reset_index().rename(columns={'category_action_label': 'field', 'cryptocurrency_amount': 'value'})
            
            count_amount['field'] = count_amount['field'].apply(lambda x: 'n_amount_'+x)
            mean_amount['field'] = mean_amount['field'].apply(lambda x: 'mean_amount_'+x)
            min_amount['field'] = min_amount['field'].apply(lambda x: 'min_amount_'+x)
            max_amount['field'] = max_amount['field'].apply(lambda x: 'max_amount_'+x)
            median_amount['field'] = median_amount['field'].apply(lambda x: 'median_amount_'+x)
            std_amount['field'] = std_amount['field'].apply(lambda x: 'std_amount_'+x)

            result = pd.concat([result,
                count_amount,mean_amount, min_amount, max_amount, median_amount, std_amount]).reset_index(drop=True)


        if 'fiat_currency_value' in events.columns:
            events['fiat_currency_value'] = pd.to_numeric(events['fiat_currency_value'])
               
            has_value = events[events['fiat_currency_value'].isnull() == False]
        
            # category_action_labelculate statistics on value data
            count_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].count().reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            mean_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].mean().reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            min_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].min().reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            max_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].max().reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            median_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].median().reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            std_category_action_label_value = has_value.groupby('category_action_label')['fiat_currency_value'].std(ddof=0).reset_index().rename(columns={'category_action_label': 'field', 'fiat_currency_value': 'value'})
            
            count_category_action_label_value['field'] = count_category_action_label_value['field'].apply(lambda x: 'n_value_'+x)
            mean_category_action_label_value['field'] = mean_category_action_label_value['field'].apply(lambda x: 'mean_value_'+x)
            min_category_action_label_value['field'] = min_category_action_label_value['field'].apply(lambda x: 'min_value_'+x)
            max_category_action_label_value['field'] = max_category_action_label_value['field'].apply(lambda x: 'max_value_'+x)
            median_category_action_label_value['field'] = median_category_action_label_value['field'].apply(lambda x: 'median_value_'+x)
            std_category_action_label_value['field'] = std_category_action_label_value['field'].apply(lambda x: 'std_value_'+x)
            
            # category_action_labelculate statistics on value data
            count_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].count().reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            mean_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].mean().reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            min_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].min().reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            max_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].max().reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            median_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].median().reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            std_category_action_value = has_value.groupby('category_action')['fiat_currency_value'].std(ddof=0).reset_index().rename(columns={'category_action': 'field'}).rename(columns={'category_action': 'field', 'fiat_currency_value': 'value'})
            
            count_category_action_value['field'] = count_category_action_value['field'].apply(lambda x: 'n_value_'+x)
            mean_category_action_value['field'] = mean_category_action_value['field'].apply(lambda x: 'mean_value_'+x)
            min_category_action_value['field'] = min_category_action_value['field'].apply(lambda x: 'min_value_'+x)
            max_category_action_value['field'] = max_category_action_value['field'].apply(lambda x: 'max_value_'+x)
            median_category_action_value['field'] = median_category_action_value['field'].apply(lambda x: 'median_value_'+x)
            std_category_action_value['field'] = std_category_action_value['field'].apply(lambda x: 'std_value_'+x)
        
            result = pd.concat([result, count_category_action_label_value, mean_category_action_label_value, min_category_action_label_value, max_category_action_label_value, median_category_action_label_value, std_category_action_label_value,
                    count_category_action_value, mean_category_action_value, min_category_action_value, max_category_action_value, median_category_action_value, std_category_action_value]).reset_index(drop=True)
        
        result['idx'] = 'dummy'
        
        result = result.pivot(index='idx',columns='field',values='value')
        result.columns = [col.replace('.','_') for col in result.columns]
        result = result.to_dict(orient='records')[0]

        if 'card_last_digits' in list(events.columns):
            result['n_unique_cards'] = events['card_last_digits'].nunique()

        result['n_events'] = len(events)

        # convert np.int64 as it doesn't serialize properly to json
        for col in result.keys():
            if isinstance(result[col], np.int64):
                result[col] = int(np.asscalar(result[col]))
    
    else:
        result = {}

          
    # add in the request features
    request = event_processing.clean_event(request_set['request'])
    result['request'] = request
    
    return result

def anonymize_email(email):
    
    if len(email) > 3 and '@' in email:
    
        split = email.split('@')
        front = split[0]
        back = split[1]
        n = len(front)

        hash_object = hashlib.sha256(str.encode(front))
        hex_dig = hash_object.hexdigest()

        return hex_dig[0:n]+'@'+back
    
    else:
        return None