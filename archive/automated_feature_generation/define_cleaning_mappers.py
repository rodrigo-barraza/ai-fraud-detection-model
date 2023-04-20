from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import FunctionTransformer

# Get the column groups by cleaning function.
from define_column_cleaning_types import (
    bitcoin_cols,
    country_cols,
    airport_cols,
    first_n_char_cols_dict,
    url_cols,
    card_type_cols,
    card_year_cols,
    datetime_cols,
    province_cols,
    string_list_cols,
    string_to_lc_cols,
    float_cols,
    id_cols,
    no_cleaning_cols
)

#--------------------------------------------------------------
# 'bitcoin' to 'BTC' text replacement
#--------------------------------------------------------------
def bitcoin_to_BTC( df ):
    return df.str.replace('^(bitcoin)$', 'BTC', case=False) # Replace column entries of 'bitcoin', disregarding case, with 'BTC'.

BitcoinToBTC = FunctionTransformer( func=bitcoin_to_BTC, validate=False)

mapper_bitcoin_clean = DataFrameMapper(
    [(c, BitcoinToBTC) for c in bitcoin_cols],
    input_df=True, df_out=True
)
#--------------------------------------------------------------
# Country to Code
#--------------------------------------------------------------
import pycountry_convert

def to_country_code( df ): # Helper function to do the actual transformation.

    # df: Can contain full country names as per 'pycountry' OR the two letter codes, mixed together.
    # The .str.lower() maps the full names to lower case for the dictionary look up. 
    # Existing two letter codes are lc'd as well but will be ignored by the dictionary replace.
    # After the lookup return everything to upper case.

    # Generate the dictionary that maps lower cased full country names to their upper case two letter codes.
    country_to_code_all_codes = pycountry_convert.map_countries(cn_name_format='lower')   
    country_to_code_dict = {}

    for k in country_to_code_all_codes:
        country_to_code_dict[k] = country_to_code_all_codes[k]['alpha_2']

    # Now do the mapping.     
    if df.isnull().values.all() == True: # This currently happens in the interac data for the 'request.metadata.addressCountry' column.
        return df
    else:
        df = df.str.lower()
        return df.replace( country_to_code_dict ).str.upper()

ToCountryCode = FunctionTransformer( func=to_country_code, validate=False)

mapper_country_clean = DataFrameMapper(
    [(c, ToCountryCode) for c in country_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Extract Airport
#--------------------------------------------------------------
def extract_airport_code( df ):
    # Extract the (three) letters between, but no including, the characters "-" and "'". 
    # These are the airport codes by the apparant pattern.    
    return df.str.extract("-(.*)'", expand=False)     

ExtractAirportCode = FunctionTransformer( func=extract_airport_code, validate=False)

mapper_airport_clean = DataFrameMapper(
    [(c, ExtractAirportCode) for c in airport_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Extract First n Characters
#--------------------------------------------------------------
def extract_first_n_chars(df, n):
    
    # Extract first n characters.
    regex_expression = '(.{0,' + str(n) + '})'
    df = df.str.extract(regex_expression, expand=False)
    
    return df

# The n=10 is just a default placeholder here. The actual n values are in the column dictionary.
ExtractFirstNChars = FunctionTransformer( func = extract_first_n_chars, kw_args={'n': 10}, validate=False)

mapper_first_n_char_clean = DataFrameMapper(
    [(c, 
         FunctionTransformer( func = extract_first_n_chars, kw_args={'n': v}, validate=False )) 
             for c,v in first_n_char_cols_dict.items()],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Extract URL Snippet
#--------------------------------------------------------------
from urllib.parse import urlparse
import urllib

def extract_url_snippet( df ):
    # Extract the 'netloc' and 'path' tuple parts from the URL and join back to a prescribed level.
    # This gives strings like: 
    #   'authentication.cardinalcommerce.com/ThreeDSecure/V1_0_2/PayerAuthentication'
    #
    # For example, set levels_down = 2 below to extract to:
    # 'authentication.cardinalcommerce.com/ThreeDSecure

    levels_down = 2

    mask = df.notnull()
    df.loc[mask] = df.loc[mask].apply(lambda x: ''.join( urllib.parse.urlsplit(x)[1:3]) )

    # We only want to grab a shallow part of the hierarchy (nominally just one down from the base url).
    # Split the whole thing by the /'s and then join back to levels_down in the hierarchy.

    df.loc[mask] = df.loc[mask].str.split('/').apply( lambda x: '/'.join(x[:levels_down]) )
    
    return df

ExtractUrlSnippet = FunctionTransformer( func = extract_url_snippet, validate=False )

mapper_url_clean = DataFrameMapper(
    [(c, ExtractUrlSnippet) for c in url_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Normalize Card Type
#--------------------------------------------------------------
def normalize_card_type(df):
    
    df[df.notnull()] = df[df.notnull()].replace({'VISA': 'VI', 'MASTERCARD': 'MC'})
    
    return df

NormalizeCardType = FunctionTransformer( func=normalize_card_type, validate=False )

mapper_card_type_clean = DataFrameMapper(
    [(c, NormalizeCardType) for c in card_type_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Normalize Card Year
#--------------------------------------------------------------
def normalize_card_year(df):
    
    # Data are originally floats with some numbers four digits and some two.
    # This returns the two digit float of the year. No checking is done.
    
    # One card is 2060 or so so this feature might be something to threshold on.
    
    df[df.notnull()] = df[df.notnull()].astype(int).astype(str).apply(lambda x: x[-2:]).astype(float)
    
    return df

NormalizeCardYear = FunctionTransformer( func=normalize_card_year, validate=False)

mapper_card_year_clean = DataFrameMapper(
    [(c, NormalizeCardYear) for c in card_year_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# Province Name to Code
#--------------------------------------------------------------

prov_terr = {
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland',
    'NT': 'Northwest Territories',
    'NS': 'Nova Scotia',
    'NU': 'Nunavut',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan',
    'YT': 'Yukon'
}

provinces = {
    'AB': 'Alberta',
    'BC': 'British Columbia',
    'MB': 'Manitoba',
    'NB': 'New Brunswick',
    'NL': 'Newfoundland',
    'NS': 'Nova Scotia',
    'ON': 'Ontario',
    'PE': 'Prince Edward Island',
    'QC': 'Quebec',
    'SK': 'Saskatchewan'
}

territories = {
    'NT': 'Northwest Territories',
    'NU': 'Nunavut',
    'YT': 'Yukon'
}
states = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

code_to_provinces_states_dict = dict( list(prov_terr.items()) + list(states.items()) )

for k,v in code_to_provinces_states_dict.items():
    code_to_provinces_states_dict[k] = v.lower()

# Create reverse dictionary.
provinces_states_to_code_dict = {v: k for k,v in code_to_provinces_states_dict.items()}

def to_province_code(df):
    
    # NOTE: No checks are performed except for an all NaN column which '.replace' returns as error.
    # 'event.metadata.addressProvince', for example, has invalid two letter codes [AA, AP, IT, US, PW].
    # Left for now.
    if df.isnull().all() == False:
        df = df.str.lower()
        return df.replace( provinces_states_to_code_dict ).str.upper()
    else:
        return df

ToProvinceCode = FunctionTransformer( func=to_province_code, validate=False)

mapper_province_clean = DataFrameMapper(
    [(c, ToProvinceCode) for c in province_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# To Lower Case and No Spaces
#--------------------------------------------------------------
def to_lc_no_spaces(df):

    return( df.str.replace('\s+','').str.lower() )

ToLcNoSpaces = FunctionTransformer( func = to_lc_no_spaces, validate=False )

mapper_string_to_lc_clean = DataFrameMapper(
    [(c, ToLcNoSpaces) for c in string_to_lc_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# To Datetime
#--------------------------------------------------------------
def to_datetime( df ):
    
    return pd.to_datetime( df )

ToDateTime = FunctionTransformer( func=to_datetime, validate=False)

mapper_to_datetime_clean = DataFrameMapper(
    [(c, ToDateTime) for c in datetime_cols],
    input_df=True, df_out=True
)
#--------------------------------------------------------------
# String List to Integer
#--------------------------------------------------------------
def string_list_to_int( df ):
    
    # Extract 
    regex_expression = "\['(.*)'\]" # This works.

    df = df.str.extract(regex_expression, expand=False)
    
    return df

StringListToInt = FunctionTransformer( func=string_list_to_int, validate=False)

mapper_string_list_to_int_clean = DataFrameMapper(
    [(c, StringListToInt) for c in string_list_cols],
    input_df=True, df_out=True
)

#--------------------------------------------------------------
# To Float
#--------------------------------------------------------------
import pandas as pd

def pd_to_numeric( df ):
    return pd.to_numeric( df, errors='coerce', downcast='float' )

def replace_with_zero( df ):
    return df.fillna( 0 )

Pd_ToNumeric = FunctionTransformer( func=pd_to_numeric, validate=False)

ReplaceNaNWithZero = FunctionTransformer( func=replace_with_zero, validate=False)
