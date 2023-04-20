from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import FunctionTransformer

# Just import the transformer functions and leave the mappers.
# Probably won't need them but they're available if necessary.
from define_cleaning_mappers import (
    BitcoinToBTC, bitcoin_cols,
    ToCountryCode, country_cols,
    ExtractAirportCode, airport_cols,
    extract_first_n_chars, first_n_char_cols_dict, # We don't use a FunctionTransformer thing since we need to pass parameter 'n'.
    ExtractUrlSnippet, url_cols,
    NormalizeCardType, card_type_cols,
    NormalizeCardYear, card_year_cols,
    ToProvinceCode, province_cols,
    ToLcNoSpaces, string_to_lc_cols,
    ToDateTime, datetime_cols,
    StringListToInt, string_list_cols,
    Pd_ToNumeric, ReplaceNaNWithZero, float_cols,
    id_cols,
    no_cleaning_cols
)

# Bundle the transformer functions into a mapper.
# The 'default = False' drops columns that were not selected by the '*_cols' variables.
# Setting 'default' to None will pass back the unselected columns unchanged. 
# Otherwise set 'default' to a default transformation for unselected input columns if desired. 
# Currently set to 'False'. The 'map_clean' function does the initial selection of columns to be used (~150) out of the ~350 available.
map_clean = DataFrameMapper(
    [(c, BitcoinToBTC) for c in bitcoin_cols] +
    [(c, ToCountryCode) for c in country_cols] +
    [(c, ExtractAirportCode) for c in airport_cols] +
    [(c, 
         FunctionTransformer( func = extract_first_n_chars, kw_args={'n': v}, validate=False )) # Do this way to pass parameter 'n'.
             for c,v in first_n_char_cols_dict.items()] +
    [(c, ExtractUrlSnippet) for c in url_cols] +
    [(c, NormalizeCardType) for c in card_type_cols] +
    [(c, NormalizeCardYear) for c in card_year_cols] +
    [(c, ToProvinceCode) for c in province_cols] +
    [(c, ToLcNoSpaces) for c in string_to_lc_cols] +
#    [(c, ToDateTime) for c in datetime_cols] + # NOTE: This incorrectly converts datatimes to nanoseconds for some reason. It doesn't on small test snippet code and works properly there, though. Don't understand it.
    [(c, None) for c in datetime_cols] +        # Just leave as 'None' for now since the two columns are already in datetime format.
    [(c, StringListToInt) for c in string_list_cols] +
    [(c, [Pd_ToNumeric, ReplaceNaNWithZero]) for c in float_cols] + # NOTE: These functions take 1-D inputs so no [] brackets around c.
    [(c, None) for c in no_cleaning_cols] +
    [(c, None) for c in id_cols],
    input_df=True, df_out=True, default=False
)