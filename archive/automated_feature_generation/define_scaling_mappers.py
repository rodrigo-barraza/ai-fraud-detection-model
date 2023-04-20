from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing.data import QuantileTransformer

# Get the column groups by scaling function.
from define_column_scaling_types import (
    original_cols,
    standard_cols,
    minmax_cols,
    maxabs_cols,
    robust_cols,
    quantile_uniform_cols,
    quantile_gaussian_cols
)

# NO SPECIALIZED MAPPERS REQUIRED.

# Go directly to 'define_scaling_map.py'.