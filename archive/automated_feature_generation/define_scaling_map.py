from sklearn_pandas import DataFrameMapper

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing.data import QuantileTransformer

from sklearn.preprocessing import Imputer

import numpy as np

# Just import the transformer functions and leave the mappers.
# Probably won't need them but they're available if necessary.
from define_scaling_mappers import (
    original_cols,
    standard_cols,
    minmax_cols,
    maxabs_cols,
    robust_cols,
    quantile_uniform_cols,
    quantile_gaussian_cols
)

# Bundle the transformer functions into a mapper.
# NOTE: Some columns hap infinities coded as np.inf. This currently imputes the column median for np.inf's.
# The 'default = None' passes back the unselected columns unchanged. Set it to 'False' and the unselected columns are dropped.
# Otherwise set 'default' to a default transformation for unselected input columns if desired. Currently 'None'.
map_scale = DataFrameMapper(
    [([c], None) for c in original_cols] +
    [([c], [Imputer(np.inf,strategy='median'), StandardScaler()]) for c in standard_cols] +
    [([c], [Imputer(np.inf,strategy='median'), MinMaxScaler()]) for c in minmax_cols] +
    [([c], [Imputer(np.inf,strategy='median'), MaxAbsScaler()]) for c in maxabs_cols] +
    [([c], [Imputer(np.inf,strategy='median'), RobustScaler()]) for c in robust_cols] +
    [([c], [Imputer(np.inf,strategy='median'), QuantileTransformer(output_distribution='uniform')]) for c in quantile_uniform_cols] +
    [([c], [Imputer(np.inf,strategy='median'), QuantileTransformer(output_distribution='normal')]) for c in quantile_gaussian_cols],
    input_df=False, df_out=True, default=None
)