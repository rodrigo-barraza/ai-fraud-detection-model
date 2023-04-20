# Define the column types by the real valued transformation required.
# See: 
# http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# for a description of the different scalers available.
#
# We will have to try different scalers, combinations and multiple scaled versions of the same column.
#
# The muliple scaled version option is not implemented. This may be useful when outliers have 
# significance but we don't want them to overwhelm inliers, either.
#
# Note that the variables are all greater than or equal to zero. They are currency amounts, exchange rates and counts.
# No negative numbers. This may factor into the choice of scaler.
#
# For now  use RobustScaler on all columns. This DOES introduce negative numbers.

# Available scaling types are:
#
# 1. Original data, i.e., none.
# 2. StandardScaler
# 3. MinMaxScaler
# 4. MaxAbsScaler
# 5. RobustScaler
# 6. QuantileTransformer (uniform output)
# 7. QuantileTransformer (Gaussian output)
# 8. Normalizer --> This does not apply here. It scales rows to unit norm and we are scaling columns.

original_cols = []

standard_cols = []

minmax_cols = []

maxabs_cols = []

robust_cols = ['event.metadata.amount',
    'event.metadata.blockioResponse.data.amount_sent',
    'event.metadata.blockioResponse.data.amount_withdrawn',
    'event.metadata.blockioResponse.data.network_fee',
    'event.metadata.cashAdvanceReimbursement',
    'event.metadata.cents',
    'event.metadata.firstAmount',
    'event.metadata.lastTradedPx',
    'event.metadata.mongoResponse.amount',
    'event.metadata.mongoResponse.price',
    'event.metadata.price',
    'event.metadata.prossessorResponse.amount',
    'event.metadata.prossessorResponse.charge_amount',
    'event.metadata.rate',
    'event.metadata.requestParams.amount',
    'event.metadata.requestParams.charge_amount',
    'event.metadata.requestParams.price',
    'event.metadata.requestParams.product_amount',
    'event.metadata.secondAmount',
    'event.value',
    'request.metadata.amount',
    'request.metadata.cashAdvanceReimbursement',
    'request.metadata.cents',
    'request.metadata.mongoResponse.amount',
    'request.metadata.mongoResponse.price',
    'request.metadata.rate',
    'request.metadata.requestParams.amount',
    'request.metadata.requestParams.charge_amount',
    'request.metadata.requestParams.price',
    'request.metadata.requestParams.product_amount',
    'request.value',
    'event.metadata.passwordLength']

quantile_uniform_cols = []

quantile_gaussian_cols = []
