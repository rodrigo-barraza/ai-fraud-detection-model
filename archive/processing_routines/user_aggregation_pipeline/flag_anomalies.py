import juno.junoutils as junoutils
import juno.junodb as junodb
import juno.junoml as junoml
import pandas as pd
import numpy as np
import json
import time, datetime

# load event data to dataframe
print("Flagging fraud and whitelist emails.")
now = datetime.datetime.now()
nowstr = "_"+now.strftime("%Y%m%d%H%M%S")

beginning = time.time()
start = beginning

# read dataframe
uas = pd.read_csv('data/uas_w_fraud.csv')

# get rid na and non-numerical columns
numerical = uas.drop(['email','mean_currency_price','median_currency_price','std_currency_price','blocked','warning','whitelist','fraud_count'], axis=1)

# scale the data
scaled = junoutils.scaleDf(numerical)

# get the fraud column
fraud_column = uas['blocked']

# train the autoencoder
autoencoder = junoml.train_autoencoder(df=scaled, fraud_column=fraud_column)

# get the predictions
pred_df = junoml.autoencoder_prediction(scaled, autoencoder)
pred_df['email'] = uas['email']

uas = uas.set_index('email').join(pred_df.set_index('email'))

# train the isolation forest
ifst = junoml.train_isolation_forest(scaled, contamination=0.01)

# append isolation_forest_result
uas['anomaly_isolation_forest'] = junoml.isolation_forest_predictions(data=scaled , isolation_forest=ifst)

# reset the index
uas = uas.reset_index()

# read the latest data from file
uas.to_csv('data/uas_w_anomaly.csv', index=False)



end = time.time()

print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))







