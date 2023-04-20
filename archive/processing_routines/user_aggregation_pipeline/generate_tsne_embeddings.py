import juno.junoutils as junoutils
import time
import pandas as pd
import datetime
import numpy as np

# load event data to dataframe
print("Calculating tSNE embeddings.")
now = datetime.datetime.now()
nowstr = "_"+now.strftime("%Y%m%d%H%M%S")

beginning = time.time()
start = beginning

# read the latest data from file
uas = pd.read_csv('data/uas.csv')

# scale the data
scaled = junoutils.scaleDf(uas.drop(['email','mean_currency_price','median_currency_price','std_currency_price'], axis=1))

# calculate tSNE embeddings
emb = junoutils.calculatetSNEEmbeddings(df=scaled)
uas['tSNE_x'] = emb[:,0]
uas['tSNE_y'] = emb[:,1]

# save the summary to csv
uas.to_csv('data/uas_w_tsne.csv', index=False)
end = time.time()

print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))







