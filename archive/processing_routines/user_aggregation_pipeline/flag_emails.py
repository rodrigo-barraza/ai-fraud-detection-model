import juno.junoutils as junoutils
import juno.junodb as junodb
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

with open('creds.json') as json_data:
    creds = json.load(json_data)

db = junodb.Database(creds)

bl = db.getBlacklist()
wl = db.getWhitelist()

# read dataframe
uas = pd.read_csv('data/uas_w_tsne.csv')

# add email flags
uas['fraud_count'] = uas['n_metadata.fraudulent_True']
uas['blocked'] = uas.email.isin(bl.email[bl.level == 'BLOCKED'])
uas['warning'] = uas.email.isin(bl.email[bl.level == 'WARNING'])
uas['whitelist'] = uas.email.isin(wl.email[wl.level == 'ALLOWED'])


# read the latest data from file
uas.to_csv('data/uas_w_fraud.csv', index=False)

end = time.time()

print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))







