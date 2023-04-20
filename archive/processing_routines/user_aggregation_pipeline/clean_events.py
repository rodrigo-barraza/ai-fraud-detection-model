import juno.junoutils as junoutils
import time
import pandas as pd
import datetime
import numpy as np

# load event data to dataframe
print("Loading raw events, cleaning and saving to csv")
now = datetime.datetime.now()
nowstr = "_"+now.strftime("%Y%m%d%H%M%S")

beginning = time.time()
start = beginning

# read the latest data from file
edf = pd.read_csv('data/events_raw.csv', low_memory=False)

# get rid of session ids for now, probably needs to be better in the future
edfc = edf[edf.eventLabel.str.len() != len('b37ce3a5-93b3-c72d-84fd-7e212d519722')]

# clean the events
edfc = junoutils.cleanEventsDF(edfc)

# save to csv
edfc.to_csv('data/events_clean.csv', index=False)
end = time.time()

print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))



