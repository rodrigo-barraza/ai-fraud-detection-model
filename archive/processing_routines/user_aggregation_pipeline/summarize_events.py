import juno.junoutils as junoutils
import time
import pandas as pd
import datetime
import numpy as np

# load event data to dataframe
print("Loading clean events, summarizing and saving summary to csv.")
now = datetime.datetime.now()
nowstr = "_"+now.strftime("%Y%m%d%H%M%S")

beginning = time.time()
start = beginning

# read the latest data from file
edf = pd.read_csv('data/events_clean.csv', low_memory=False)

# summarize the records
uas = junoutils.summarizeRecordsSmall(edf)

# save the summary to csv
uas.to_csv('data/uas.csv', index=False)
end = time.time()

print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))







