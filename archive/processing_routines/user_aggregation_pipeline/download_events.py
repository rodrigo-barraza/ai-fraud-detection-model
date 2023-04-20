import juno.junodb as junodb
import time
import pandas as pd
import datetime
import numpy as np
import json


with open('creds.json') as json_data:
    creds = json.load(json_data)

db = junodb.Database(creds)

# load event data to dataframe
print("Loading events dataframe and saving to csv")
now = datetime.datetime.now()
nowstr = "_"+now.strftime("%Y%m%d%H%M%S")
look_back = now - datetime.timedelta(days=7)

beginning = time.time()
start = beginning
# edb = db.getEvents(flat=True, date=look_back)

edb = db.getEvents(flat=True)

edb.to_csv('data/events_raw.csv', index=False)
end = time.time()
print("Completed in : {} seconds".format(end-start))

print("Done in {} seconds".format(end-beginning))



