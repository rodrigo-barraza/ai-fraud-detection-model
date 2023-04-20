# -----------------------------------------------------------------------------------------
# BASED ON https://www.datacamp.com/community/tutorials/machine-learning-models-api-python
# -----------------------------------------------------------------------------------------

# Dependencies
import os
import pickle # Serialization module.

# ML related.
import sklearn
from sklearn.externals import joblib
import pandas as pd

#  API definition
def handle(req):

    #print(req)

    current_dir = os.path.dirname(__file__)

    # Load trained linear regression model.
    lr = joblib.load(os.path.join(current_dir, "model.pkl")) # Load "model.pkl"
    #print ('Model loaded')

    model_columns = joblib.load(os.path.join(current_dir, "model_columns.pkl")) # Load "model_columns.pkl"
    #print ('Model columns loaded')

    # Convert query string into pandas dataframe.
    query = pd.get_dummies(pd.read_json(req))
    query = query.reindex(columns=model_columns, fill_value=0)

    # Perform regression prediction.
    prediction = list(lr.predict(query))

    return(prediction)
