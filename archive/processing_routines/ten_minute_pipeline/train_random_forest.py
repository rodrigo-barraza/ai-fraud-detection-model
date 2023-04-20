import sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np 
import pickle

last10mins = pd.read_csv('data/all_interact_summaries.csv')

# the fraudsters we know about right now
fraudsters = ['gaelkevin@hotmail.com', 'royer.8383@gmail.com', 'adventurous7381@gmail.com']

last10mins.head()

last10mins['fraud'] = False

# update 
last10mins.fraud[last10mins.email.isin(fraudsters) & (last10mins.ca_interac_request > 3)] = True

fraud = last10mins[last10mins.fraud == True]
not_fraud = last10mins[last10mins.fraud == False]

print(fraud.shape[0])

n_fraud = fraud.shape[0]
n_not_fraud = not_fraud.shape[0]

fraud = fraud.iloc[np.random.randint(0,n_fraud,n_not_fraud)]

data = pd.concat([fraud,not_fraud])

# set up data for algorithm
X = data.drop(['fraud','email'], axis=1)
y = data.fraud

print("Frauds: ", len(X[y == True]))
print("Not Frauds: ", len(X[y == False]))

# save the columns to check against the API
columns = X.columns

# set up the classifier
rf = RandomForestClassifier(max_depth=5)

# train the classifier
rf.fit(X=X, y=y)

# 
print("Accuracy is: ", np.sum(rf.predict(X) == y)/len(y))

# model
model = {'model': rf, 'column_order': columns}

# save the trained model to a file
pickle.dump(model, open('models/last10randomforest.pickle', "wb" ))

