# Einstein Data Science Package

This is a python package that contains some helper methods to read data from MongoDB, utilities to massage
the data and code to create plots related to the aggregate data and to support an experimental dashboard.

- db.py
    - includes helper methods that wrap the pymongo package to create useful datasets for analytics and machine learning
- event_processing.py
    - contains code to clean / normalize the raw events from mongo, into a smaller more usable dataset with consistent data types
    - adds functionality to create interact and credit card purchase request sets and summaries
- ml.py
    - includes methods to train autoencoders for anomaly detection (old and needs updating)
    - includes methods to train an optimal random forest fraud detection model on credit card and interac requests
- utils.py
    - misc methods to help with data pre-processing etc.