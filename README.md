# data-science-poc


Initial Proof of Concept of Data Science / Machine Learning

Directory Listing:

- [archive](archive/)
    - Older code saved for reference, but not currently being worked on. Would require updating to further used.
- [aws_testing](aws_testing/)
    - Examples script that polls MongoDB and delivers events to Kinesis Firehose
- [data](data/)
    - Data directory that holds working copy of data files, including a full backup of the MongoDB. Data files are not committed to the github repo.
- [einsteinds](einsteinds/)
    - A python package that contains code for accessing specific information from the MongoDB, creating visuals, cleaning the data and training ml models.
- [kafka_demo_pipeline](kafka_demo_pipeline/)
    - A containerized kafka pipeline that demonstrates the various components of the kafka ecosystem and how they could be used to handle events.
- [kafka_ml_pipeline](kafka_ml_pipeline/)
    - The begginings of creating the event flow for the machine learning algorithm, but not complete as we decided to change gears and move to a simpler AWS service model.
- [live_dashboard](live_dashboard/)
    - A plotly dashboard for experimenting with visualizations around things like trades, purchases, user activity, using methods from the einsteinds library.
- [notebooks](notebooks/)
    - A folder to hold code examples, working analyses and experiments all in Jupyter Notebook/Lab
- [simple_fraud_detection_model](simple_fraud_detection_model/)
    - A self contained fraud detection model that uses the einsteinds package and could be deployed inside a container and run off the current MongoDB setup. The goal here was to create a minimum working example.
