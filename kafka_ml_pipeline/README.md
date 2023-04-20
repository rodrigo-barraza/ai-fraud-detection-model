# Prototype Machine Learning Pipeline Using Kafka, Python

This pipeline has three parts.

1. Request Listener: Poll's Einstein's MongoDB for credit card and interac purchase requests and sends them to a Kafka stream.
2. Create Request Set: Listens to the requests stream generated in 1. and for each request, queries MongoDB for the events in the hour before the request for the user. Then generates a request set and sends the request set to a Kafka stream containing request sets.
3. Listens to the request set stream and for each new request set, produces a summarized version of the request that can be used for training or generating a prediction from a machine learning model.

The next logical step would be to pipe the request summaries into a prediction routine.

Currently these services use Apache Kafka. But, the logic for consuming and producing events in a stream is very similar to that of AWS Kinesis, so this code could be modified to be deployed on AWS with Kinesis.

# Requirements:

- Docker
- Anaconda

# Running The Pipeline:

Install the requirements above.

To run the pipeline run `docker-compose up -d`

**WARNING: Note that this pipeline is reading real events from MongoDB to generate requests and create request sets. Before running this, confirm it is ok to generate load on the database. Or, take a local backup of the database and use the local copy instead.**