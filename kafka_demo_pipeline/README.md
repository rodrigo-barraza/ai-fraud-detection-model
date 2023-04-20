# Prototype Machine Learning Pipeline Using Kafka, Spark, Python


# Current Status

Currently have a docker-compose pipeline that consists of a number of containers that show functionality of a few different parts of the Kafka ecosystem including:

A python script that uses the Kafka producer library and MongoDB to generates anonymized event data based on real Einstein data and deliver it into a Kafka stream.

Kafka Stream processing apps written in Java that

- separate out Credit Card and Interac deposit requests from the rest of the events and,
- create 'Request sets' which are sets of events preceeding a request which are the basis for the fraud detection machine algorithms.

KSQL - Confluent's Streaming Query Language for ad-hoc querying of streams. Examples include:

- Creating a rolling aggregation of the number of events in a time window
- Creating a rollowing aggregation of the number of events by user in a time window

There are also examples of consumers written in python that store the latest aggregated event counts above in a Redis database for display in a real time dashboard, similar to how we might created streaming analytics.

# Issues

- Currently having an issue where characters in the MongoDB events are causing the Kafka Stream processing apps to fail.
- Have to run `docker-compose up -d` a number of times to get everthing working because startup of certain containers won't work until other services are up and running which takes time.

# Roadmap

1. Connect Debezium to Einstein's MongoDB to generate Kafka events directly from the database changelog.
2. Convert python scripts for data cleaning, ML training and prediction to utilize Kafka topics rather than reading from MongoDB
3. Experiment with generating alarms based on Kafka Streams of event data and posting to Slack

# Requirements:

- Docker

# Running The Pipeline:

In order for the pipeline to run correctly a credentials file with the MongoDB connection string needs to be at the following path `./generate_events/creds.json`. The format of the file is:

```
{
	"connection_string": <connection_string>
}
```

The connection string needs to be shared with you via LastPass.

In bash, while in the top level directory run `docker-compose up -d`. This should spin up the entire pipeline. There are currently some issues with some containers not starting properly if other ones arent ready yet, so you may have to run `docker-compose up -d` a few times to get everything up and running.

To inspect a running container do `docker ps` to list the running containers and then `docker attach <instance_id>`

Once the pipeline is up and running, navigate to the following page to show the experimental KSQL UI. This allows you to inspect the Kafka topics, streams, tables etc and run new KSQL queries.

http://localhost:8082

There is also an example streaming dashboard running at:

http://localhost:8050



