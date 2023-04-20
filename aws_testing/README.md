# AWS Testing

As part of the Global Events project. We tested using various AWS Services, including Kinesis Firehose, AWS ElasticSearch Service and S3. Firehose, Elastic Search and S3 were all configured manuallly using the AWS Console, so there is no configuration files here.

[poll_to_kinesis.py](poll_to_kinesis.py) is a script that poll's the mongo event database and sends events through the Kinesis Firehose. It keeps track of the latest mongo `_id` of the event as a marker and stores that in DynamoDB. This could be easily converted to a Lambda and used to generate a stream of events without changing the EventService to deliver to Kinesis. This might be useful for testing.