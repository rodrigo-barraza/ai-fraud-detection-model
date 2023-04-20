'''
Example program that reads events recorded by the einstein eventlog in JSON format and keeps track of 
sliding statistics for each user as new events come in.

Uses the base Consumer and Producer libraries rather than the stream processing library.

Every time the user values are updated a record is written to the changelog-usermetrics kafka stream, which can be used
to reconstruct state in the event of a failure.
'''


# for numerical work
import pandas as pd
import numpy as np

import datetime
import json

from confluent_kafka import Producer
from confluent_kafka import Consumer, KafkaError

import bson
from bson import json_util

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--topic', required=True)
    args = parser.parse_args()

    print("Subscribing to topic {}".format(args.topic))

    # create the Kafka Producer and Consumer
    c = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'mygroup',
        'default.topic.config': {
            'auto.offset.reset': 'smallest'
        }
    })

    # subscribe to the main events Kafka Stream
    c.subscribe([args.topic])

    while True:

        msg = c.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(msg.error())
                break

        try:
            print(msg.value().decode('utf-8'),'\n')
        except BaseException as e:
            print(e)

    c.close()

# Here's our payoff idiom!
if __name__ == '__main__':
    main()