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
from confluent_kafka import Consumer

import bson
from bson import json_util


def main():

    # create the Kafka Producer and Consumer
    p = Producer({'bootstrap.servers': 'confluent:9092'})
    c = Consumer({
        'bootstrap.servers': 'confluent:9092',
        'group.id': 'mygroup',
        'default.topic.config': {
            'auto.offset.reset': 'smallest'
        }
    })

    # subscribe to the main events Kafka Stream
    c.subscribe(["events"])

    def update_count(event):
        
        user = event['metadata'].get('email')
        category_action = event['eventCategory']+'_'+event['eventAction']
        created = datetime.datetime.now()
        
        if user != None:
            if count_dict.get(user) != None:
                if count_dict[user].get(category_action) == None:
                    count_dict[user][category_action] = SlidingStat(window, 1, created, len)
                else:
                    count_dict[user][category_action].add_value(1, created)
            else:
                count_dict[user] = {}
                count_dict[user][category_action] = SlidingStat(window, 1, created, len)
                
            print('User:', user, 'Event Type:', category_action, 'Count:', count_dict[user][category_action].get_stat())
        
    class SlidingStat:
        
        def __init__(self, window, value, time, method):
            self.values = [{'value': value, 'time': time}]
            self.window = window # time window in seconds
            self.method = method
            self.update_values()
            
        def get_values(self):
            return [v['value'] for v in self.values]
            
        def update_values(self):
            now = datetime.datetime.now()
            self.values = [v for v in self.values if v['time'] >= now - datetime.timedelta(seconds=self.window)]
            self.stat = self.method(self.get_values())
            
        def get_stat(self):
            self.update_values()
            
            return self.stat
        
        def add_value(self, value, time):
            self.values.append({'value': value, 'time': time})
            self.update_values()

        def log_change_kafka(self):
            '''Updates the schema changes to a kafka topic as new events come in.'''
            
            pass 
            # update = self.get_dtypes()
            # key = update['name']
            
            # p.poll(0)

            #     # Asynchronously produce a message, the delivery report callback
            #     # will be triggered from poll() above, or flush() below, when the message has
            #     # been successfully delivered or failed permanently.
            # p.produce('changelog-eventschema', key=key.encode('utf-8'), value=json_util.dumps(update).encode('utf-8'), callback=delivery_report)

            # # Wait for any outstanding messages to be delivered and delivery report
            # # callbacks to be triggered.
            # p.flush()

    

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
            update_count(json_util.loads(msg.value().decode('utf-8')), count_dict)
        except BaseException as e:
            print(e)

        c.close()

# Here's our payoff idiom!
if __name__ == '__main__':
    main()