'''
Example program that reads events recorded by the einstein eventlog in JSON format and keeps track of 
the schema of the JSON records so this can be used for processing later on.

Uses the base Consumer and Producer libraries rather than the stream processing library.

Every time the schema is updated a record is written to the changelog-eventschema kafka stream, which can be used
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

    def update_schema(event, count_dict, current_level=''):
        '''
        Updates the schema dictionary with the latest values from the given event.
        '''
        
        # for each field in the JSON event
        for field in event.keys():
            
            if current_level == '':
                field_name = field
            else:
                field_name = current_level+'.'+field
            
            if type(event[field]) == type({}):
                update_schema(event[field], count_dict, current_level=field_name)
            else:
                if count_dict.get(field_name) == None:
                    count_dict[field_name] = SchemaValue(field_name, event[field])
                else:
                    count_dict[field_name].add_value(event[field])
                    
        
    class SchemaValue:
        '''
        Class to store objects that keep track of the schema of different fields in the different events.
        '''
        
        def __init__(self, name, value):
            
            dtype = value.__class__.__name__
            self.name = name
            self.types = {}
            self.latest_value = None
            self.majority_type = {'dtype': type(value), 'count': 1}
            self.latest_type = type(value)
            self.add_value(value)
            
        def add_value(self, value):
            
            dtype = value.__class__.__name__
            
            if self.types.get(dtype) == None:
                self.types[dtype] = {}
                self.types[dtype]['values'] = []
                self.types[dtype]['values'].append(value)
                self.types[dtype]['count'] = 1
            else:
                self.types[dtype]['values'].append(value)
                self.types[dtype]['count'] += 1
                
            self.latest_type = dtype
            self.latest_value = value
            
            if self.majority_type['count'] <= self.types[dtype]['count']:
                self.majority_type = {'dtype': dtype, 'count': self.types[dtype]['count']}
                
            self.log_change_kafka()
                
        def log_change_kafka(self):
            '''Updates the schema changes to a kafka topic as new events come in.'''
            
            update = self.get_dtypes()
            key = update['name']
            
            p.poll(0)

                # Asynchronously produce a message, the delivery report callback
                # will be triggered from poll() above, or flush() below, when the message has
                # been successfully delivered or failed permanently.
            p.produce('changelog-eventschema', key=key.encode('utf-8'), value=json_util.dumps(update).encode('utf-8'), callback=delivery_report)

            # Wait for any outstanding messages to be delivered and delivery report
            # callbacks to be triggered.
            p.flush()
            
            
        def get_type_distribution(self):
            dist = {}
            
            for data_type in self.types.keys():
                schema[data_type] = self.types[data_type]['count']
                
            return dist
            
        def get_latest_type(self):
            
            return self.latest_type
            
        def get_majority_type(self):
            
            return self.majority_type
        
        def get_dtypes(self):
            
            result = {}
            result['name'] = self.name
            result['types'] = {}
            
            for dtype in self.types.keys():
                
                result['types'][dtype] = self.types[dtype]['count']
                
            result['latest_type'] = self.latest_type
            result['majority_type'] = self.majority_type['dtype']
            
            return result

        def __repr__(self):
            return json.dumps(self.get_dtypes())


        schema_dict = {}

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
            update_schema(json_util.loads(msg.value().decode('utf-8')), schema_dict)
        except BaseException as e:
            print(e)

        c.close()

# Here's our payoff idiom!
if __name__ == '__main__':
    main()