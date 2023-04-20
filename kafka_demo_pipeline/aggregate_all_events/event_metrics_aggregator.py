import redis
from confluent_kafka import Consumer, KafkaError
import bson
import json

import wait

wait.for_topics(['EVENTS_PER_TYPE'], host='kafka-rest',port='29080')
wait.for_host(port=6379, host='redis')

r = redis.Redis(
    host='redis',
    port=6379)

c = Consumer({
    'bootstrap.servers': 'kafka:29092',
    'group.id': 'mygroup',
    'default.topic.config': {
        'auto.offset.reset': 'smallest'
    }
})

c.subscribe(['EVENTS_PER_TYPE'])

def update_eventtypes(eventtype, count):
    
    eventtypes_dict_string = r.get({'event_types'})
        
    if eventtypes_dict_string in [b'None',None]:
        eventtypes_dict = {}
    else:
        eventtypes_dict = json.loads(eventtypes_dict_string.decode('utf-8'))

    eventtypes_dict[eventtype] = count

    r.set({'event_types'}, json.dumps(eventtypes_dict))

def get_eventtypes():
    
    eventtypes_dict_string = r.get({'event_types'})
        
    if eventtypes_dict_string in [b'None',None]:
        return None
    else:
        return json.loads(eventtypes_dict_string.decode('utf-8'))
    
def process_event(json_event):
    
    eventtype = json_event['EVENTTYPE']
    event_count = json_event['EVENTS']
    
    if eventtype != None:
        update_eventtypes(eventtype, event_count)
        
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
    
    decoded = msg.value().decode('utf-8')
    json_event = json.loads(decoded)
    
    process_event(json_event)
        
    print('Received message: {}'.format(msg.value().decode('utf-8')))

c.close()