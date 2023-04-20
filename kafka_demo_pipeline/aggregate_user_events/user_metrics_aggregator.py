import redis
from confluent_kafka import Consumer, KafkaError
import bson
import json
import wait

wait.for_topics(['EVENTS_PER_USER'], host='kafka-rest',port='29080')
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

c.subscribe(['EVENTS_PER_USER'])

def update_user_list(user_email):
    
    user_list_string = r.get('user_list')
    
    if user_list_string in [b'None',None]:
        user_list_object = {'user_list': []}
    else:
        user_list_object = json.loads(user_list_string.decode('utf-8'))
    
    user_list_set = set(user_list_object['user_list'])
    
    user_list_set.add(user_email)
    
    user_list_object['user_list'] = list(user_list_set)
    
    r.set('user_list', json.dumps(user_list_object))

def get_user_list():
    
    user_list_string = r.get('user_list')
    
    if user_list_string in [b'None',None]:
        return None
    else:
        user_list_object = json.loads(user_list_string.decode('utf-8'))
        return user_list_object['user_list']
    
def update_user_eventtypes(user_email, eventtype, count):
    
    user_dict_string = r.get({'user_email': user_email})
        
    if user_dict_string in [b'None',None]:
        user_dict = {}
    else:
        user_dict = json.loads(user_dict_string.decode('utf-8'))

    user_dict[eventtype] = count

    r.set({'user_email': user_email}, json.dumps(user_dict))
    
def get_user_eventtypes(user_email):
    
    user_dict_string = r.get({'user_email': user_email})
        
    if user_dict_string in [b'None',None]:
        return None
    else:
        return json.loads(user_dict_string.decode('utf-8'))
    
def process_event(json_event):
    
    email = json_event['EMAIL']
    eventtype = json_event['EVENTTYPE']
    event_count = json_event['EVENTS']
    
    print(email)
    if email != None:
        update_user_list(email)
        update_user_eventtypes(email, eventtype, event_count)
        
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
