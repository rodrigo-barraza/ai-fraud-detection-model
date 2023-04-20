import requests
import wait

wait.for_topics(['events'], host='kafka-rest',port='29080')
wait.for_host(port=8088, host='ksql-server')


headers = {'Content-type': 'application/json', 'Accept': 'application/json'}

json_template = {
  "ksql": "",
  "streamsProperties": {
    "ksql.streams.auto.offset.reset": "earliest"
  }
}

queries = [
    "CREATE STREAM event_stream (metadata STRING, eventCategory STRING, eventAction STRING) WITH (kafka_topic = 'events', value_format = 'json');",
    "CREATE STREAM user_eventtypes as SELECT extractjsonfield(metadata, '$.email') as email, concat(concat(eventcategory,'_'),eventaction) as eventtype from event_stream;",
    "CREATE TABLE events_per_user AS SELECT email, eventtype, COUNT(*) AS events FROM user_eventtypes WINDOW TUMBLING (SIZE 60 minutes) GROUP BY email, eventtype;",
    "CREATE TABLE events_per_type AS SELECT eventtype, COUNT(*) AS events FROM user_eventtypes WINDOW TUMBLING (SIZE 60 minutes) GROUP BY eventtype;",
]

for query in queries:

    json = json_template
    json["ksql"] = query

    r = requests.post('http://ksql-server:8088/ksql', json=json, headers=headers)

    print('Executing', query,'\n\n')
    print('Result', r.json(),'\n\n')