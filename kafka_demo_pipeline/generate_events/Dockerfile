# base image that contains the anaconda python3 stuf
FROM continuumio/anaconda3

RUN pip install pymongo
RUN pip install confluent_kafka

# copy the data generation scripts
COPY ./* ./

# run 10 concurrent user sessions sending events to kafka
CMD python run_concurrent_sessions.py -n 5