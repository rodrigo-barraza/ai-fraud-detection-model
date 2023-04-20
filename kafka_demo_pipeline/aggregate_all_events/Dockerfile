# base image that contains the anaconda python3 stuf
FROM continuumio/anaconda3

RUN pip install pymongo
RUN pip install confluent_kafka
RUN pip install redis

# copy the data generation scripts
COPY ./* ./

CMD python event_metrics_aggregator.py