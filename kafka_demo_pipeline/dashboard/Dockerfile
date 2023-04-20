# base image that contains the anaconda python3 stuf
FROM continuumio/anaconda3

RUN pip install pymongo
RUN pip install confluent_kafka
RUN pip install redis
RUN pip install --upgrade dash  # The core dash backend
RUN pip install --upgrade dash-renderer  # The dash front-end
RUN pip install --upgrade dash-html-components     # HTML components
RUN pip install --upgrade dash-core-components   # Supercharged components
RUN pip install plotly --upgrade  # Latest Plotly graphing library

# copy the data generation scripts
COPY ./* ./

CMD python dashboard.py