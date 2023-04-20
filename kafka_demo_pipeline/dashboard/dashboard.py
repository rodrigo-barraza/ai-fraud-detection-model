# plotly dashboard

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import json

from pandas.io.json import json_normalize

import datetime

import redis
from confluent_kafka import Consumer, KafkaError
import bson

import wait

wait.for_topics(['EVENTS_PER_TYPE','EVENTS_PER_USER'], host='kafka-rest',port='29080')
wait.for_host(port=6379, host='redis')

# set up connection to redis
r = redis.Redis(
    host='redis',
    port=6379)

###################################

# SUPPORT FUNCTIONS

def get_user_list():
    
    user_list_string = r.get('user_list')
    
    if user_list_string in [b'None',None]:
        return None
    else:
        user_list_object = json.loads(user_list_string.decode('utf-8'))
        return user_list_object['user_list']
      
def get_user_eventtypes(user_email):
    
    user_dict_string = r.get({'user_email': user_email})
        
    if user_dict_string in [b'None',None]:
        return None
    else:
        return json.loads(user_dict_string.decode('utf-8'))

def get_eventtypes():
    
    eventtypes_dict_string = r.get({'event_types'})
        
    if eventtypes_dict_string in [b'None',None]:
        return None
    else:
        return json.loads(eventtypes_dict_string.decode('utf-8'))
    
def email_list():
    '''Create the dropdown options.'''

    emails = sorted(get_user_list())

    options = []

    for email in emails:
        if email not in ['', None]:
            options.append({'label': email, 'value': email})

    return options

def get_user_chart_data(user_email):

    user_event_dict = get_user_eventtypes(user_email)
    x = list(user_event_dict.keys())
    y = list(user_event_dict.values())

    figure=go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                name='User Event Counts Last 60 Minutes',
                marker=go.Marker(
                    color='rgb(55, 83, 109)'
                )
            )
        ],
        layout=go.Layout(
            title='User Event Counts Last 60 Minutes',
            showlegend=True
        )
    )

    return figure

def get_events_chart_data():

    event_dict = get_eventtypes()
    event_type = list(event_dict.keys())
    event_count = list(event_dict.values())

    df = pd.DataFrame({'event_type': event_type, 'event_count': event_count})
    df = df.sort_values(by='event_count', ascending=True)
 
    figure=go.Figure(
        data=[
            go.Bar(
                x=df.event_count,
                y=df.event_type,
                name='Event Counts Last 60 Minutes',
                marker=go.Marker(
                    color='rgb(55, 83, 109)'
                ),
                orientation = 'h'
            )
        ],
        layout=go.Layout(
            title='Event Counts Last 60 Minutes',
            showlegend=True,
            yaxis={'automargin': True}
        )
    )

    return figure

                              
app = dash.Dash()
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"})
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js "})


app.layout = html.Div(children=[
    html.Div([

        html.H1(children='Einstein User Activity Dashboard'),
        
        # User Dropdown
        dcc.Dropdown(
            id='user-dropdown',
            # options=email_list(),
            # value=email_list()[0]
        ),

        dcc.Graph(
            style={'height': 300},
            id='user-eventtype-graph'
        ),

        dcc.Graph(
            style={'height': 300},
            id='eventtype-graph'
        ),

        dcc.Interval(
            id='update-interval',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])
])

@app.callback(Output('user-eventtype-graph', 'figure'),
    [Input(component_id='user-dropdown', component_property='value'),
    Input('update-interval', 'n_intervals')]
)
def update_chart(user_email, interval):
    
    figure = get_user_chart_data(user_email)

    return figure

@app.callback(Output('eventtype-graph', 'figure'),
    [Input('update-interval', 'n_intervals')]
)
def update_event_chart(interval):
    
    figure = get_events_chart_data()

    return figure

@app.callback(Output('user-dropdown', 'options'),
    [Input('update-interval', 'n_intervals')]
)
def update_dropdown(interval):
    
    return email_list()


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')