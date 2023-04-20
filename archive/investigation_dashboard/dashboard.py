# plotly dashboard

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import einstein_exchange_components as eec
import pandas as pd 
import numpy as np 
from dash.dependencies import Input, Output

import juno.junoutils as junoutils
import juno.junodb as junodb
import juno.junoplots as junoplots
                                
app = dash.Dash()
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"})
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js "})


summary = pd.read_csv('../user_aggregation_pipeline/data/uas_w_anomaly.csv')


def email_list():

    options = []

    emails = list(summary[summary.whitelist == False].sort_values(by='anomaly_score_autoencoder', ascending=False)['email'].values)
    # emails = list(summary[summary.blocked == False].sort_values(by='email', ascending=False)['email'].values)


    for email in emails:
        options.append({'label': email, 'value': email})
    
    return options

def metric_list():

    options = []

    dont_show = ['email','blocked','warning','whitelist','tSNE_x','tSNE_y', 'anomaly_isolation_forest', 'anomaly_score_autoencoder', 'biggest_anomaly_metric_autoencoder', 'biggest_anomaly_score_autoencoder']+junoutils.prefixColumns(['fraud','mean','median'], summary.columns)

    metrics = sorted(list(set(list(summary.columns)) - set(dont_show)))

    for metric in metrics:
        options.append({'label': metric, 'value': metric})
    
    return options


app.layout = html.Div(children=[
    html.Div([

        html.H1(children='Einstein Juno'),

        html.Div(children='''
            Fraud detection and anti-money laundering for the cryptocurrency industry.
        '''),
        
        # User Dropdown
        dcc.Dropdown(
            id='user-dropdown',
            options=email_list(),
            value=email_list()[0]
        ),
        # Metric Dropdown
        dcc.Dropdown(
            id='metric-dropdown',
            options=metric_list(),
            value=metric_list()[0]
        ),

        # Clustering Graph
        dcc.Graph(
            id='cluster-plot',
            figure=junoplots.cluster_figure(summary, user_emails=[email_list()[0]['value']], metric_names=[metric_list()[0]['value']])
        ),

        # Key Metric Graph
        dcc.Graph(
            id='user-keymetric-plot',
            figure=junoplots.user_keymetric_plot(summary, email_list()[0]['value'])
        ),

        # Metric Graph
        dcc.Graph(
            id='user-metric-plot',
            figure=junoplots.user_metric_plot(summary, 'youngalihamilton@gmail.com')#email_list()[0]['value'])
        ),

        # Metric Graph
        dcc.Graph(
            id='metric-plot',
            figure=junoplots.metric_figure(summary, metric_list()[10]['value'])
        ),

        # Metric Distribution
        dcc.Graph(
            id='metric-dist',
            figure=junoplots.metric_dist(summary, metric_list()[0]['value'])
        ),

        # # Markov
        # eec.MarkovStateComponent(
        #     id='markov-diagram',
        #     nodes=junoutils.user_node_json(email_list()[0]['value'], nodes),
        #     links=junoutils.user_link_json(email_list()[0]['value'], links)
        # ),

        # # User Flow Chart
        ], className='container-fluid'),
])


@app.callback(
    Output(component_id='cluster-plot', component_property='figure'),
    [Input(component_id='user-dropdown', component_property='value'),
    Input(component_id='metric-dropdown', component_property='value')]
)
def update_cluster(user, metric):
    return junoplots.cluster_figure(summary, user_emails=[user], metric_names=[metric])

@app.callback(
    Output(component_id='user-metric-plot', component_property='figure'),
    [Input(component_id='user-dropdown', component_property='value')]
)
def update_user_metric(user):
    return junoplots.user_metric_plot(summary, user)

@app.callback(
    Output(component_id='user-keymetric-plot', component_property='figure'),
    [Input(component_id='user-dropdown', component_property='value')]
)
def update_user_keymetric(user):
    return junoplots.user_keymetric_plot(summary, user)

@app.callback(
    Output(component_id='metric-plot', component_property='figure'),
    [Input(component_id='metric-dropdown', component_property='value')]
)
def update_metric(metric):
    return junoplots.metric_figure(summary, metric)

@app.callback(
    Output(component_id='metric-dist', component_property='figure'),
    [Input(component_id='metric-dropdown', component_property='value')]
)
def update_metric_dist(metric):
    return junoplots.metric_dist(summary, metric)


if __name__ == '__main__':
    app.run_server(debug=True)