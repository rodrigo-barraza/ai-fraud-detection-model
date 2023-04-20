# plotly dashboard

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd 
import numpy as np 
import plotly.graph_objs as go
from dash.dependencies import Input, Output

from pymongo import MongoClient

from bson import json_util

import json

from pandas.io.json import json_normalize

import dash_table_experiments as dt

import datetime

import urllib

import flask

import io
from io import StringIO

from einsteinds import db as edb
from einsteinds import plots as eplots
from einsteinds import utils

from textwrap import dedent as d

# load the database credentials from file
with open('creds.json') as json_data:
    creds = json.load(json_data)

db = edb.Database(creds)
plots = eplots.EventPlots(creds)

def get_aggregation_function(name):

    agg_function_dict = {
        'mean':np.nanmean, 
        'median': np.nanmedian, 
        'unique': pd.Series.nunique, 
        'count': pd.Series.count,
        'max': np.nanmax,
        'min': np.nanmin,
        'sum': np.nansum
    }

    func = agg_function_dict.get(name)

    return func


def get_unique_col_values(column, events):

    return sorted(list(events[column].unique()))

def filter_column(column, value, events):

    return events[events[column] == value]

def max_date():
    
    return datetime.datetime.now()

def min_date():
    
    return datetime.datetime.now() - datetime.timedelta(days=7)

def email_list(user_list=None):
    '''Create the dropdown options.'''

    if user_list == None:
        emails = db.get_user_list()
    else:
        emails = user_list

    options = []

    for email in emails:
        options.append({'label': email, 'value': email})
    
    return options

                                
app = dash.Dash()
app.css.append_css({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css"})
app.scripts.append_script({"external_url": "https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js "})


app.layout = html.Div(children=[
    html.Div([
        html.H1(children='Einstein User Activity Dashboard'),

        # date dropdown
        dcc.DatePickerRange(
            id='date-range',
            start_date=datetime.datetime.today() - datetime.timedelta(days=7),
            end_date=datetime.datetime.today()
        ),

        dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='User Inspector', children=[
                # User Dropdown
                dcc.Dropdown(
                    id='user-dropdown',
                    options=email_list(),
                    value=email_list()[0]
                ),

                dcc.Dropdown(
                    id='aggregation-field',
                    options = [{'label': '_id', 'value': '_id'}],
                    value = {'label': '_id', 'value': '_id'}
                ),

                dcc.Dropdown(
                    id='aggregation-function',
                    options = [{'label': 'mean', 'value': 'mean'},
                            {'label': 'median', 'value': 'median'},
                            {'label': 'unique', 'value': 'unique'},
                            {'label': 'count', 'value': 'count'},
                            {'label': 'sum', 'value': 'sum'}],
                    value={'label': 'count', 'value': 'count'}
                ),

                html.Div(id='user-plots'),

                html.A(
                    'Download CSV',
                    id='csv-link',
                    download="user_log.csv",
                    href="",
                    target="_blank"
                ),

                dt.DataTable(
                    rows=[{}], # initialise the rows
                    row_selectable=True,
                    filterable=True,
                    sortable=True,
                    selected_row_indices=[],
                    id='datatable'
                ),
            ]),
            dcc.Tab(label='Overview', children=[
                html.Div([
                    dcc.Graph(id='daily-active-users'),
                    dcc.Graph(id='daily-new-users'),
                    dcc.Graph(id='daily-sessions'),
                    dcc.Graph(id='most-active-users'),
                    dcc.Graph(id='overview-deposits'),
                    dcc.Graph(id='overview-trades'),
                    html.Div([
                    dcc.Markdown(d("""
                            **Click Data**

                            Click on points in the graph.
                        """)),
                        html.Pre(id='click-data'),
                    ]),
                ])
            ]),
            dcc.Tab(label='Deposits', children=[
                html.Div(id='deposits')
            ]),
            dcc.Tab(label='Trades', children=[
                html.Div(id='trades-div')
            ]),
        ]),
        ], className='container-fluid'),
])


@app.callback(Output('user-plots', 'children'),
    [Input(component_id='user-dropdown', component_property='value'),
    Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date'),
    Input(component_id='aggregation-field', component_property='value'),
    Input(component_id='aggregation-function', component_property='value')]
)
def update_metrics(user, start_date, end_date, field, agg_name):

    ev = db.get_events_in_range(user=user, start_date=start_date, end_date=end_date)

    print(agg_name)

    function = get_aggregation_function(agg_name)

    def generate_table(dataframe, max_rows=10):
        return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))]
        )

    ev['date'] = ev.created.apply(lambda x: str(x)[0:10])

    def isnumber(x):
        try:
            float(x)
            return True
        except:
            return False

    if agg_name in ['mean','median','sum']:
        try:
            ev[field] = pd.to_numeric(ev[field], errors='coerce')

            if np.sum(ev[field].isnull() == False) == 0:
                return "Can't do numerical aggregation on non-numeric datatype"
        except:
            return "Can't do numerical aggregation on non-numeric datatype"

    gb = (ev.groupby(['date','eventCategory','eventAction','eventLabel'], as_index=False)[field]
        .aggregate(function)
        .rename(index=str, columns={field: agg_name})
    )

    # lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}

    htmls=[]

    for category in gb.eventCategory.unique():
        for action in gb.eventAction.unique():

            this_df = gb[(gb.eventCategory == category) & (gb.eventAction == action)]

            if this_df.shape[0] > 0:

                traces = [go.Scatter(x=this_df[this_df.eventLabel == label]['date'], 
                                    y=this_df[this_df.eventLabel == label][agg_name], 
                                    name=label) for label in this_df.eventLabel.unique()]

                all_this = this_df.groupby('date', as_index=False)[agg_name].sum().sort_values(by='date')

                traces.append(go.Scatter(x=all_this['date'], 
                                    y=all_this[agg_name], 
                                    name='all'))

                layout = go.Layout({'title': (category+" "+action).title(), 'xaxis': {'range': [gb.date.min(),gb.date.max()]}})

                htmls.append(dcc.Graph(style={'height': 300}, 
                    id=category+"-"+action+'-'+'plot', 
                    figure=go.Figure(data=traces, layout=layout)
                ))



    
    return htmls


@app.callback(Output('user-dropdown', 'options'),
    [Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')]
)
def update_user_options(start_date, end_date):

    user_list = db.get_user_list(start_date=start_date, end_date=end_date)
    
    return email_list(user_list)


@app.callback(Output(component_id='user-dropdown', component_property='value'),
    [Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')]
)

def update_user_value(start_date, end_date):

    user_list = db.get_user_list(start_date=start_date, end_date=end_date)
    
    return email_list(user_list)[0]


@app.callback(Output('datatable', 'rows'),
[Input(component_id='user-dropdown', component_property='value'),
    Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_datatable(user, start_date, end_date):
    
    ev = db.get_events_in_range(user=user, start_date=start_date, end_date=end_date).dropna(axis=1, how='all')

    ev.columns = [col.replace('metadata.','') if 'metadata.' in col else col for col in ev.columns]

    return ev.drop('_id',axis=1).to_dict(orient='records')

@app.callback(Output('aggregation-field', 'options'),
[Input(component_id='user-dropdown', component_property='value'),
    Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_field_choices(user, start_date, end_date):
    
    ev = db.get_events_in_range(user=user, start_date=start_date, end_date=end_date).dropna(axis=1, how='all')

    return [{'label': col, 'value': col} for col in sorted(ev.columns)]


@app.callback(Output('csv-link', 'href'),
[Input(component_id='user-dropdown', component_property='value'),
    Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_link(user, start_date, end_date):
    return '/dash/urlToDownload?user={}&start_date={}&end_date={}'.format(user,start_date,end_date)

@app.server.route('/dash/urlToDownload') 
def download_csv():
    user = flask.request.args.get('user')
    start_date = flask.request.args.get('start_date')
    end_date = flask.request.args.get('end_date')

    ev = db.get_events_in_range(user=user, start_date=start_date, end_date=end_date).dropna(axis=1, how='all')

    ev.to_csv('./dowloadFile.csv', index=False)
    ev_csv = ev.to_csv(index=False, encoding='utf-8')

    print(user, start_date, end_date)
    print(ev_csv[0:100])

    str_io = io.StringIO()
    str_io.write(ev_csv)
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(mem,
                           mimetype='text/csv',
                           attachment_filename='downloadFile.csv',
                           as_attachment=True)


@app.callback(Output('daily-active-users', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_daily_active_users(start_date, end_date):
    return plots.users_by_day(start_date=start_date, end_date=end_date)

@app.callback(Output('daily-new-users', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_daily_new_users(start_date, end_date):
    return plots.daily_new_users(start_date=start_date, end_date=end_date)

@app.callback(Output('daily-sessions', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_daily_sessions(start_date, end_date):
    return plots.sessions_by_day(start_date=start_date, end_date=end_date)

@app.callback(Output('most-active-users', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_most_active_users(start_date, end_date):
    return plots.most_active_users(start_date=start_date, end_date=end_date)

@app.callback(Output('overview-deposits', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_overview_deposits(start_date, end_date):
    return plots.deposits_by_day(start_date=start_date, end_date=end_date)

@app.callback(Output('overview-trades', 'figure'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_overview_trades(start_date, end_date):
    return plots.trades_by_day(start_date=start_date, end_date=end_date)

@app.callback(Output('trades-div', 'children'),
[Input(component_id='date-range', component_property='start_date'),
    Input(component_id='date-range', component_property='end_date')])
def update_trades(start_date, end_date):
    
    htmls=[]

    htmls.append(dcc.Graph(id='trade-value-summary', figure=plots.daily_trade_summary(start_date=start_date, end_date=end_date)))

    for figobj in plots.detailed_trade_plots(start_date, end_date):

        htmls.append(dcc.Graph(id=figobj['pair'], figure=figobj['fig']))
    
    return htmls

@app.callback(
    Output('click-data', 'children'),
    [Input('overview-trades', 'clickData')])
def display_click_data(clickData):
    day = datetime.datetime.strptime(clickData['points'][0]['x'], "%Y-%m-%d")

    start = day - datetime.timedelta(hours=7)
    end = start + datetime.timedelta(days=1)

    return json_util.dumps(db.get_trades(start, end).sort_values('estimated_value', ascending=False).to_dict('records'), indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)