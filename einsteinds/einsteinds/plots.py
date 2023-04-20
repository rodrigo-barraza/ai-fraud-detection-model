# for plotting
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.plotly as py

import datetime

from einsteinds import utils
from einsteinds import event_processing
from einsteinds import db

import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity

def cluster_figure(df, user_emails=None, metric_names=None, top=20):

    X_tsne = df[['tSNE_x','tSNE_y']].values

    trace = go.Scatter(
        x = X_tsne[:,0],
        y = X_tsne[:,1],
        mode = 'markers',
        name = 'all users',
        text = df.email,
        opacity = 0.6,
        marker = dict(color='rgb(83,176,277)')
    )

    y = df.email

    data = [trace]

    if metric_names != None:
        variables_to_highlight = metric_names
        trace_highlights = [np.argsort(df[col].values)[::-1][0:top] for col in variables_to_highlight]

        highlights = []

        

        for name, idx in zip(variables_to_highlight, trace_highlights):

            # print(df[name][idx])

            highlights += [go.Scatter(dict(
                x = X_tsne[idx,0],
                y = X_tsne[idx,1],
                mode = 'markers',
                marker = {'size': df[name][idx].astype('float64').values/10},
                name = 'top {} '.format(top)+name,
                text = y[idx]+': '+df[name][idx].astype(str)+' '+name
            ))]
        
        data = data + highlights

    if user_emails != None:

        users = [go.Scatter(dict(
                x = df['tSNE_x'][df.email.isin(user_emails)],
                y = df['tSNE_y'][df.email.isin(user_emails)],
                mode = 'markers',
                text = df['email'][df.email.isin(user_emails)],
                name = 'Selected Users'
            ))]
        
        data = data + users
    
    fraudsters = [go.Scatter(dict(
                x = df['tSNE_x'][df['blocked'] == True],
                y = df['tSNE_y'][df['blocked'] == True],
                mode = 'markers',
                marker = {'size': 5},
                text = df['email'][df['blocked'] == True],
                name = 'Suspected Fraud'
            ))]

    fraud_count = [go.Scatter(dict(
                x = df['tSNE_x'][df['fraud_count'] > 0],
                y = df['tSNE_y'][df['fraud_count'] > 0],
                mode = 'markers',
                marker = {'size': utils.scaleValues(df['fraud_count'][df['fraud_count'] > 0])*10},
                text = df['email'][df['fraud_count'] > 0],
                name = 'Fraud Count'
            ))]

    autoencoder_anomalies = [go.Scatter(dict(
                x = df['tSNE_x'][df['anomaly_score_autoencoder'] > 0.2],
                y = df['tSNE_y'][df['anomaly_score_autoencoder'] > 0.2],
                mode = 'markers',
                # marker = {'size': utils.scaleValues(df['fraud_count'][df['fraud_count'] > 0])*10},
                text = df['email'][df['anomaly_score_autoencoder'] > 0.2],
                name = 'Anomaly by Autoencoder'
            ))]

    if_anomalies = [go.Scatter(dict(
                x = df['tSNE_x'][df['anomaly_isolation_forest'] == True],
                y = df['tSNE_y'][df['anomaly_isolation_forest'] == True],
                mode = 'markers',
                # marker = {'size': utils.scaleValues(df['fraud_count'][df['fraud_count'] > 0])*10},
                text = df['email'][df['anomaly_isolation_forest'] == True],
                name = 'Anomaly by Isolation Forest'
            ))]

    data = data + fraudsters + fraud_count + autoencoder_anomalies + if_anomalies

    layout = go.Layout(
        title='User Activity Map',
        hovermode='closest',
        legend=dict(orientation="h"),
        xaxis=dict(
            title='',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
            ticks='',
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            title='',
            ticklen=5,
            gridwidth=2,
            ticks='',
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            #scaleanchor='x',
            #scaleratio=0.5
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    return fig

def user_metric_plot(df, user_email):

    df = df.drop(['biggest_anomaly_metric_autoencoder']+utils.prefixColumns(['fraud','mean','median','std_'],df.columns), axis=1)
    
    user_df = df[df.email == user_email]
    distance = utils.distance(all_df=df, user_df=user_df)
    distance = distance[np.absolute(distance.distance) > 1] # only show outliers
    distance = distance[distance.metric.str.contains('_na') == False]
    distance = distance[distance.metric.str.contains('_notna') == False]
    distance = distance.tail(20)
    metric_names = distance.metric

    mins = []
    maxs = []
    means = []
    medians = []
    stds = []
    user_real_values = []
    user_values = []
    neg_25s = []
    pos_25s = []
    
    
    for metric in metric_names:
        
        std = np.std(df[metric].values)
        mean = np.nanmean(df[metric].values)
        stds.append(std)
        means.append(mean)
        
        neg_25 = (np.percentile(df[metric].values, 25)-mean)/std
        pos_25 = (np.percentile(df[metric].values, 75)-mean)/std
        neg_25s.append(neg_25)
        pos_25s.append(pos_25)
        
        median = (np.nanmedian(df[metric].values)-mean)/std
        medians.append(median)
        
        minz = (np.nanmin(df[metric].values)-mean)/std
        maxz = (np.nanmax(df[metric].values)-mean)/std
        mins.append(minz)
        maxs.append(maxz)
        
        user = user_df[metric].values[0]
        user_z_score = (user-mean)/std
        
        user_real_values.append(user)
        user_values.append(user_z_score)

    bar = pd.DataFrame({'metric': metric_names,
                        'min': mins,
                        'max': maxs,
                        'mean': means,
                        'median': medians,
                        'neg_25': neg_25s,
                        'pos_25': pos_25s,
                        'user_value': user_values,
                        'user_real_value': user_real_values})
    # print(bar)
    
    buffer = go.Bar(
                x=bar['min'],
                y=bar.metric,
                orientation = 'h',
                marker=dict(
                    color='rgb(255,255,255)',
                ),
                opacity=1.0,
                hoverinfo='none',

    )
    
    dist = go.Bar(
                x=bar['max']-bar['min'],
                y=bar.metric,
                orientation = 'h',
                marker=dict(
                    color='rgb(83,176,277)',
                ),
                opacity=1.0,
                text=bar.metric+': '+bar.user_real_value.astype('str'),
                hoverinfo='text',

    )        
        
    shp = []
    
    # user_values
    for i, metric in enumerate(metric_names):
        
        met = bar[bar.metric == metric]
        
        # user value
        shp.append({
            'type': 'line',
            'x0': met['user_value'].values[0],
            'y0': i-0.35,
            'x1': met['user_value'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(255, 0, 0, 1)',
                'width': 8
            },
        })
        
        
        
        # median
        shp.append({
            'type': 'line',
            'x0': met['median'].values[0],
            'y0': i-0.35,
            'x1': met['median'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'dash': 'dot',
                'width': 4
            },
        })
        
        
        # box
        shp.append({
            'type': 'rect',
            'x0': met['neg_25'].values[0],
            'y0': i-0.35,
            'x1': met['pos_25'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # lower line
        shp.append({
            'type': 'line',
            'x0': met['min'].values[0],
            'y0': i,
            'x1': met['neg_25'].values[0],
            'y1': i,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # upper line
        shp.append({
            'type': 'line',
            'x0': met['pos_25'].values[0],
            'y0': i,
            'x1': met['max'].values[0],
            'y1': i,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # min line
        shp.append({
            'type': 'line',
            'x0': met['min'].values[0],
            'y0': i-0.3,
            'x1': met['min'].values[0],
            'y1': i+0.3,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # max line
        shp.append({
            'type': 'line',
            'x0': met['max'].values[0],
            'y0': i-0.3,
            'x1': met['max'].values[0],
            'y1': i+0.3,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
   
        
    data = [buffer, dist]
    
    biggest = np.max(np.absolute(list(bar['min'])+list(bar['max'])))
    

    layout = go.Layout(
        title='Outlying User Behaviour',
        yaxis=dict(
            showticklabels=True,
            range=[0,20],
            tickfont=dict(
                size=10,
                )
            ),
        margin=go.Margin(
            l=350,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
        barmode='stack',
        xaxis=dict(
            range=[-biggest, biggest]
        ),
        showlegend=False,
        shapes = shp
    )

    fig = go.Figure(data=data, layout=layout)

    return fig




# def user_transitions(df, user_email):

def metric_figure(df, metric_name):

    # df = df[df.blocked == False]

    variable = metric_name

    df = df[['email',metric_name]]

    df = df.sort_values(by=metric_name, ascending=True)
    df = df.tail(20)

    data = [go.Bar(
                x=df[metric_name],
                y=df.email,
                orientation = 'h',
                text = df.email,
                textposition = 'outside',
                insidetextfont = dict(color='#FFFFFF'),
                outsidetextfont = dict(color='#000000'),
                constraintext = 'both',
                hoverinfo='none',
                marker = dict(color='rgb(83,176,277)')

    )]

    # annotations = []

    # for x, y in zip(df[metric_name], df.email):
    #     annotations += [
    #     dict(
    #         x=x,
    #         y=y,
    #         xref='x',
    #         yref='y',
    #         text=y,
    #         showarrow=False,
    #         arrowhead=7,
    #         ax=0,
    #         ay=0
    #     )]


    layout = go.Layout(
        title='Top 20 Users by {}'.format(variable),
        xaxis=dict(
            #type='log',
            autorange=True
        ),
        yaxis=dict(showticklabels=False),
        margin=go.Margin(
            # l=50,
            r=100,
            # b=100,
            # t=100,
            pad=4
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


def metric_dist(df, metric):

    # df = df[df.blocked == False]


    d = [df[metric].values]

    fig = ff.create_distplot(d, [metric], show_hist=False)

    fig['layout'].update(title='{} Distribution'.format(metric))

    return fig

def user_keymetric_plot(df, user_email):
    
    # df = df[df.blocked == False]

    user_df = df[df.email == user_email]

    # metric_names = ['cardholdername_unique',
    #                 'cardnumbers_unique',
    #                 'category_action_n_buy_request',
    #                 'category_action_n_buy_rejected',
    #                 'category_action_n_buy_fulfilled',
    #                 'category_label_action_n_customer-vault_verify-card_request',
    #                 'category_label_action_n_customer-vault_verify-card_completed', 
    #                 'category_label_action_n_customer-vault_verify-card_failed',
    #                 'value_mean']

    metric_names = ['unique_card_last_digits',
        'unique_card_name',
        'n_cl_customer-vault_verify-card',
        'n_cla_customer-vault_verify-card_failed',
        'n_eventCategory_buy',
        'n_eventCategory_interac',
        'n_ca_buy_rejected',
        'n_ca_buy_request',
        'n_ca_interac_request',
        ]

    mins = []
    maxs = []
    means = []
    medians = []
    stds = []
    user_values = []
    user_real_values = []
    neg_25s = []
    pos_25s = []
    
    
    for metric in metric_names:
        
        std = np.std(df[metric].values)
        mean = np.nanmean(df[metric].values)
        stds.append(std)
        means.append(mean)
        
        neg_25 = (np.percentile(df[metric].values, 25)-mean)/std
        pos_25 = (np.percentile(df[metric].values, 75)-mean)/std
        neg_25s.append(neg_25)
        pos_25s.append(pos_25)
        
        median = (np.nanmedian(df[metric].values)-mean)/std
        medians.append(median)
        
        minz = (np.nanmin(df[metric].values)-mean)/std
        maxz = (np.nanmax(df[metric].values)-mean)/std
        mins.append(minz)
        maxs.append(maxz)
        
        user = user_df[metric].values[0]
        user_z_score = (user-mean)/std
        
        user_real_values.append(user)
        user_values.append(user_z_score)

    bar = pd.DataFrame({'metric': metric_names,
                        'min': mins,
                        'max': maxs,
                        'mean': means,
                        'median': medians,
                        'neg_25': neg_25s,
                        'pos_25': pos_25s,
                        'user_value': user_values,
                        'user_real_value': user_real_values})
    # print(bar)
    
    buffer = go.Bar(
                x=bar['min'],
                y=bar.metric,
                orientation = 'h',
                marker=dict(
                    color='rgb(255,255,255)',
                ),
                opacity=1.0,
                hoverinfo='none',

    )
    
    dist = go.Bar(
                x=bar['max']-bar['min'],
                y=bar.metric,
                orientation = 'h',
                marker=dict(
                    color='rgb(83,176,277)',
                ),
                opacity=1.0,
                text=bar.metric+': '+bar.user_real_value.astype('str'),
                hoverinfo='text',

    )        
        
    shp = []
    
    # user_values
    for i, metric in enumerate(metric_names):
        
        met = bar[bar.metric == metric]
        
        # user value
        shp.append({
            'type': 'line',
            'x0': met['user_value'].values[0],
            'y0': i-0.35,
            'x1': met['user_value'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(255, 0, 0, 1)',
                'width': 8
            },
        })
        
        
        
        # median
        shp.append({
            'type': 'line',
            'x0': met['median'].values[0],
            'y0': i-0.35,
            'x1': met['median'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'dash': 'dot',
                'width': 4
            },
        })
        
        
        # box
        shp.append({
            'type': 'rect',
            'x0': met['neg_25'].values[0],
            'y0': i-0.35,
            'x1': met['pos_25'].values[0],
            'y1': i+0.35,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # lower line
        shp.append({
            'type': 'line',
            'x0': met['min'].values[0],
            'y0': i,
            'x1': met['neg_25'].values[0],
            'y1': i,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # upper line
        shp.append({
            'type': 'line',
            'x0': met['pos_25'].values[0],
            'y0': i,
            'x1': met['max'].values[0],
            'y1': i,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # min line
        shp.append({
            'type': 'line',
            'x0': met['min'].values[0],
            'y0': i-0.3,
            'x1': met['min'].values[0],
            'y1': i+0.3,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
        
        # max line
        shp.append({
            'type': 'line',
            'x0': met['max'].values[0],
            'y0': i-0.3,
            'x1': met['max'].values[0],
            'y1': i+0.3,
            'line': {
                'color': 'rgba(0, 0, 0, 1)',
                'width': 1
            },
        })
   
        
    data = [buffer, dist]
    
    biggest = np.max(np.absolute(list(bar['min'])+list(bar['max'])))
    

    layout = go.Layout(
        title='Key User Metrics',
        yaxis=dict(showticklabels=True),
        margin=go.Margin(
            l=350,
            r=100,
            b=100,
            t=100,
            pad=4
        ),
        barmode='stack',
        xaxis=dict(
            range=[-biggest, biggest]
        ),
        showlegend=False,
        shapes = shp
    )

    fig = go.Figure(data=data, layout=layout)

    return fig

class EventPlots(object):

    def __init__(self, creds):
        self._creds = creds
        self._conn_string = creds['connection_string']
        self._db = db.Database(self._creds)

    def init_connection(self, credentials):

        # load database credentials from file not in git repo
        self._creds = credentials

        # get the connection string
        self._conn_string = self._creds['connection_string']

        # initialize the database
        self._db = db.Database(self._creds)


    # New Plots for Live Dashboard
    def sessions_by_day(self, start_date=None, end_date=None):
        
        df = self._db.get_sessions_by_day(start_date=start_date, end_date=end_date)
        
        data = [go.Bar(x=df.day, y=df.n_sessions, name='User Sessions'), go.Scatter(x=df.day, y=df.n_sessions, name='User Sessions')]
        
        layout = go.Layout(title='User Sessions per Day')
        
        return go.Figure(data=data, layout=layout)


    def daily_new_users(self, start_date=None, end_date=None):
        
        new_users = self._db.get_new_users_by_day(start_date, end_date)
        
        data = [go.Bar(x=new_users.day, y=new_users.new_users, name='Daily New Users')]
        
        layout = go.Layout(title='New Users by Day')
        
        return go.Figure(data=data, layout=layout)     


    def most_active_users(self, start_date=None, end_date=None):
        
        df = self._db.get_activity_by_user(start_date=start_date, end_date=end_date)
        df = df.tail(20)
        
        data = [go.Bar(x=df.n_sessions, y=df.email, name='Number of Sessions', orientation='h')]
        
        layout = go.Layout(title='User Sessions', yaxis = {'automargin': True}, height=int(30*df.shape[0]))
        
        return go.Figure(data=data, layout=layout)


    def most_active_users_distribution(self, start_date=None, end_date=None):
        
        df = self._db.get_activity_by_user(start_date=start_date, end_date=end_date)
        
        minn = df.n_sessions.dropna().min()
        maxn = df.n_sessions.dropna().max()
        
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(df.n_sessions.values.reshape(-1,1))
        
        n_sessions = np.arange(minn,maxn, 0.5)
        
        kernel_density_estimates = np.exp(kde.score_samples(n_sessions.reshape(-1,1)))
        
        data = [go.Scatter(x=n_sessions, y=kernel_density_estimates, name='Number of Sessions')]
        
        layout = go.Layout(title='User Sessions')
        
        return go.Figure(data=data, layout=layout)


    def users_by_day(self, start_date=None, end_date=None):
        
        df = self._db.get_users_by_day(start_date=start_date, end_date=end_date)
        
        data = [go.Bar(x=df.day, y=df.n_unique_users, name='Unique User Emails'), go.Scatter(x=df.day, y=df.n_unique_users, name='Unique User Emails')]
        
        layout = go.Layout(title='Unique User Emails per Day')
        
        return go.Figure(data=data, layout=layout)


    def deposits_by_day(self, start_date=None, end_date=None):
        
        deposits = self._db.get_deposits_by_day(start_date, end_date)
        
        data = [go.Bar(x=deposits[deposits.deposit_type == dtype].day, 
                        y=deposits[deposits.deposit_type == dtype].value_usd, 
                        name=dtype) for dtype in deposits.deposit_type.unique()]
        
        layout = go.Layout(title='Deposit Value Per Day', barmode='stack')
        
        return go.Figure(data=data, layout=layout)


    def trades_by_day(self, start_date=None, end_date=None):
        
        trades = self._db.get_trades_by_day(start_date, end_date).round(4)
        
        data = [go.Scatter(x=trades[(trades.side == side) & (trades.type == ttype)].day, 
                        y=trades[(trades.side == side) & (trades.type == ttype)].estimated_value, 
                        name=side+' '+ttype) for side in trades.side.unique() for ttype in trades.type.unique()]
        
        layout = go.Layout(title='Trade Value Per Day')
        
        return go.Figure(data=data, layout=layout)


    def daily_trade_summary(self, start_date=None, end_date=None):
        
        # get the summary of the trades for that period
        trades = self._db.get_daily_trades_summary(start_date=start_date, end_date=end_date)

        # generate an overall summary
        overall = (trades[trades.trade_result == "Accepted"]
                    .groupby(['day','side','trade_type'])['estimated_value_sum']
                    .sum()
                    .reset_index())
        
        data = [go.Bar(x=overall[(overall.side == side) & (overall.trade_type == ttype)].day, 
                        y=overall[(overall.side == side) & (overall.trade_type == ttype)].estimated_value_sum, 
                        name=side+" "+ttype) for ttype in overall.trade_type.unique() for side in overall.side.unique()]

        layout = go.Layout(
            title='Trade Value by Day',
            barmode='stack'
        )

        return go.Figure(data=data, layout=layout)


    def detailed_trade_plots(self, start_date=None, end_date=None):
        
        # get the summary of the trades for that period
        trades = self._db.get_daily_trades_summary(start_date=start_date, end_date=end_date)

        figs = []
        
        for pair in sorted(trades.trading_pair.unique()):
            
            overall = trades[trades.trading_pair == pair]
        
            data = [go.Bar(x=overall[(overall.side == side) & (overall.trade_type == ttype)].day, 
                            y=overall[(overall.side == side) & (overall.trade_type == ttype)].cryptocurrency_amount_sum, 
                            name=side+" "+ttype) for ttype in overall.trade_type.unique() for side in overall.side.unique()]

            layout = go.Layout(
                title='{} Amount by Day'.format(pair),
                barmode='stack'
            )

            figs.append({'pair': pair, 'fig': go.Figure(data=data, layout=layout)})
            
        return figs

