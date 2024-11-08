from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import glob
import sys
import os
from dotenv import load_dotenv
import joblib
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from src.data_pipeline.utils import get_most_recent_folder
from src.data_pipeline.trading_strategy import calculate_pnl, plot_trades


load_dotenv()
results_dir = get_most_recent_folder(os.getenv("RESULTS_DIR"))

preds_df = pd.read_csv(os.path.join(results_dir, "model_preds_df.csv"))
preds_df.index.name = 'Date'
errors_df = pd.read_csv(os.path.join(results_dir, "errors_df.csv"))

trade_log = pd.read_csv(os.path.join(results_dir, 'trade_log.csv'), index_col=0)
close_prices = pd.read_csv(os.path.join(results_dir, 'close_prices.csv'), index_col=0)


### CREATE FIGURE FOR THE TRADING CHART OF THE OPTIMAL MODEL ###
# Create the main trading chart figure
fig = go.Figure()

# Add price line
fig.add_trace(
    go.Scatter(
        x=close_prices.index,
        y=close_prices['Adj Close'],
        name='Price',
        line=dict(color='blue', width=1),
        opacity=0.7
    )
)

# Filter trade positions
long_entries = trade_log[
    (trade_log['old_position'] == 0) & (trade_log['new_position'] == 1)
]
long_exits = trade_log[
    (trade_log['old_position'] == 1) & (trade_log['new_position'] == 0)
]
short_entries = trade_log[
    (trade_log['old_position'] == 0) & (trade_log['new_position'] == -1)
]
short_exits = trade_log[
    (trade_log['old_position'] == -1) & (trade_log['new_position'] == 0)
]

# Add long entries
fig.add_trace(
    go.Scatter(
        x=long_entries.index,
        y=close_prices.loc[close_prices.index.isin(long_entries.index), 'Adj Close'],
        mode='markers',
        name='Long Entry',
        marker=dict(
            symbol='triangle-up',
            size=15,
            color='green'
        )
    )
)

# Add long exits
fig.add_trace(
    go.Scatter(
        x=long_exits.index,
        y=close_prices.loc[close_prices.index.isin(long_exits.index), 'Adj Close'],
        mode='markers',
        name='Long Exit',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='red'
        )
    )
)

# Add short entries
fig.add_trace(
    go.Scatter(
        x=short_entries.index,
        y=close_prices.loc[close_prices.index.isin(short_entries.index), 'Adj Close'],
        mode='markers',
        name='Short Entry',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='purple'
        )
    )
)

# Add short exits
fig.add_trace(
    go.Scatter(
        x=short_exits.index,
        y=close_prices.loc[close_prices.index.isin(short_exits.index), 'Adj Close'],
        mode='markers',
        name='Short Exit',
        marker=dict(
            symbol='triangle-up',
            size=15,
            color='orange'
        )
    )
)

# Update layout
fig.update_layout(
    title='Trading Strategy Performance',
    xaxis_title='Date',
    yaxis_title='Price',
    template='plotly_white',
    hovermode='x unified',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)



# Calculate optimal model PnL results
results = calculate_pnl(close_prices, trade_log)
total_return = (results['cumulative_returns'].iloc[-1] - 1) * 100
n_trades = len(trade_log[trade_log['old_position'] != trade_log['new_position']])
win_rate = (results['strategy_returns'] > 0).mean() * 100

    

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = [
    html.H1('ML Forecast Trading Strategy'),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Model Type and Hyperparameters'),
                    dcc.Dropdown(
                        id='model-name',
                        options=[{'label': i, 'value': i} for i in preds_df.modelName.unique()],
                        value=preds_df.modelName.unique()[0]
                    ),
                    dcc.Dropdown(
                        id='model-params'
                    )
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Avg RMSE', className='card-title text-center'),
                    html.Div(id='avg-rmse', className='text-center')
                ], className='d-flex flex-column align-items-center justify-content-center')
            ], className='h-100')
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Avg MAPE', className='card-title text-center'),
                    html.Div(id='avg-mape', className='text-center')                    
                ], className='d-flex flex-column align-items-center justify-content-center')
            ], className='h-100')
        ], width=3)
    ]),

    html.Br(),
    dcc.Graph(id='preds-graph'),
    
    html.Br(),
    html.Hr(style={'borderTop': '2px solid #000000'}),
    html.Br(),

    html.H2('Optimal Model Trading Performance'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Total Return (%)', className='card-title text-center'),
                    html.Div(id='total-return', children=[round(total_return, 2)], className='text-center')
                ], className='d-flex flex-column align-items-center justify-content-center')
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Number of Trades', className='card-title text-center'),
                    html.Div(id='num-trades', children=[n_trades], className='text-center')
                ], className='d-flex flex-column align-items-center justify-content-center')
            ], className='h-100')
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Win Rate (%)', className='card-title text-center'),
                    html.Div(id='win-rate', children=[round(win_rate, 2)], className='text-center')                    
                ], className='d-flex flex-column align-items-center justify-content-center')
            ], className='h-100')
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Avg Return/Trade (%)', className='card-title text-center'),
                    html.Div(id='avg-trade-return', children=[round(total_return/n_trades, 2)], className='text-center')                    
                ], className='d-flex flex-column align-items-center justify-content-center')
            ], className='h-100')
        ], width=3)
    ]),
    dcc.Graph(id='trade-chart', figure=fig)
]


@callback(
    Output('model-params', 'options'),
    Output('model-params', 'value'),
    Input('model-name', 'value')
)
def get_model_params(model_name):
    model_params = preds_df[preds_df.modelName==model_name].modelParams.unique()

    return [{'label': i, 'value': i} for i in model_params], model_params[0]

@callback(
    Output('preds-graph', 'figure'),
    Input('model-name', 'value'),
    Input('model-params', 'value')
)
def plot_model_charts(model_name, model_params):
    ts_df = preds_df[(preds_df.modelName==model_name) & (preds_df.modelParams==model_params)]
    ts_df = ts_df[['yTrue', 'yPred']]
    
    fig = go.Figure()

    # Add the first time series
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df['yTrue'],
            name='yTrue',
            mode='lines',
            line=dict(color='#1f77b4', width=2)
        )
    )

    # Add the second time series
    fig.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df['yPred'],
            name='yPred',
            mode='lines',
            line=dict(color='#ff7f0e', width=2)
        )
    )

    # Update the layout
    fig.update_layout(
        title='1-day Ahead Forecast - Preds vs True',
        xaxis_title='Date',
        yaxis_title='Returns',
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


@callback(
    Output('avg-rmse', 'children'),
    Output('avg-mape', 'children'),
    Input('model-name', 'value'),
    Input('model-params', 'value')
)
def get_avg_errors(model_name, model_params):
    model_errors = errors_df[(errors_df.modelName==model_name) & (errors_df.modelParams==model_params)]
    rmse = model_errors[model_errors.ErrorType=='RMSE']
    mape = model_errors[model_errors.ErrorType=='MAPE']

    return round(rmse.ErrorVal.mean(), 3), round(mape.ErrorVal.mean(), 3)










if __name__=="__main__":
    app.run(debug=True)