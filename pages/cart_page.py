import dash
from dash import html, dcc, callback, Input, Output

from dash.dependencies import Input, Output, State, MATCH, ALL

from dash import dcc
import plotly.graph_objs as go

import pandas as pd
import datetime
import yfinance as yf

dash.register_page(__name__,path='/cart-page')

layout = html.Div([
    html.H4('Your Selected Stocks', style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#333'}),
    dcc.Loading(
        id="loading-stock-returns",
        children=[html.Div(id='stock-returns-graph')],
        type="circle",
    ),
    html.Div(id='selected-stocks-list', className='cart-items'),
    dcc.Link('Confirm and Proceed to Optimization', href='/portfolio-optimization', className='nav-link'),
], className='cart-page', style={'maxWidth': '1200px', 'margin': '0 auto'})


@callback(
    Output('selected-stocks-list', 'children'),
    Input('cart', 'data')
)
def generate_cart_items(cart_data):
    if not cart_data:
        return html.Div("Your cart is empty", style={'textAlign': 'center', 'color': '#888', 'margin': '20px'})
    return [
        html.Div([
            html.Span(stock, className='stock-name'),
            html.Button('Remove', id={'type': 'remove-stock-button', 'index': stock}, n_clicks=0, className='remove-stock-button')
        ], className='cart-item', style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}) for stock in cart_data
    ]
    
    
def fetch_and_compute_returns(stock_tickers):
    """
    Fetches historical data for the given stock tickers using a single call to yf.download(),
    and computes the yearly returns.

    Parameters:
    - stock_tickers: A list of stock ticker symbols as strings.

    Returns:
    A pandas DataFrame containing the returns of the stocks.
    """
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365)
    stock_data = yf.download(stock_tickers, start=start_date, end=end_date)
    
    if len(stock_tickers)<=1:
        return None

    if isinstance(stock_data.columns, pd.MultiIndex) and 'Adj Close' in stock_data.columns.levels[0]:
        adj_close_data = stock_data['Adj Close']
    else:
        adj_close_data = stock_data

    daily_returns = adj_close_data.pct_change()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    cumulative_returns = cumulative_returns.fillna(method='ffill')
    return cumulative_returns

@callback(
    Output('stock-returns-graph', 'children'),
    Input('cart', 'data')
)
def update_graph(cart_data):
    if not cart_data:
        return dcc.Graph(figure=go.Figure(layout={'template': 'plotly_dark'}).update_layout(
            title="No stocks selected",
            xaxis={'visible': False, 'showgrid': False},
            yaxis={'visible': False, 'showgrid': False}
        ))

    returns_df = fetch_and_compute_returns(cart_data)
    
    try:
        traces = []
        for stock in cart_data:
            traces.append(go.Scatter(
                x=returns_df.index,
                y=returns_df[stock],
                mode='lines+markers',
                name=stock
            ))

        layout = go.Layout(
            title='1-Year Stock Returns',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Cumulative Return'},
            hovermode='closest',
            template='plotly_dark',
            margin={'l': 60, 'r': 10, 't': 45, 'b': 60},
            legend={'orientation': 'h', 'yanchor': 'bottom', 'xanchor': 'center', 'y': 1, 'x': 0.5},
            font=dict(size=12, color='white')
        )
        
        figure = go.Figure(data=traces, layout=layout)
        return dcc.Graph(figure=figure)
    except:
        return dcc.Graph(figure=None)