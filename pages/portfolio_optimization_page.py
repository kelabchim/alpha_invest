import datetime
import dash
from dash import html, dcc, callback, Input, Output, State,ALL
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
from dash.exceptions import PreventUpdate

import plotly.express as px
import plotly.graph_objects as go

from utils.PortfolioOptimizer import PortfolioOptimizer
from utils.PortfolioVaRSimulator import PortfolioVaRSimulator

dash.register_page(__name__,path='/portfolio-optimization')

layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H4('Portfolio Optimization Selected Stocks', style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div(id='optimization-selected-stocks-list', className='cart-items'),
    html.Div([
        html.Label('Number of Simulations:', className='input-label'),
        dcc.Input(id='num-simulations', type='number', value=10000, className='input-field'),
        html.Label('Look Back Period (in terms of days):', className='input-label'),
        dcc.Input(id='look-back', type='number', value=120, className='input-field'),
    ], className='input-container'),
    html.Button('Optimize with Monte Carlo', id='optimize-portfolio-button', className='nav-link', n_clicks=0),
    html.Div(id='graph-container', style={'display': 'none'}, children=[
    dcc.Loading(
        id="loading-optimization", 
        children=[dcc.Graph(
            id='risk-return-graph',
            config={'displayModeBar': False}  # Adjusted as per previous suggestion
        )], 
        type="circle"
    )
    ]),
    html.Div(id='table-container', style={'display': 'none'}, children=[
    dash_table.DataTable(
        id='data-table', 
        columns=[], 
        data=[], 
        style_table={'overflowX': 'auto'}, 
        filter_action='native', 
        sort_action='native', 
        row_selectable='single', 
        page_action='native', 
        page_current=0, 
        page_size=10, 
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'}, 
        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
    ),
    html.Button('View Selected Portfolio', id='submit-portfolio-button', className='nav-link', n_clicks=0),
    dcc.Store(id='portfolio-weights-store'),
    html.Div(id='portfolio-weights-container'),
     html.Div(id='portfolio-plots-container', children=[
        dcc.Graph(id='portfolio-value-paths-graph', config={'displayModeBar': False}),
        dcc.Graph(id='portfolio-returns-histogram', config={'displayModeBar': False})
    ])
    ])
    ], className='optimization-page')

@callback(
    Output('optimization-selected-stocks-list', 'children'),
    Input('cart', 'data')
)
def generate_cart_items_optimization(cart_data):
    if not cart_data:
        return html.Div("Your cart is empty", style={'textAlign': 'center', 'color': '#888', 'margin': '20px'})
    return [
        html.Div([
            html.Span(stock, className='stock-name'),
            html.Div([
                html.Div('Minimum Weight Constraint:'),
                dcc.Input(
                    id={
                        'type': 'weight-input', 
                        'index': stock
                    }, 
                    type='number', 
                    value=0, 
                    className='weight-input'
                )
            ], style={'marginBottom': '10px'}),
        ], className='cart-item', style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '10px', 'borderBottom': '1px solid #ddd'}) for stock in cart_data
    ]


@callback(
    [Output('risk-return-graph', 'figure'), 
     Output('data-table', 'columns'), 
     Output('data-table', 'data'),
     Output('table-container', 'style'),  # Existing output for the table
     Output('graph-container', 'style'),  # Add this line for the graph
     Output('portfolio-weights-store', 'data')], # Add this line to store weights
    [Input('optimize-portfolio-button', 'n_clicks'), 
     Input('data-table', 'selected_rows')],
    [State('num-simulations', 'value'), 
     State('look-back', 'value'), 
     State('cart', 'data'),
     State('data-table', 'data'),
     State({'type': 'weight-input', 'index': ALL}, 'value')
     ]
)
def update_content(n_clicks, selected_rows, num_simulations, lookback, cart_data, table_data,min_weights):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'optimize-portfolio-button':
        if not cart_data:
            return px.scatter(title="No stocks selected"), [], []

        tickers = [stock for stock in cart_data]  # Adjust according to your cart_data structure
        min_weight_constraints = min_weights
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=lookback)
        simulator = PortfolioOptimizer(tickers, start_date, end_date)
        stats_uniform, weights_uniform = simulator.simulate_portfolios(num_simulations=num_simulations, method='uniform',min_weight_constraints=min_weight_constraints)
        fig = px.scatter( stats_uniform, x='risk_measure', y='portfolio_return', hover_data=stats_uniform.columns)
        fig.update_layout(plot_bgcolor='rgb(35, 35, 35)', paper_bgcolor='rgb(35, 35, 35)', font_color='white')
        columns = [{"name": i, "id": i} for i in  stats_uniform.columns]
        data =  stats_uniform.to_dict('records')

        return fig, columns, data, {'display': 'block'}, {'display': 'block'}, weights_uniform.to_dict('records')   # Make both graph and table visible

    elif trigger_id == 'data-table':
        if not selected_rows or not table_data:
            raise PreventUpdate

        df = pd.DataFrame(table_data)
        selected_df = df.iloc[selected_rows]

        fig = px.scatter(df, x='risk_measure', y='portfolio_return', hover_data=df.columns)
        fig.update_traces(marker=dict(size=6, opacity=0.5))  # Adjust other points

        fig.add_trace(go.Scatter(
            x=selected_df['risk_measure'], y=selected_df['portfolio_return'],
            mode='markers',
            marker=dict(
                color='Magenta',  # Highly visible color
                size=20,  # Larger size
                line=dict(
                    color='Yellow',  # Contrasting border color
                    width=3  # Wider border for halo effect
                )
            ),
            name='Selected'
        ))

        fig.update_layout(
            plot_bgcolor='rgb(35, 35, 35)',
            paper_bgcolor='rgb(35, 35, 35)',
            font_color='white'
        )
        columns = [{"name": i, "id": i} for i in df.columns]
        data = df.to_dict('records')

        return fig, columns, data, dash.no_update, dash.no_update , dash.no_update # Do not change visibility here

    raise PreventUpdate


import plotly.graph_objects as go

def plot_portfolio_results(portfolio_value_paths, portfolio_returns, VaR, CVaR):
    value_paths_fig = go.Figure()

    for path in portfolio_value_paths:
        value_paths_fig.add_trace(go.Scatter(y=path, mode='lines', line=dict(width=1), opacity=0.4))

    value_paths_fig.update_layout(
        title='Portfolio Value Paths',
        xaxis_title='Time (Days)',
        yaxis_title='Portfolio Value',
        template='plotly_dark'
    )

    # Prepare the Histogram of Portfolio Returns plot
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Histogram(x=portfolio_returns, nbinsx=50, marker_color='blue', opacity=0.7))

    # Adding VaR line and annotation
    returns_fig.add_shape(type='line', x0=VaR, y0=0, x1=VaR, y1=1, xref='x', yref='paper', line=dict(color='red', dash='dash'))
    returns_fig.add_annotation(x=VaR, y=0.95, xref='x', yref='paper', text='VaR', showarrow=True, arrowhead=1, ax=-40, ay=-40)

    # Optionally, add CVaR annotation if needed
    # returns_fig.add_annotation(x=CVaR, y=0.95, xref='x', yref='paper', text='CVaR', showarrow=True, arrowhead=1, ax=-40, ay=-30)

    returns_fig.update_layout(
        title='Histogram of Portfolio Returns',
        xaxis_title='Returns',
        yaxis_title='Frequency',
        template='plotly_dark'
    )

    return value_paths_fig, returns_fig

@callback(
    [Output('portfolio-weights-container', 'children'),
     Output('portfolio-value-paths-graph', 'figure'),
     Output('portfolio-returns-histogram', 'figure')],
    [Input('submit-portfolio-button', 'n_clicks')],
    [State('data-table', 'selected_rows'),
     State('portfolio-weights-store', 'data')]
)
def display_selected_portfolio_weights(n_clicks,selected_rows,weights_data):
    if n_clicks == 0 or not selected_rows or not weights_data:
        raise PreventUpdate

    selected_portfolio_index = selected_rows[0]
    selected_weights = weights_data[selected_portfolio_index]  # Assuming direct indexing
    
    weights_display = html.Ul([
        html.Li(f"{stock}: {weight:.2%}") for stock, weight in selected_weights.items()
    ])

    simulator  = PortfolioVaRSimulator(selected_weights,datetime.date.today(),n_sim=1000)
    
    portfolio_returns, portfolio_value_paths, VaR,CVaR = simulator.simulate_returns()
    
    value_paths_fig, returns_fig = plot_portfolio_results(portfolio_value_paths, portfolio_returns, VaR, CVaR)
    
    print(weights_display)

    return weights_display, value_paths_fig, returns_fig

