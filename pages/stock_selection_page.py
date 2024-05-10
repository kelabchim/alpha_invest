from dash import html

from dash import dcc, html
from dash import html, dcc, callback, Input, Output, State ,MATCH,ALL,no_update
import dash_bootstrap_components as dbc
import json
from utils.api_calls import *
import dash
from utils.api_calls import *
from dash.exceptions import PreventUpdate
from utils.PandasDataFrameAgent import agent
from dash.exceptions import PreventUpdate
from dash.exceptions import PreventUpdate
from dash import no_update

from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import no_update

dash.register_page(__name__,path='/stock-selection')

all_stock_df = fetch_data_from_api('http://0.0.0.0:8080/stocks/')


sectors = ['Healthcare', 'Consumer Discretionary', 'Industrials', 'Utilities', 'Real Estate', 'Communication Services', 'Basic Materials', 'Energy', 'Consumer Staples', 'Technology', 'Financials']
industries = ['Beverages—Non-Alcoholic', 'Lumber & Wood Production', 'Insurance Brokers', 'Insurance—Life', 'Packaged Foods', 'Capital Markets', 'Medical Distribution', 'Leisure', 'Department Stores', 'Household & Personal Products', 'Trucking', 'Oil & Gas Midstream', 'Industrial Distribution', 'Insurance—Specialty', 'Apparel Retail', 'Scientific & Technical Instruments', 'Banks—Diversified', 'Oil & Gas Drilling', 'Advertising Agencies', 'Restaurants', 'Biotechnology', 'Farm Products', 'Entertainment', 'Computer Hardware', 'Internet Content & Information', 'Resorts & Casinos', 'Medical Devices', 'Grocery Stores', 'Footwear & Accessories', 'Luxury Goods', 'Gold', 'Telecom Services', 'Consulting Services', 'Home Improvement Retail', 'Pharmaceutical Retailers', 'Railroads', 'Real Estate Services', 'Beverages—Brewers', 'Specialty Business Services', 'Electronics & Computer Distribution', 'Oil & Gas Equipment & Services', 'Beverages—Wineries & Distilleries', 'REIT—Office', 'Packaging & Containers', 'Healthcare Plans', 'Confectioners', 'Solar', 'Drug Manufacturers—Specialty & Generic', 'Farm & Heavy Construction Machinery', 'Credit Services', 'Metal Fabrication', 'Electrical Equipment & Parts', 'Auto Parts', 'Software—Application', 'Specialty Industrial Machinery', 'Specialty Retail', 'Discount Stores', 'Tools & Accessories', 'Travel Services', 'Utilities—Regulated Electric', 'Furnishings, Fixtures & Appliances', 'Marine Shipping', 'Electronic Components', 'REIT—Diversified', 'Publishing', 'Personal Services', 'Chemicals', 'Health Information Services', 'Agricultural Inputs', 'Mortgage Finance', 'Utilities—Regulated Gas', 'Auto Manufacturers', 'Steel', 'Oil & Gas Refining & Marketing', 'Integrated Freight & Logistics', 'Semiconductors', 'REIT—Hotel & Motel', 'Specialty Chemicals', 'Diagnostics & Research', 'Lodging', 'Recreational Vehicles', 'Financial Conglomerates', 'Semiconductor Equipment & Materials', 'REIT—Specialty', 'Airlines', 'Business Equipment & Supplies', 'Internet Retail', 'Building Materials', 'Utilities—Independent Power Producers', 'Insurance—Property & Casualty', 'Utilities—Diversified', 'Communication Equipment', 'Gambling', 'Consumer Electronics', 'Medical Instruments & Supplies', 'Residential Construction', 'Financial Data & Stock Exchanges', 'Tobacco', 'Apparel Manufacturing', 'Electronic Gaming & Multimedia', 'Banks—Regional', 'Utilities—Renewable', 'Engineering & Construction', 'Software—Infrastructure', 'REIT—Mortgage', 'Medical Care Facilities', 'Waste Management', 'REIT—Healthcare Facilities', 'Broadcasting', 'REIT—Retail', 'Rental & Leasing Services', 'REIT—Industrial', 'Copper', 'Staffing & Employment Services', 'Oil & Gas E&P', 'Asset Management', 'Aerospace & Defense', 'Oil & Gas Integrated', 'Insurance—Diversified', 'Auto & Truck Dealerships', 'REIT—Residential', 'Aluminum', 'Security & Protection Services', 'Insurance—Reinsurance', 'Utilities—Regulated Water', 'Information Technology Services', 'Building Products & Equipment', 'Food Distribution', 'Drug Manufacturers—General']

search_bar = html.Div([
    html.Div(['Ticker:'], className='search-prefix'),
    dcc.Input(id='search-input', type='text', placeholder='Enter stock ticker...', className='search-input'),
    html.Div([
        html.Button('Search', id='submit-filters', n_clicks=0, className='submit-button')
    ], className='submit-button-container'),
], className='search-bar')

filters_layout = html.Div([
    html.Div([
        html.Label('Industry', className='filter-label'),
        dcc.Dropdown(
            id='industry-filter',
            multi=True,
            options=[{'label': industry, 'value': industry} for industry in industries],
            placeholder='All Industries',
            className='dropdown-filter'
        ),
    ], className='filter-container'),
    html.Div([
        html.Label('Sector', className='filter-label'),
        dcc.Dropdown(
            id='sector-filter',
            multi=True,
            options=[{'label': sector, 'value': sector} for sector in sectors],
            placeholder='All Sectors',
            className='dropdown-filter'
        ),
    ], className='filter-container'),
    html.Div([
        html.Label('Market Cap(USD)', className='filter-label'),
        dcc.Dropdown(
            id='price-range-filter',
            options=[
                {'label': 'Under $100', 'value': '0-100'},
                {'label': '$100 to $500', 'value': '100-500'},
                {'label': 'Over $500', 'value': '500-10000'},
            ],
            placeholder='Any Price Range',
            className='dropdown-filter'
        ),
    ], className='filter-container'),
    html.Div([
        html.Label('Search Query', className='filter-label'),
        dcc.Input(
            id='llm-query-input',
            type='text',
            placeholder='e.g., Give me top 10 sharpe stocks technology stocks in past 30 days',
            className='llm-query-input'
        ),
    ], className='filter-container'),
    #Add more filters here, each wrapped in a html.Div with a label
], className='filters-layout')

modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ml-auto",n_clicks=0)
        ),
    ],
    id="modal",
    is_open=False,
    size="lg",
)

#main
layout = html.Div([
    search_bar,
    filters_layout,
    dcc.Loading(
        id="loading-1",
        type="default",  #You can choose from 'graph', 'cube', 'circle', 'dot', or 'default'
        children=html.Div(id='stock-cards-container'),
    ),
    #Add this near the bottom of your layout
    html.Div(id='callback-output'),
    modal,
], className='stock-selection-page')


def process_stock_data(stock_data):
    """Process stock data to prepare content for the modal."""
    data = stock_data['data']
    yfinance_data = data['full_yfinance_data']

    # Prepare content for the modal
    content = [
        html.H2(data['company_name'], className="modal-title"),
        html.H4(f"Ticker: {data['ticker']} - Price: ${data['live_price']}", className="modal-subtitle"),
        html.Hr(),
        html.P(f"Industry: {yfinance_data['industry']}", className="modal-text"),
        html.P(f"Sector: {yfinance_data['sector']}", className="modal-text"),
        html.P(f"Market Cap: {yfinance_data['marketCap']}", className="modal-text"),
        html.P(f"Total Employees: {yfinance_data['fullTimeEmployees']}", className="modal-text"),
        html.P(f"PE Ratio (Trailing): {yfinance_data.get('trailingPE', 'N/A')}", className="modal-text"),
        html.P(f"PE Ratio (Forward): {yfinance_data.get('forwardPE', 'N/A')}", className="modal-text"),
        html.P(f"One Week Returns(%): {data['stock_performance']['one_week_change_percent']}", className="modal-text"),
        html.P(f"One Month Volatility of Returns: {data['stock_performance']['one_month_volatility']}", className="modal-text"),
        html.P("Company Summary:", className="modal-text"),
        html.P(yfinance_data['longBusinessSummary'], className="modal-text"),
        html.Hr(),
        html.H4("Recent News", className="modal-subtitle"),
        html.Ul([html.Li(html.A(news_item['title'], href="#", target="_blank")) for news_item in data['company_news']], className="modal-news-list"),
    ]

    return content



@callback(
    [Output("modal", "is_open"),
     Output("modal-title", "children"),
     Output("modal-body", "children")],
    [Input({'type': 'more-info', 'index': ALL}, 'n_clicks'),
     Input("close-modal", "n_clicks")],
    [State("modal", "is_open"),
     State({'type': 'more-info', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def toggle_modal(n_clicks_info, n_clicks_close, is_open, ids):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]["prop_id"]
    if n_clicks_close >0 and 'close-modal' in button_id and is_open:
        return False, no_update, no_update  # Close the modal
    button_action = json.loads(button_id.split('.')[0])
    

    if button_action and button_action['type'] == 'more-info' and any(n_clicks_info):
        stock_index = button_action['index']
        stock_data = fetch_stock_data(stock_index)  # Fetch data based on the stock index
        print(stock_data)
        if stock_data:
            modal_content = process_stock_data(stock_data)
            return True, "Stock Information", modal_content  # Open the modal with data

    raise PreventUpdate


def create_stock_card(stock):
    return html.Div([
        html.Div([
            html.H5(stock['Stock'], className='card-title'),
            html.P(stock['Industry'], className='card-text'),
            html.P(f"Close Price: {round(stock['Close'],4)}", className='card-text'),
        ], className='card-body'),
         html.Button('More Info', id={'type': 'more-info', 'index': stock['Stock']},
                    className='btn btn-info', n_clicks=0),
        html.Button('Add to Cart', id={'type': 'add-to-cart-button', 'index': stock['Stock']}, className='btn btn-primary')
    ], className='stock-card card', style={
        'margin': '10px', 'padding': '20px', 'border': '1px solid #ddd',
        'borderRadius': '5px', 'width': '300px', 'display': 'inline-block',
        'boxShadow': '2px 2px 2px lightgrey'
    })


@callback(
    Output('stock-cards-container', 'children'),
    [Input('submit-filters', 'n_clicks')],
    [State('search-input', 'value'),
     State('sector-filter', 'value'),  #This can be a list
     State('industry-filter', 'value'),
     State('llm-query-input','value')]  #This can be a list
)
def update_stock_cards(n_clicks, ticker_input, sector_filter, industry_filter,llm_query_input):
    if n_clicks > 0:
        filtered_df = all_stock_df.copy()  # Assuming all_stock_df is your DataFrame
        if ticker_input:
            filtered_df = filtered_df[filtered_df['Stock'].str.contains(ticker_input, case=False, na=False)]
        if sector_filter:
            filtered_df = filtered_df[filtered_df['Sector'].isin(sector_filter)]
        if industry_filter:
            filtered_df = filtered_df[filtered_df['Industry'].isin(industry_filter)]
        if llm_query_input:
            output = agent.run(llm_query_input)
            filtered_df = filtered_df[filtered_df['Stock'].isin(output)]
        if filtered_df.empty:
            return [html.Div('No stocks match your search and filter criteria. Please adjust your filters and try again.', className='no-results-message')]
        stock_cards = [create_stock_card(stock) for index, stock in filtered_df.iterrows()]
        return stock_cards
    return []


@callback(
    [Output('cart', 'data'),
     Output('cart-count', 'children')],
    [Input({'type': 'add-to-cart-button', 'index': ALL}, 'n_clicks'),
     Input({'type': 'remove-stock-button', 'index': ALL}, 'n_clicks')],
    [State('cart', 'data')]
)
def update_cart(n_clicks,remove_clicks,cart_data:list):
    ctx = dash.callback_context

    if ctx.triggered[0]['value']==None:
        raise PreventUpdate

    if not ctx.triggered:
        return cart_data, str(len(cart_data)) if cart_data else '0'

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    stock_index = json.loads(button_id)['index']
    
    button_action, stock_index = json.loads(button_id.split('.')[0])['type'], json.loads(button_id.split('.')[0])['index']
    
    # print("ADDING",cart_data,button_action)
    # print("REMOVING", remove_clicks) 
    if cart_data is None:
        cart_data = []  # Initializes cart_data if it's None
    
    if button_action == 'add-to-cart-button' and stock_index not in cart_data:
        cart_data.append(stock_index)
    elif button_action == 'remove-stock-button' and stock_index in cart_data and any(remove_clicks)>0 and cart_data[remove_clicks.index(max(remove_clicks))]==stock_index:
        cart_data = [stock for stock in cart_data if stock != stock_index]

    cart_count = str(len(cart_data))

    return cart_data, cart_count








