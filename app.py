import dash
from dash import dcc, html
import json

external_stylesheets = ['https://use.fontawesome.com/releases/v5.8.1/css/all.css']

app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets,use_pages=True)

app.layout = html.Div(
    [
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='cart', storage_type='session',data=[]),
    html.Div(
        children=[
            html.Img(src=app.get_asset_url('brand.png'), style={'height': '100px','align':'center'}),
            dcc.Link('Stock Selection', href='/stock-selection', className='nav-link'),
            dcc.Link(
    html.Div([
        html.I(className="fa fa-shopping-cart", style={'color': 'white'}),  # Make icon white
        html.Span('0', id='cart-count', className='cart-count', style={'color': 'white'}),  # Make count white
    ], className='cart-icon', style={
        'display': 'inline-flex', 
        'alignItems': 'center', 
        'backgroundColor': '#000',
        'padding': '5px', 
        'borderRadius': '5px'
    }),
    href='/cart-page',
    style={'textDecoration': 'none'}  # Removes underline from link
),
        ],
        style={
            'backgroundColor': '#111', 
            'color': '#ddd', 
            'alignItems': 'center'
        },
        
        className='header'
    ),
    
    # html.Div(id='page-content'),
    dash.page_container,

])

if __name__ == '__main__':
    app.run_server(debug=False)
