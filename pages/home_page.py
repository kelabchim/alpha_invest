import dash
from dash import html, dcc, callback, Input, Output, State, get_asset_url

dash.register_page(__name__,path='/')

layout = html.Div(
        style={'display': 'flex', 'flexWrap': 'wrap'},
        children=[
            html.Div(
            children=[
                html.H1('Welcome to ALPHA INVEST', style={'color': 'white', 'fontSize': '3em', 'fontWeight': 'bold'}),
                html.P('Empower Your Investment Journey with Precision and Insight. Powered by Quantitative Methods such as Monte Carlo Simulations and LangChain technology.', style={'color': 'silver'}),
                dcc.Link('Get Started', href='/', className='start-button')
            ],
            style={
                'flex': '1',
                'minWidth': '50%',
                'background': 'linear-gradient(to right, #243b55, #141e30)',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center',
                'textAlign': 'center',
                'padding': '0 4rem'
            }
        ),
            
            html.Div(
                id='carousel',
                children=[
                    html.Button('❮', id='prev-button', n_clicks=0, style={
                        'position': 'absolute', 'top': '50%', 'left': '32px', 'zIndex': 1,
                        'background': 'none', 'border': 'none', 'color': '#fff', 'fontSize': '24px'}),
                    html.Img(id='carousel-image', style={'width': '100%', 'height': '800px','align':'center'}),
                    html.Button('❯', id='next-button', n_clicks=0, style={
                        'position': 'absolute', 'top': '50%', 'right': '32px', 'zIndex': 1,
                        'background': 'none', 'border': 'none', 'color': '#fff', 'fontSize': '24px'})
                ],
                style={'flex': '1', 'minWidth': '30%', 'maxWidth': '500px', 'margin': 'auto', 'position': 'relative'}
            ),
            dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=4)
        ]
    )

@callback(
    Output('carousel-image', 'src'),
    [Input('interval-component', 'n_intervals'),
     Input('prev-button', 'n_clicks'),
     Input('next-button', 'n_clicks')],
    [State('carousel-image', 'src')]
)
def update_carousel_image(n_intervals, prev_n_clicks, next_n_clicks, current_src):
    ctx = dash.callback_context
    images = ['page_1.png', 'page_2.png', 'page_3.png', 'page_4.png']
    current_index = images.index(current_src.split('/')[-1]) if current_src else -1
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'interval-component'

    if trigger_id in ['next-button', 'interval-component']:
        next_index = (current_index + 1) % len(images)
    elif trigger_id == 'prev-button':
        next_index = (current_index - 1) % len(images)
    else:
        next_index = current_index

    return get_asset_url(images[next_index])