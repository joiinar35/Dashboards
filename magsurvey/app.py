import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc

# Initialize the Dash 

# app = dash.Dash(
#     __name__,
#     external_stylesheets=[
#         dbc.themes.BOOTSTRAP,
#         "assets/style.css"  # Custom CSS
#     ],
#     # external_scripts=[
#     #     'https://cdn.plot.ly/plotly-2.24.1.min.js'
#     # ],
#     suppress_callback_exceptions=True,
#     title="Magnetic Survey Analysis")

app = dash.Dash(__name__, 
                 external_stylesheets=[dbc.themes.BOOTSTRAP,
                "assets/style.css"],  # Custom CSS
                 suppress_callback_exceptions=True,
                 title="Magnetometry Analysis")



# Add this to your app layout or in a separate CSS file
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Magnetic Survey Analysis", 
                   className="text-center mb-3",
                   style={'color': '#2c3e50'}),
            html.Hr(),
            dcc.Tabs(
                id="left-tabs",
                value='surveyed-data',
                vertical=True,
                children=[
                    dcc.Tab(
                        label='Survey Data',
                        value='surveyed-data',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        style={'width': '100%'}  # Individual tab width
                    ),
                    dcc.Tab(
                        label='Magnetic Gradients',
                        value='magnetic-gradients', 
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        style={'width': '100%'}  # Individual tab width
                    ),
                    dcc.Tab(
                        label='Reduction to the Pole',
                        value='reduction-to-pole',
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                        style={'width': '100%'}  # Individual tab width
                    ),
                ],
                style={
                    'height': '100%', 
                   # 'borderRight': '1px solid #dee2e6',
                    'width': '100%'
                }
            ),
        ], width=3, className="bg-light p-4", style={'minHeight': '100vh'}),

        dbc.Col([
            html.Div(id='tab-content', className="p-4")
        ], width=9),
    ])
], fluid=True)

# Import page layouts
from pages.surveyed_data import layout as surveyed_data_layout
from pages.magnetic_gradients import layout as magnetic_gradients_layout
from pages.reduction_to_pole import layout as reduction_to_pole_layout

@callback(
    Output('tab-content', 'children'),
    Input('left-tabs', 'value')
)
def render_content(tab):
    if tab == 'surveyed-data':
        return surveyed_data_layout
    elif tab == 'magnetic-gradients':
        return magnetic_gradients_layout
    elif tab == 'reduction-to-pole':
        return reduction_to_pole_layout
    else:
        return html.Div("Select a tab")

if __name__ == '__main__':
    app.run(debug=True)