from dash import dcc, html, Input, Output, dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from utils.data_loader import df, element_columns

# Layout for Pair Matrix page
pair_matrix_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("The Pair Matrix provides a comprehensive view of relationships between multiple geochemical elements:"),
            html.Ul([
                html.Li([html.Strong("Upper Triangle: ")," Scatter plots showing relationships between element pairs"]),
                html.Li([html.Strong("Diagonal: "), " Histograms with KDE showing distribution of each element"]),
                html.Li([html.Strong("Lower Triangle: "),  "KDE plots showing density relationships"]),
                html.Li([html.Strong("Correlation coefficients (r) "), "are displayed in the upper triangle"])
            ]),
            html.P("This visualization helps identify patterns, correlations, and potential outliers in your geochemical data.")
        ], className="explanation-text"), width=12),
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Pair Matrix Controls", className="mb-3"),
            html.Label("Sample Size (for performance):"),
            dcc.Dropdown(
                id='sample-size-dropdown',
                options=[
                    {'label': 'Full Dataset', 'value': 'full'},
                    {'label': '100 samples', 'value': 100},
                    {'label': '200 samples', 'value': 200}
                ],
                value=100,
                clearable=False
            ),
            html.Hr(),
            html.Label("Select Elements to Include:"),
            dcc.Dropdown(
                id='element-selector',
                options=[{'label': col, 'value': col} for col in element_columns],
                value=element_columns[:6].tolist() if len(element_columns) >= 6 else element_columns.tolist(),
                multi=True,
                placeholder="Select elements to include in pair matrix"
            ),
            html.Button('Generate Pair Matrix', id='generate-pair-matrix-btn', n_clicks=0,
                       className="btn-primary mt-3"),
            html.Div(id='pair-matrix-controls-placeholder')
        ], width=3, className="sidebar"),
        
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id='pair-matrix-image'), width=12, className="graph-container"),
            ]),
        ], width=9)
    ])
], fluid=True)

# Callbacks for Pair Matrix page
def pair_matrix_callbacks(app):
    @app.callback(
        Output('pair-matrix-image', 'children'),
        [Input('generate-pair-matrix-btn', 'n_clicks')],
        [dash.dependencies.State('sample-size-dropdown', 'value'),
         dash.dependencies.State('element-selector', 'value')]
    )
    def update_pair_matrix(n_clicks, sample_size, selected_elements):
        # ... (same Pair Matrix callback code from original)
        pass