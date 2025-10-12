from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.data_loader import data_for_analysis, df
from factor_analyzer import FactorAnalyzer
import numpy as np
from scipy.interpolate import griddata

# Calculate initial number of factors for dropdown
if not data_for_analysis.empty:
    fa_initial = FactorAnalyzer(rotation=None, n_factors=data_for_analysis.shape[1])
    fa_initial.fit(data_for_analysis)
    eigenvalues_fa_initial, _ = fa_initial.get_eigenvalues()
    n_factors_initial = sum(eigenvalues_fa_initial > 1) if eigenvalues_fa_initial is not None else 1
else:
    n_factors_initial = 1

# Layout for Factor Analysis page
factor_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("Factor Analysis helps identify underlying factors that explain the relationships between variables."),
            html.P("The Factor Analysis Loadings Heatmap shows the relationships between your original geochemical variables and the underlying factors:"),
            html.Ul([
                html.Li([html.Strong("Rows: "), "Original geochemical variables (e.g., Ba, Co)."]),
                html.Li([html.Strong("Columns: "),  "Factors identified by the analysis (Factor 1, Factor 2, etc.)."]),
                html.Li([html.Strong("Colors and Values: "), "Indicate the 'loading' of a variable on a factor."]),
                html.Ul([
                    html.Li("High positive loading (warm colors, closer to 1): Variable is strongly and positively associated with the factor."),
                    html.Li("High negative loading (cool colors, closer to -1): Variable is strongly and negatively associated with the factor."),
                    html.Li("Loading near zero: Variable has little influence on that factor.")
                ])
            ]),
            html.P("By interpreting variables with high absolute loadings on each factor, you can understand the geological or geochemical processes represented by each factor.")
        ], className="explanation-text"), width=12),
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Factor Analysis Controls", className="mb-3"),
            html.Label("Number of Factors:"),
            dcc.Dropdown(
                id='n-factors-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(1, len(data_for_analysis.columns) + 1)],
                value=n_factors_initial,
                clearable=False
            ),
            html.Div(id='fa-controls-placeholder')
        ], width=3, className="sidebar"),
        
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='fa-scree-plot'), width=6, className="graph-container"),
                dbc.Col(dcc.Graph(id='fa-variance-plot'), width=6, className="graph-container"),
            ]),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='fa-loadings-heatmap'), width=12, className="graph-container"),
            ]),
            
            dbc.Row(id='factor-score-maps-row', children=[]),
        ], width=9)
    ])
], fluid=True)

# Callbacks for Factor Analysis page
def factor_analysis_callbacks(app):
    @app.callback(
        [Output('fa-scree-plot', 'figure'),
         Output('fa-variance-plot', 'figure'),
         Output('fa-loadings-heatmap', 'figure'),
         Output('factor-score-maps-row', 'children')],
        Input('n-factors-dropdown', 'value')
    )
    def update_fa_plots(n_factors):
        # ... (same Factor Analysis callback code from original)
        pass