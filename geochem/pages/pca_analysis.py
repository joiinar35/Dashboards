from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from utils.data_loader import data_for_analysis, scaled_data_df, df, gdf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from scipy.interpolate import griddata

# Layout for PCA Analysis page
pca_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("The PCA loadings heatmap shows how much each original variable contributes to each principal component."),
            html.P("• Rows represent your original geochemical variables (e.g., Ba, Co, Cr)."),
            html.P("• Columns represent the principal components (PC1, PC2, etc.)."),
            html.P("• Colors and Values: The color and the number in each cell indicate the 'loading' of that variable on that principal component."),
            html.Ul([
                html.Li("A high positive loading (warm colors, closer to 1) means the variable is strongly and positively correlated with that principal component."),
                html.Li("A high negative loading (cool colors, closer to -1) means the variable is strongly and negatively correlated with that principal component."),
                html.Li("A loading close to zero (around the center color) means the variable has little influence on that principal component.")
            ]),
            html.P("By looking at the variables with high absolute loadings on each principal component, you can interpret what each component represents in terms of the original geochemical data.")
        ], className="explanation-text"), width=12),
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("PCA Controls", className="mb-3"),
            html.Label("Number of Components:"),
            dcc.Dropdown(
                id='n-components-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(2, len(data_for_analysis.columns) + 1)],
                value=min(3, len(data_for_analysis.columns)) if not data_for_analysis.empty else 2,
                clearable=False
            ),
            html.Hr(),
            html.H4("Clustering Controls", className="mb-3"),
            html.Label("Number of Clusters (k):"),
            dcc.Dropdown(
                id='n-clusters-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(2, 12)],
                value=2,
                clearable=False
            ),
            html.Div(id='pca-controls-placeholder')
        ], width=3, className="sidebar"),
        
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='pca-scatter-plot'), width=6, className="graph-container"),
                dbc.Col(dcc.Graph(id='pca-scree-plot'), width=6, className="graph-container"),
            ]),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='pca-loadings-heatmap'), width=12, className="graph-container"),
            ]),
            
            dbc.Row([
                dbc.Col(html.H4("Clustering based on PCA Components", className="text-center my-4"), width=12),
            ]),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='cluster-map'), width=12, className="graph-container"),
            ]),
        ], width=9)
    ])
], fluid=True)

# Callbacks for PCA Analysis page
def pca_analysis_callbacks(app):
    @app.callback(
        [Output('pca-scatter-plot', 'figure'),
         Output('pca-scree-plot', 'figure'),
         Output('pca-loadings-heatmap', 'figure')],
        [Input('n-components-dropdown', 'value'),
         Input('n-clusters-dropdown', 'value')]
    )
    def update_pca_plots(n_components, n_clusters):
        # ... (same PCA plots callback code from original)
        pass

    @app.callback(
        Output('cluster-map', 'figure'),
        [Input('n-components-dropdown', 'value'),
         Input('n-clusters-dropdown', 'value')]
    )
    def update_cluster_map(n_components, n_clusters):
        # ... (same cluster map callback code from original)
        pass