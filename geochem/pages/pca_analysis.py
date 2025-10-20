"""
PCA Analysis page and callbacks for principal component analysis and clustering.
"""

from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import from shared_data module
try:
    from shared_data import data_for_analysis, scaled_data_df, df, gdf
except ImportError:
    # Fallback: define empty data structures
    data_for_analysis = pd.DataFrame()
    scaled_data_df = pd.DataFrame()
    df = pd.DataFrame()
    gdf = pd.DataFrame()

# Helper function for safe data access
def get_data():
    """Safely return data structures with fallbacks."""
    return data_for_analysis, scaled_data_df, df, gdf

# Layout for PCA Analysis page
pca_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P(html.Strong("The PCA loadings heatmap shows how much each original variable contributes to each principal component.")),
            html.P([html.Strong("• Rows"), "represent your original geochemical variables (e.g., Ba, Co, Cr)."]),
            html.P([html.Strong("• Columns"), "represent the principal components (PC1, PC2, etc.)."]),
            html.P([html.Strong("• Colors and Values: "), "The color and the number in each cell indicate the 'loading' of that variable on that principal component."]),
            html.P(html.Ul([
                html.Li("A high positive loading (warm colors, closer to 1) means the variable is strongly and positively correlated with that principal component."),
                html.Li("A high negative loading (cool colors, closer to -1) means the variable is strongly and negatively correlated with that principal component."),
                html.Li("A loading close to zero (around the center color) means the variable has little influence on that principal component.")
            ])),
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
                options=[{'label': str(i), 'value': i} for i in range(2, len(data_for_analysis.columns) - 5)] if not data_for_analysis.empty else [],
                value=min(2, len(data_for_analysis.columns)) if not data_for_analysis.empty else None,
                clearable=False,
                disabled=data_for_analysis.empty
            ),
            html.Hr(),
            html.H4("Clustering Controls", className="mb-3"),
            html.Label("Number of Clusters (k):"),
            dcc.Dropdown(
                id='n-clusters-dropdown',
                options=[{'label': str(i), 'value': i} for i in range(2, 6)],
                value=2,
                clearable=False
            ),
            html.Div(id='pca-controls-placeholder')
        ], width=2, className="sidebar"),

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
            # Optional: Add a location to show error messages to the user
            dbc.Row([
                dbc.Col(html.Div(id='pca-error-message', className="text-danger"), width=12)
            ])
        ], width=10)
    ])
], fluid=True)

def empty_figure(msg):
    """Return an empty figure with a message."""
    return go.Figure().update_layout(
        title=dict(
            text=f"<b>{msg}</b>",
            x=0.5, y=0.9, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial"))
    )

def pca_analysis_callbacks(app):
    """Register all callbacks for the PCA analysis page."""
    
    @app.callback(
        [
            Output('pca-scatter-plot', 'figure'),
            Output('pca-scree-plot', 'figure'),
            Output('pca-loadings-heatmap', 'figure'),
            Output('pca-error-message', 'children'),
        ],
        [
            Input('tabs', 'value'),
            Input('n-components-dropdown', 'value')
        ]
    )
    def update_pca_plots(tab_value, n_components):
        """Update PCA plots based on selected number of components."""
        # Get current data
        data_for_analysis, scaled_data_df, df, gdf = get_data()
        error_msg = ""

        if tab_value != 'tab-pca':
            return [empty_figure("PCA Scatter Plot: Not selected."),
                    empty_figure("PCA Scree Plot: Not selected."),
                    empty_figure("PCA Loadings Heatmap: Not selected."),
                    ""]

        if scaled_data_df is None or scaled_data_df.empty:
            return [empty_figure("Not enough data for PCA."),
                    empty_figure("Not enough data for PCA."),
                    empty_figure("Not enough data for PCA."),
                    "No data available for PCA analysis."]

        # Validate n_components
        max_components = min(scaled_data_df.shape[0], scaled_data_df.shape[1])
        if n_components is None or not (1 <= n_components <= max_components):
            error_msg = f"Please select between 1 and {max_components} components."
            return [empty_figure(error_msg)] * 3 + [error_msg]

        # Defensive: check for NaN
        if scaled_data_df.isnull().values.any():
            error_msg = "Input data contains NaN values. Please clean or impute your data."
            return [empty_figure(error_msg)] * 3 + [error_msg]

        try:
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(scaled_data_df)
            pca_explained_variance = pca.explained_variance_ratio_
            pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

            # Create PCA results dataframe
            pca_df = pd.DataFrame(
                pca_components, 
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            pca_df = pca_df.set_index(scaled_data_df.index)

            # Scatter Plot
            x_col = 'PC1'
            y_col = 'PC2' if n_components >= 2 else 'PC1'
            scatter_fig = px.scatter(
                pca_df,
                x=x_col,
                y=y_col,
                title='PCA: PC1 vs PC2' if n_components >= 2 else 'PCA: PC1',
                hover_data=pca_df.columns
            )
            scatter_fig.update_layout(
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                title=dict(
                    text='<b>PCA: PC1 vs PC2</b>' if n_components >= 2 else '<b>PCA: PC1</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial"))
            )

            # Scree Plot
            explained_variance_subset = pca_explained_variance[:n_components]
            cumulative_variance_subset = np.cumsum(explained_variance_subset)
            
            scree_fig = go.Figure()
            scree_fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_variance_subset,
                name='Individual Explained Variance',
                showlegend=False
            ))
            scree_fig.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=cumulative_variance_subset,
                mode='lines+markers',
                name='Cumulative Explained Variance'
            ))
            scree_fig.update_layout(
                title=dict(
                    text=f'<b>PCA Scree Plot ({n_components} Components)</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")),
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend=dict(
                    x=0.95, y=0.75,
                    xanchor='right', yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.5)'
                )
            )

            # Loadings Heatmap
            loadings_df = pd.DataFrame(
                pca_loadings,
                index=scaled_data_df.columns,
                columns=[f'PC{i+1}' for i in range(pca_loadings.shape[1])]
            )
            
            # Normalize loadings for consistent heatmap color scaling
            max_abs_loading = np.abs(loadings_df.values).max()
            zmin, zmax = (-1, 1) if max_abs_loading <= 1 else (-max_abs_loading, max_abs_loading)
            
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=loadings_df.values,
                x=loadings_df.columns,
                y=loadings_df.index,
                colorscale='RdBu',
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(title='<b>Loading Value</b>', titleside='right')
            ))
            heatmap_fig.update_layout(
                title=dict(
                    text='<b>PCA Loadings Heatmap</b>',
                    x=0.5, y=1, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")),
                xaxis_title='Principal Component',
                yaxis_title='Variable',
                margin={"r": 0, "t": 40, "l": 0, "b": 0}
            )

            return scatter_fig, scree_fig, heatmap_fig, ""

        except Exception as e:
            error_msg = f"Error during PCA analysis: {str(e)}"
            return [empty_figure(error_msg)] * 3 + [error_msg]

    @app.callback(
        Output('cluster-map', 'figure'),
        [
            Input('tabs', 'value'),
            Input('n-components-dropdown', 'value'),
            Input('n-clusters-dropdown', 'value')
        ]
    )
    def update_cluster_map(tab_value, n_components, n_clusters):
        """Update cluster map based on PCA components and cluster count."""
        data_for_analysis, scaled_data_df, df, gdf = get_data()

        if tab_value != 'tab-pca':
            return empty_figure("Cluster Map: Not selected.")

        # Validate inputs and data availability
        if (scaled_data_df is None or scaled_data_df.empty or
            gdf is None or gdf.empty or
            n_clusters is None or n_clusters < 2 or
            df is None or
            'x_utm' not in df.columns or 'y_utm' not in df.columns):
            return empty_figure("Cluster Map: Not enough data or invalid number of clusters.")

        # Check n_components validity
        max_components = min(scaled_data_df.shape[0], scaled_data_df.shape[1])
        if n_components is None or not (1 <= n_components <= max_components):
            return empty_figure(f"Cluster Map: Please select between 1 and {max_components} components.")

        # Check for NaN values
        if scaled_data_df.isnull().values.any():
            return empty_figure("Cluster Map: Input data contains NaN values.")

        # Check enough samples for KMeans
        if scaled_data_df.shape[0] < n_clusters:
            return empty_figure(
                f"Cluster Map: Not enough samples ({scaled_data_df.shape[0]}) for {n_clusters} clusters."
            )

        try:
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_components = pca.fit_transform(scaled_data_df)

            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(pca_components)
        except Exception as e:
            return empty_figure(f"Cluster Map: Error during PCA or clustering - {str(e)}")

        # Create cluster dataframe with coordinates
        try:
            # Safely align indices
            cluster_df = pd.DataFrame({
                'cluster': clusters,
                'x_utm': df.loc[scaled_data_df.index, 'x_utm'].values,
                'y_utm': df.loc[scaled_data_df.index, 'y_utm'].values
            })
        except Exception as e:
            return empty_figure(f"Cluster Map: Error aligning indices/coordinates for clusters - {str(e)}")

        # Create GeoDataFrame for spatial operations
        try:
            gdf_clusters = gpd.GeoDataFrame(
                cluster_df,
                geometry=gpd.points_from_xy(cluster_df.x_utm, cluster_df.y_utm),
                crs="EPSG:32721"  # UTM zone 21S
            )
        except Exception as e:
            return empty_figure(f"Cluster Map: Error creating GeoDataFrame - {str(e)}")

        # Create interpolated cluster map
        try:
            points = np.array([gdf_clusters.geometry.x, gdf_clusters.geometry.y]).T
            values = gdf_clusters['cluster'].values

            # Create interpolation grid
            grid_density = 70  # Reduced for better performance
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            if x_max == x_min or y_max == y_min:
                return empty_figure("Cluster Map: Not enough unique geographic coordinates for interpolation.")

            xi = np.linspace(x_min, x_max, grid_density)
            yi = np.linspace(y_min, y_max, grid_density)
            xi, yi = np.meshgrid(xi, yi)

            # Check for sufficient unique coordinates
            if len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
                return empty_figure("Cluster Map: Not enough unique geographic coordinates for interpolation.")

            # Perform interpolation (nearest neighbor for discrete clusters)
            grid_values = griddata(points, values, (xi, yi), method='nearest')

            # Create contour plot
            fig = go.Figure(data=go.Contour(
                z=grid_values,
                x=xi[0, :],
                y=yi[:, 0],
                ncontours=n_clusters,
                colorscale='Viridis',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                hoverinfo='z',
                hovertemplate='Cluster: %{z:.0f}<extra></extra>',
                colorbar=dict(
                    title='<b>Cluster</b>',
                    titleside='right',
                    tickvals=np.arange(n_clusters),
                    ticktext=[str(i) for i in range(n_clusters)]
                )
            ))

            fig.update_layout(
                title=dict(
                    text=f'<b>Interpolated Cluster Map (k={n_clusters})</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")),
                xaxis=dict(
                    showticklabels=False, showgrid=False, 
                    zeroline=False, showline=False, ticks=''
                ),
                yaxis=dict(
                    showticklabels=False, showgrid=False, 
                    zeroline=False, showline=False, ticks=''
                ),
                height=800,
            )
            
            return fig

        except Exception as e:
            return empty_figure(f"Cluster Map: Error during interpolation - {str(e)}")