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
        [Input('tabs', 'value'),
         Input('n-clusters-dropdown', 'value')]
    )
    def update_pca_plots(tab_value, n_components):
        if tab_value != 'tab-pca' or scaled_data_for_analysis_df.empty:
            return go.Figure().update_layout(title=dict(text="<b>PCA Scatter Plot: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial"))), go.Figure().update_layout(title=dict(text="<b>PCA Scree Plot: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial"))), go.Figure().update_layout(title=dict(text="<b>PCA Loadings Heatmap: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

    # Ensure n_components is valid for PCA
        valid_n_components = min(n_components if n_components is not None else 1, scaled_data_for_analysis_df.shape[1])
        if valid_n_components < 1:
            return go.Figure().update_layout(title=dict(text="<b>PCA Scatter Plot: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial"))), go.Figure().update_layout(title=dict(text="<b>PCA Scree Plot: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial"))), go.Figure().update_layout(title=dict(text="<b>PCA Loadings Heatmap: Not enough data or components.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


    # Re-run PCA with the selected number of components
        pca = PCA(n_components=valid_n_components)
        pca.fit(scaled_data_for_analysis_df)
        pca_components = pca.transform(scaled_data_for_analysis_df)
        pca_explained_variance = pca.explained_variance_ratio_
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
        # Update pca_df with new components
        pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(pca_components.shape[1])])
        pca_df = pca_df.set_index(scaled_data_for_analysis_df.index)
    
        # --- Update PCA Scatter Plot ---
        scatter_fig = px.scatter(
            pca_df,
            x='PC1',
            y='PC2' if valid_n_components >= 2 else 'PC1', # Plot PC1 vs PC1 if only 1 component
            title='PCA: PC1 vs PC2' if valid_n_components >= 2 else 'PCA: PC1',
            hover_data=pca_df.columns # Display all PC values on hover
        )
        scatter_fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, title=dict(text='<b>PCA: PC1 vs PC2</b>' if valid_n_components >= 2 else '<b>PCA: PC1</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
    
    
        # --- Update PCA Scree Plot ---
        # The x-axis should show all selected components
        explained_variance_subset = pca_explained_variance[:valid_n_components] # Use valid_n_components here
        cumulative_variance_subset = np.cumsum(explained_variance_subset)
    
        scree_fig = go.Figure()
        scree_fig.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(valid_n_components)], # Use valid_n_components here
            y=explained_variance_subset,
            name='Individual Explained Variance',
            showlegend=False # Hide this legend
        ))
        scree_fig.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(valid_n_components)], # Use valid_n_components here
            y=cumulative_variance_subset,
            mode='lines+markers',
            name='Cumulative Explained Variance'
        ))
    
        scree_fig.update_layout(
            title=dict(text=f'<b>PCA Scree Plot ({valid_n_components} Components)</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            margin={"r":0,"t":40,"l":0,"b":0},
            legend=dict(
                x=0.95, # Position the legend inside the plot
                y=0.95,
                xanchor='right', # Anchor point for x
                yanchor='top', # Anchor point for y
                bgcolor='rgba(255, 255, 255, 0.5)' # Optional: add background for readability
            )
        )
    
        # --- Update PCA Loadings Heatmap ---
        loadings_df = pd.DataFrame(
            pca_loadings,
            index=scaled_data_for_analysis_df.columns,
            columns=[f'PC{i+1}' for i in range(pca_loadings.shape[1])]
        )
    
        heatmap_fig = go.Figure(data=go.Heatmap(
                           z=loadings_df.values,
                           x=loadings_df.columns,
                           y=loadings_df.index,
                           colorscale='RdBu',
                           zmin=-1,
                           zmax=1,
                           colorbar=dict(
                               title='<b>Loading Value</b>',
                               titleside='right'
                           )))
    
        heatmap_fig.update_layout(
            title=dict(text='<b>PCA Loadings Heatmap</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
            xaxis_title='Principal Component',
            yaxis_title='Variable',
            margin={"r":0,"t":40,"l":0,"b":0}
        )
    
        return scatter_fig, scree_fig, heatmap_fig
        pass

   @app.callback(
        Output('cluster-map', 'figure'),
        [Input('tabs', 'value'),
         Input('n-clusters-dropdown', 'value')]
   )
    def update_cluster_map(tab_value, n_components, n_clusters):
    # Check for valid inputs and data
        if tab_value != 'tab-pca' or scaled_data_for_analysis_df.empty or gdf.empty or n_clusters is None or n_clusters < 2 or 'x_utm' not in df.columns or 'y_utm' not in df.columns:
            return go.Figure().update_layout(title=dict(text="<b>Cluster Map: Not enough data or invalid number of clusters.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

    # Re-run PCA with the selected number of components (needed for clustering)
    # Ensure n_components is valid for PCA before clustering
       valid_n_components = min(n_components if n_components is not None else 1, scaled_data_for_analysis_df.shape[1])
        if valid_n_components < 1:
           return go.Figure().update_layout(title=dict(text="<b>Cluster Map: Not enough data or components for PCA.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

        pca = PCA(n_components=valid_n_components)
        pca_components = pca.fit_transform(scaled_data_for_analysis_df)

    # Perform KMeans clustering on the PCA components
    # Ensure there are enough samples for clustering with the chosen k
        if pca_components.shape[0] < n_clusters:
         return go.Figure().update_layout(title=dict(text=f"<b>Cluster Map: Not enough samples ({pca_components.shape[0]}) for {n_clusters} clusters.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(pca_components)
        except ValueError as e:
            print(f"Error during KMeans clustering: {e}")
            return go.Figure().update_layout(title=dict(text=f"<b>Cluster Map: Error during clustering - {e}</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


    # Add cluster labels and coordinates to a DataFrame for plotting
    # Ensure the index of scaled_data_for_analysis_df is used to align with df
        cluster_df = pd.DataFrame({
        'cluster': clusters,
        'x_utm': df.loc[scaled_data_for_analysis_df.index, 'x_utm'].values,
        'y_utm': df.loc[scaled_data_for_analysis_df.index, 'y_utm'].values
        })


    # Create GeoDataFrame
        gdf_clusters = gpd.GeoDataFrame(
        cluster_df,
        geometry=gpd.points_from_xy(cluster_df.x_utm, cluster_df.y_utm),
        crs="EPSG:32721" # UTM zone 21S
        )

    # Create an interpolated map of clusters
        points = np.array([gdf_clusters.geometry.x, gdf_clusters.geometry.y]).T
        values = gdf_clusters['cluster'].values

    # Create a regular grid for interpolation (adjust density for performance)
        grid_density = 100
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        xi = np.linspace(x_min, x_max, grid_density)
        yi = np.linspace(y_min, y_max, grid_density)
        xi, yi = np.meshgrid(xi, yi)

    # Interpolation (using nearest neighbor for discrete clusters)
    # Handle potential errors during interpolation
        try:
        # Check if there are enough unique points for interpolation
            if len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
                return go.Figure().update_layout(title=dict(text='<b>Cluster Map: Not enough unique geographic coordinates for interpolation.</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

            grid_values = griddata(points, values, (xi, yi), method='nearest')

        # Create contour map for the clusters
            fig = go.Figure(data=go.Contour(
            z=grid_values,
            x=xi[0, :],
            y=yi[:, 0],
            ncontours=n_clusters, # Number of contours equals number of clusters
            colorscale='Viridis', # Use a suitable colorscale for discrete categories
            contours=dict(
                coloring='heatmap', # Use heatmap coloring for filled contours
                showlabels=True, # Show cluster labels on contours
                labelfont=dict(size=12, color='white')
            ),
            hoverinfo='z',
            hovertemplate='Cluster: %{z:.0f}<extra></extra>', # Display cluster number on hover
            colorbar=dict(
                title='<b>Cluster</b>',
                titleside='right',
                tickvals=np.arange(n_clusters), # Show tick marks for each cluster
                ticktext=[str(i) for i in range(n_clusters)] # Label ticks with cluster numbers
                )
            ))

            fig.update_layout(
            title=dict(text=f'<b>Interpolated Cluster Map (k={n_clusters})</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
             xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
             yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
            height=600,
            )
        except Exception as e:
             print(f"Error during interpolation for cluster map: {e}")
             fig = go.Figure().update_layout(title=dict(text='<b>Cluster Map: Error during interpolation.</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


        return fig
        pass
