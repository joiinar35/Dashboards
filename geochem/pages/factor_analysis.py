"""
Factor Analysis page and callbacks for identifying underlying factors in geochemical data.
"""

import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import logging
from factor_analyzer import FactorAnalyzer

# Import from shared_data module
try:
    from shared_data import data_for_analysis, df, column_title_map
except ImportError:
    # Fallback: define empty data structures
    data_for_analysis = pd.DataFrame()
    df = pd.DataFrame()
    column_title_map = {}

# Calculate initial number of factors for dropdown
if not data_for_analysis.empty:
    try:
        fa_initial = FactorAnalyzer(rotation=None, n_factors=data_for_analysis.shape[1])
        fa_initial.fit(data_for_analysis)
        eigenvalues_fa_initial, _ = fa_initial.get_eigenvalues()
        n_factors_initial = np.sum(eigenvalues_fa_initial > 1) if eigenvalues_fa_initial is not None else 1
    except Exception as e:
        logging.warning(f"Error calculating initial factors: {e}")
        n_factors_initial = 1
else:
    n_factors_initial = 1

def get_dropdown_options():
    """Generate dropdown options based on available data."""
    if data_for_analysis.empty:
        return [{'label': '1', 'value': 1}]
    
    n_cols = len(data_for_analysis.columns)
    max_n = max(1, n_cols-7)
    return [{'label': str(i), 'value': i} for i in range(1, max_n + 1)]

# Layout for Factor Analysis page
factor_analysis_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P(html.Strong("Factor Analysis helps identify underlying factors that explain the relationships between variables.")),
            html.P("The Factor Analysis Loadings Heatmap shows the relationships between your original geochemical variables and the underlying factors:"),
            html.P(html.Ul([
                html.Li([html.Strong("Rows: "), "Original geochemical variables (e.g., Ba, Co)."]),
                html.Li([html.Strong("Columns: "),  "Factors identified by the analysis (Factor 1, Factor 2, etc.)."]),
                html.Li([html.Strong("Colors and Values: "), "Indicate the 'loading' of a variable on a factor."]),
                html.Ul([
                    html.Li("High positive loading (warm colors, closer to 1): Variable is strongly and positively associated with the factor."),
                    html.Li("High negative loading (cool colors, closer to -1): Variable is strongly and negatively associated with the factor."),
                    html.Li("Loading near zero: Variable has little influence on that factor.")
                ])
            ])),
            html.P("By interpreting variables with high absolute loadings on each factor, you can understand the geological or geochemical processes represented by each factor."),
            html.P("The Factor Score Maps provide a glimpse the geographical predominance of each factor along the study area")
        ], className="explanation-text"), width=12),
    ]),

    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Factor Analysis Controls", className="mb-3"),
            html.Label("Number of Factors:"),
            dcc.Dropdown(
                id='n-factors-dropdown',
                options=get_dropdown_options(),
                value=n_factors_initial,
                clearable=False,
                disabled=data_for_analysis.empty
            ),
            html.Div(id='fa-controls-placeholder')
        ], width=2, className="sidebar"),

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
        ], width=10)
    ])
], fluid=True)

def make_empty_figure(title):
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, y=0.9,
            xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig

def make_scree_plot(eigenvalues):
    """Create scree plot for factor analysis eigenvalues."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f'{i+1}' for i in range(len(eigenvalues))],
        y=eigenvalues,
        mode='lines+markers',
        name='Eigenvalues'
    ))
    fig.add_shape(
        type="line", 
        x0=-0.5, y0=1, x1=len(eigenvalues)-0.5, y1=1,
        line=dict(color="Red", width=2, dash="dash")
    )
    fig.update_layout(
        title=dict(
            text='<b>Factor Analysis Scree Plot (Eigenvalues)</b>',
            x=0.5, y=0.9, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        xaxis_title='Factor Number',
        yaxis_title='Eigenvalue',
        showlegend=False,
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig

def make_variance_plot(fa_variance, valid_n_factors):
    """Create variance explained plot for factor analysis."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'Factor {i+1}' for i in range(valid_n_factors)],
        y=fa_variance[1][:valid_n_factors],  # Proportion of variance explained
        name='Proportion<br>Explained Variance'
    ))
    fig.add_trace(go.Scatter(
        x=[f'Factor {i+1}' for i in range(valid_n_factors)],
        y=fa_variance[2][:valid_n_factors],  # Cumulative proportion
        mode='lines+markers',
        name='Cumulative<br>Explained Variance'
    ))
    fig.update_layout(
        title=dict(
            text=f'<b>Factor Analysis Explained Variance ({valid_n_factors} Factors)</b>',
            x=0.5, y=0.9, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        xaxis_title='Factor Number',
        yaxis_title='Proportion of Variance',
        legend_title='Variance Type',
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(
            x=0.95, y=0.75, xanchor='right', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )
    return fig

def make_loadings_heatmap(fa_loadings, data_for_analysis, valid_n_factors):
    """Create heatmap of factor loadings."""
    loadings_df_fa = pd.DataFrame(
        fa_loadings,
        index=data_for_analysis.columns,
        columns=[f'Factor {i+1}' for i in range(fa_loadings.shape[1])]
    )
    fig = go.Figure(data=go.Heatmap(
        z=loadings_df_fa.values,
        x=loadings_df_fa.columns,
        y=loadings_df_fa.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='<b>Loading Value</b>', titleside='right')
    ))
    fig.update_layout(
        title=dict(
            text='<b>Factor Analysis Loadings Heatmap</b>',
            x=0.5, y=1, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        xaxis_title='Factor',
        yaxis_title='Variable',
        margin={"r": 0, "t": 40, "l": 0, "b": 0}
    )
    return fig

def make_factor_score_map(gdf_fa_scores, factor_score_col, i):
    """Create interpolated map for factor scores."""
    points = np.array([gdf_fa_scores.geometry.x, gdf_fa_scores.geometry.y]).T
    values = gdf_fa_scores[factor_score_col].values
    
    # Check if enough points for interpolation
    if len(points) < 4 or len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
        return make_empty_figure(f'<b>Factor {i+1} Score Map: Not enough data for interpolation.</b>')
    
    try:
        grid_density = 100
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        xi = np.linspace(x_min, x_max, grid_density)
        yi = np.linspace(y_min, y_max, grid_density)
        xi, yi = np.meshgrid(xi, yi)
        
        grid_values = griddata(points, values, (xi, yi), method='cubic')
        
        fig = go.Figure(data=go.Contour(
            z=grid_values,
            x=xi[0, :],
            y=yi[:, 0],
            ncontours=25,
            colorscale='RdBu',
            contours=dict(coloring='fill', showlabels=False),
            hoverinfo='z',
            hovertemplate='<b>%{z:.2f}</b><extra></extra>',
            colorbar=dict(
                title=f'<b>Factor {i+1} Score</b>',
                titleside='right'
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>Interpolated Factor {i+1} Score Map</b>',
                x=0.5, y=0.9, xanchor="center", yanchor="top",
                font=dict(size=16, color="black", family="Arial")
            ),
            xaxis=dict(
                showticklabels=False, showgrid=False, 
                zeroline=False, showline=False, ticks=''
            ),
            yaxis=dict(
                showticklabels=False, showgrid=False, 
                zeroline=False, showline=False, ticks=''
            ),
            height=600
        )
        return fig
        
    except Exception as e:
        logging.error(f"Error during interpolation for Factor {i+1}: {e}")
        return make_empty_figure(f'<b>Factor {i+1} Score Map: Error during interpolation.</b>')

def factor_analysis_callbacks(app):
    """Register all callbacks for the factor analysis page."""
    
    @app.callback(
        [
            Output('fa-scree-plot', 'figure'),
            Output('fa-variance-plot', 'figure'),
            Output('fa-loadings-heatmap', 'figure'),
            Output('factor-score-maps-row', 'children')
        ],
        [
            Input('tabs', 'value'),
            Input('n-factors-dropdown', 'value')
        ]
    )
    def update_fa_plots(tab_value, n_factors):
        """Update Factor Analysis plots and maps based on selected tab and number of factors."""
        # Initialize default return values
        scree_fig_fa = make_empty_figure("<b>Factor Analysis Scree Plot: Not enough data.</b>")
        variance_fig_fa = make_empty_figure("<b>Factor Analysis Explained Variance: Not enough data.</b>")
        heatmap_fig_fa = make_empty_figure("<b>Factor Analysis Loadings Heatmap: Not enough data.</b>")
        map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data for Factor Analysis maps."), width=12))]

        # Check if correct tab and data exists
        if tab_value != 'tab-fa' or data_for_analysis.empty:
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        # Validate number of factors
        valid_n_factors = min(
            n_factors if n_factors is not None else 1, 
            len(data_for_analysis.columns) if not data_for_analysis.empty else 0
        )
        
        if valid_n_factors < 1:
            error_msg = "Not enough data or factors selected."
            scree_fig_fa = make_empty_figure(f"<b>Factor Analysis Scree Plot: {error_msg}</b>")
            variance_fig_fa = make_empty_figure(f"<b>Factor Analysis Explained Variance: {error_msg}</b>")
            heatmap_fig_fa = make_empty_figure(f"<b>Factor Analysis Loadings Heatmap: {error_msg}</b>")
            map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or factors selected for Factor Analysis maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        try:
            # Perform Factor Analysis
            fa = FactorAnalyzer(rotation='varimax', n_factors=valid_n_factors)
            fa.fit(data_for_analysis)
            eigenvalues_fa, _ = fa.get_eigenvalues()
            fa_loadings = fa.loadings_
            fa_variance = fa.get_factor_variance()
            fa_scores = fa.transform(data_for_analysis)
            
            # Create factor scores dataframe
            fa_scores_df = pd.DataFrame(
                fa_scores, 
                columns=[f'Factor_{i+1}_Score' for i in range(fa_scores.shape[1])]
            )
            fa_scores_df.index = data_for_analysis.index
            
        except Exception as e:
            error_msg = f"Error during Factor Analysis: {str(e)}"
            logging.error(error_msg)
            scree_fig_fa = make_empty_figure(f"<b>Factor Analysis Scree Plot: {error_msg}</b>")
            variance_fig_fa = make_empty_figure(f"<b>Factor Analysis Explained Variance: {error_msg}</b>")
            heatmap_fig_fa = make_empty_figure(f"<b>Factor Analysis Loadings Heatmap: {error_msg}</b>")
            map_rows = [dbc.Row(dbc.Col(html.Div(error_msg), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        # Create Scree Plot
        if eigenvalues_fa is not None and len(eigenvalues_fa) > 0:
            scree_fig_fa = make_scree_plot(eigenvalues_fa)
        else:
            scree_fig_fa = make_empty_figure("<b>Factor Analysis Scree Plot: No eigenvalues calculated.</b>")

        # Create Variance Plot
        if (fa_variance is not None and len(fa_variance) > 2 and 
            len(fa_variance[1]) >= valid_n_factors and len(fa_variance[2]) >= valid_n_factors):
            variance_fig_fa = make_variance_plot(fa_variance, valid_n_factors)
        else:
            variance_fig_fa = make_empty_figure("<b>Factor Analysis Explained Variance: Variance data not available.</b>")

        # Create Loadings Heatmap
        if (fa_loadings is not None and not data_for_analysis.empty and
            fa_loadings.shape[0] == len(data_for_analysis.columns) and
            fa_loadings.shape[1] == valid_n_factors):
            heatmap_fig_fa = make_loadings_heatmap(fa_loadings, data_for_analysis, valid_n_factors)
        else:
            heatmap_fig_fa = make_empty_figure("<b>Factor Analysis Loadings Heatmap: Loadings data not available.</b>")

        # Generate Factor Score Maps
        map_rows = []
        if (df.empty or 'x_utm' not in df.columns or 'y_utm' not in df.columns or
            data_for_analysis.empty or fa_scores_df.empty):
            logging.error("Not enough data or missing coordinates for factor score maps.")
            map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or missing coordinates for factor score maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        try:
            # Add coordinates to factor scores
            fa_scores_df_with_coords = fa_scores_df.copy()
            fa_scores_df_with_coords['x_utm'] = df.loc[data_for_analysis.index, 'x_utm'].values
            fa_scores_df_with_coords['y_utm'] = df.loc[data_for_analysis.index, 'y_utm'].values

            # Create GeoDataFrame
            gdf_fa_scores = gpd.GeoDataFrame(
                fa_scores_df_with_coords,
                geometry=gpd.points_from_xy(
                    fa_scores_df_with_coords.x_utm, 
                    fa_scores_df_with_coords.y_utm
                ),
                crs="EPSG:32721"
            )
            
        except Exception as e:
            error_msg = f"Error preparing geographic data: {str(e)}"
            logging.error(error_msg)
            map_rows = [dbc.Row(dbc.Col(html.Div(error_msg), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        if gdf_fa_scores.empty:
            logging.error("GeoDataFrame for factor scores is empty.")
            map_rows = [dbc.Row(dbc.Col(html.Div("No geographic data for factor scores."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        # Create maps for each factor
        map_cols = []
        for i in range(valid_n_factors):
            factor_score_col = f'Factor_{i+1}_Score'
            if factor_score_col in gdf_fa_scores.columns:
                map_fig = make_factor_score_map(gdf_fa_scores, factor_score_col, i)
            else:
                map_fig = make_empty_figure(f'<b>Factor {i+1} Score Map: Data not available.</b>')
            map_cols.append(dbc.Col(dcc.Graph(figure=map_fig), width=6))

        # Arrange maps in rows (2 per row)
        for i in range(0, len(map_cols), 2):
            map_rows.append(dbc.Row(map_cols[i:i+2], className="mb-4"))

        return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
