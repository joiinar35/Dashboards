import pandas as pd
import geopandas as gpd
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

    dcc.Store(id='df-store', data=df.to_dict('records'))
    dcc.Store(id='data-for-analysis-store', data=data_for_analysis.to_dict('records'))

# Callbacks for Factor Analysis page
def factor_analysis_callbacks(app):
    @app.callback(
       [Output('fa-scree-plot', 'figure'),
     Output('fa-variance-plot', 'figure'),
     Output('fa-loadings-heatmap', 'figure'),
     Output('factor-score-maps-row', 'children')], # Output for dynamic maps
     [Input('tabs', 'value'), Input('df-store', 'data'), Input('data-for-analysis-store', 'data'),
     Input('n-factors-dropdown', 'value')] # Added input for dropdown
    )
    def update_fa_plots(tab_value, df_data, data_for_analysis_data, n_factors):import pandas as pd
import geopandas as gpd
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from utils.data_loader import data_for_analysis, df
from factor_analyzer import FactorAnalyzer
import numpy as np
from scipy.interpolate import griddata
import logging

# Calculate initial number of factors for dropdown
if not data_for_analysis.empty:
    fa_initial = FactorAnalyzer(rotation=None, n_factors=data_for_analysis.shape[1])
    fa_initial.fit(data_for_analysis)
    eigenvalues_fa_initial, _ = fa_initial.get_eigenvalues()
    n_factors_initial = sum(eigenvalues_fa_initial > 1) if eigenvalues_fa_initial is not None else 1
else:
    n_factors_initial = 1

def get_dropdown_options():
    n_cols = len(data_for_analysis.columns)
    max_n = max(1, n_cols)
    return [{'label': str(i), 'value': i} for i in range(1, max_n + 1)]

# Layout for Factor Analysis page
factor_analysis_layout = dbc.Container([
    dcc.Store(id='df-store', data=df.to_dict('records')),
    dcc.Store(id='data-for-analysis-store', data=data_for_analysis.to_dict('records')),
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
                options=get_dropdown_options(),
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

def make_empty_figure(title):
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, y=0.9,
            xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        margin={"r":0, "t":40, "l":0, "b":0}
    )
    return fig

def make_scree_plot(eigenvalues):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[f'{i+1}' for i in range(len(eigenvalues))],
        y=eigenvalues,
        mode='lines+markers',
        name='Eigenvalues'
    ))
    fig.add_shape(type="line", x0=-0.5, y0=1, x1=len(eigenvalues)-0.5, y1=1,
                  line=dict(color="Red", width=2, dash="dash"))
    fig.update_layout(
        title=dict(
            text='<b>Factor Analysis Scree Plot (Eigenvalues)</b>',
            x=0.5, y=0.9, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        xaxis_title='Factor Number',
        yaxis_title='Eigenvalue',
        showlegend=False,
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

def make_variance_plot(fa_variance, valid_n_factors):
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
        margin={"r":0,"t":40,"l":0,"b":0},
        legend=dict(
            x=0.95, y=0.95, xanchor='right', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.5)'
        )
    )
    return fig

def make_loadings_heatmap(fa_loadings, data_for_analysis, valid_n_factors):
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
        colorbar=dict(
            title='<b>Loading Value</b>',
            titleside='right'
        )))
    fig.update_layout(
        title=dict(
            text='<b>Factor Analysis Loadings Heatmap</b>',
            x=0.5, y=0.9, xanchor="center", yanchor="top",
            font=dict(size=16, color="black", family="Arial")
        ),
        xaxis_title='Factor',
        yaxis_title='Variable',
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    return fig

def make_factor_score_map(gdf_fa_scores, factor_score_col, i):
    points = np.array([gdf_fa_scores.geometry.x, gdf_fa_scores.geometry.y]).T
    values = gdf_fa_scores[factor_score_col].values
    # Interpolation: check if enough points (at least 4 and both x/y vary)
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
            contours=dict(
                coloring='fill',
                showlabels=False
            ),
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
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
            height=400
        )
        return fig
    except Exception as e:
        logging.error(f"Error during interpolation for Factor {i+1}: {e}")
        return make_empty_figure(f'<b>Factor {i+1} Score Map: Error during interpolation.</b>')

def factor_analysis_callbacks(app):
    @app.callback(
        [
            Output('fa-scree-plot', 'figure'),
            Output('fa-variance-plot', 'figure'),
            Output('fa-loadings-heatmap', 'figure'),
            Output('factor-score-maps-row', 'children')
        ],
        [
            Input('tabs', 'value'),
            Input('df-store', 'data'),
            Input('data-for-analysis-store', 'data'),
            Input('n-factors-dropdown', 'value')
        ]
    )
    def update_fa_plots(tab_value, df_data, data_for_analysis_data, n_factors):
        """Update Factor Analysis plots and maps based on selected tab and number of factors."""
        df_local = pd.DataFrame(df_data)
        data_for_analysis_local = pd.DataFrame(data_for_analysis_data)

        scree_fig_fa = make_empty_figure("<b>Factor Analysis Scree Plot: Not enough data.</b>")
        variance_fig_fa = make_empty_figure("<b>Factor Analysis Explained Variance: Not enough data.</b>")
        heatmap_fig_fa = make_empty_figure("<b>Factor Analysis Loadings Heatmap: Not enough data.</b>")
        map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data for Factor Analysis maps."), width=12))]

        if tab_value != 'tab-fa' or data_for_analysis_local.empty:
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        valid_n_factors = min(n_factors if n_factors is not None else 1, len(data_for_analysis_local.columns) if not data_for_analysis_local.empty else 0)
        if valid_n_factors < 1:
            scree_fig_fa = make_empty_figure("<b>Factor Analysis Scree Plot: Not enough data or factors selected.</b>")
            variance_fig_fa = make_empty_figure("<b>Factor Analysis Explained Variance: Not enough data or factors selected.</b>")
            heatmap_fig_fa = make_empty_figure("<b>Factor Analysis Loadings Heatmap: Not enough data or factors selected.</b>")
            map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or factors selected for Factor Analysis maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        try:
            fa = FactorAnalyzer(rotation='varimax', n_factors=valid_n_factors)
            fa.fit(data_for_analysis_local)
            eigenvalues_fa, _ = fa.get_eigenvalues()
            fa_loadings = fa.loadings_
            fa_variance = fa.get_factor_variance()
            fa_scores = fa.transform(data_for_analysis_local)
            fa_scores_df = pd.DataFrame(fa_scores, columns=[f'Factor_{i+1}_Score' for i in range(fa_scores.shape[1])])
            fa_scores_df.index = data_for_analysis_local.index
        except Exception as e:
            logging.error(f"Error during Factor Analysis fit or transformation: {e}")
            scree_fig_fa = make_empty_figure(f"<b>Factor Analysis Scree Plot: Error ({e}).</b>")
            variance_fig_fa = make_empty_figure(f"<b>Factor Analysis Explained Variance: Error ({e}).</b>")
            heatmap_fig_fa = make_empty_figure(f"<b>Factor Analysis Loadings Heatmap: Error ({e}).</b>")
            map_rows = [dbc.Row(dbc.Col(html.Div(f"Error during Factor Analysis: {e}"), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        # Scree plot
        if eigenvalues_fa is not None and len(eigenvalues_fa) > 0:
            scree_fig_fa = make_scree_plot(eigenvalues_fa)
        else:
            scree_fig_fa = make_empty_figure("<b>Factor Analysis Scree Plot: No eigenvalues calculated.</b>")

        # Variance plot
        if fa_variance is not None and len(fa_variance) > 2 and len(fa_variance[1]) >= valid_n_factors and len(fa_variance[2]) >= valid_n_factors:
            variance_fig_fa = make_variance_plot(fa_variance, valid_n_factors)
        else:
            variance_fig_fa = make_empty_figure("<b>Factor Analysis Explained Variance: Variance data not available or incomplete.</b>")

        # Loadings heatmap
        if (
            fa_loadings is not None and not data_for_analysis_local.empty and
            fa_loadings.shape[0] == len(data_for_analysis_local.columns) and
            fa_loadings.shape[1] == valid_n_factors
        ):
            heatmap_fig_fa = make_loadings_heatmap(fa_loadings, data_for_analysis_local, valid_n_factors)
        else:
            heatmap_fig_fa = make_empty_figure("<b>Factor Analysis Loadings Heatmap: Loadings data not available or shape mismatch.</b>")

        # Generate Factor Score Maps
        map_rows = []
        if (
            df_local.empty or 'x_utm' not in df_local.columns or 'y_utm' not in df_local.columns or
            data_for_analysis_local.empty or fa_scores_df.empty
        ):
            logging.error("Not enough data or missing coordinates in original df or fa_scores_df for factor score maps.")
            map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or missing coordinates for factor score maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        # Add x_utm and y_utm to the scores dataframe using indices from data_for_analysis
        fa_scores_df_with_coords = fa_scores_df.copy()
        if len(data_for_analysis_local) == len(fa_scores_df_with_coords):
            try:
                # Index alignment: match indices for coordinate assignment
                fa_scores_df_with_coords['x_utm'] = df_local.loc[data_for_analysis_local.index, 'x_utm'].values
                fa_scores_df_with_coords['y_utm'] = df_local.loc[data_for_analysis_local.index, 'y_utm'].values
            except Exception as e:
                logging.error(f"Error adding coordinates to factor scores dataframe: {e}")
                map_rows = [dbc.Row(dbc.Col(html.Div(f"Error adding coordinates for factor score maps: {e}"), width=12))]
                return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
        else:
            logging.error("Index mismatch when adding coordinates to factor scores.")
            map_rows = [dbc.Row(dbc.Col(html.Div("Index mismatch when adding coordinates for factor score maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows

        try:
            gdf_fa_scores = gpd.GeoDataFrame(
                fa_scores_df_with_coords,
                geometry=gpd.points_from_xy(fa_scores_df_with_coords.x_utm, fa_scores_df_with_coords.y_utm),
                crs="EPSG:32721"  # Adjust CRS as appropriate
            )
        except Exception as e:
            logging.error(f"Error creating GeoDataFrame for factor scores: {e}")
            map_rows = [dbc.Row(dbc.Col(html.Div(f"Error creating geographic data for factor scores: {e}"), width=12))]
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
         """Update Factor Analysis plots and maps based on selected tab and number of factors."""
    # Access global df and data_for_analysis here
         df = pd.DataFrame(df_data)
        data_for_analysis = pd.DataFrame(data_for_analysis_data)
    # Initialize figures and map children for empty state
        scree_fig_fa = go.Figure()
        variance_fig_fa = go.Figure()
        heatmap_fig_fa = go.Figure()
        map_rows = []
        
        
        if tab_value != 'tab-fa' or data_for_analysis.empty:
            scree_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Scree Plot: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
            variance_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Explained Variance: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
            heatmap_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Loadings Heatmap: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
            map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data for Factor Analysis maps."), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows


        # Re-run Factor Analysis with the selected number of factors
        # Ensure n_factors is valid and less than or equal to the number of features
        valid_n_factors = min(n_factors if n_factors is not None else 1, len(data_for_analysis.columns) if not data_for_analysis.empty else 0)
        if valid_n_factors < 1:
             scree_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Scree Plot: Not enough data or factors selected.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             variance_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Explained Variance: Not enough data or factors selected.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             heatmap_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Loadings Heatmap: Not enough data or factors selected.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or factors selected for Factor Analysis maps."), width=12))]
             return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
         try:
             fa = FactorAnalyzer(rotation='varimax', n_factors=valid_n_factors)
             fa.fit(data_for_analysis)
             eigenvalues_fa, _ = fa.get_eigenvalues()
             fa_loadings = fa.loadings_
             fa_variance = fa.get_factor_variance()
             fa_scores = fa.transform(data_for_analysis)
             fa_scores_df = pd.DataFrame(fa_scores, columns=[f'Factor_{i+1}_Score' for i in range(fa_scores.shape[1])])
             fa_scores_df = fa_scores_df.set_index(data_for_analysis.index)
         except Exception as e:
             print(f"Error during Factor Analysis fit or transformation: {e}")
             scree_fig_fa.update_layout(title=dict(text=f"<b>Factor Analysis Scree Plot: Error ({e}).</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             variance_fig_fa.update_layout(title=dict(text=f"<b>Factor Analysis Explained Variance: Error ({e}).</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             heatmap_fig_fa.update_layout(title=dict(text=f"<b>Factor Analysis Loadings Heatmap: Error ({e}).</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
             map_rows = [dbc.Row(dbc.Col(html.Div(f"Error during Factor Analysis: {e}"), width=12))]
             return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows


    # --- Update FA Scree Plot ---
    # Ensure eigenvalues_fa is not None and has data
        if eigenvalues_fa is not None and len(eigenvalues_fa) > 0:
            scree_fig_fa = go.Figure()
            scree_fig_fa.add_trace(go.Scatter(
                x=[f'{i+1}' for i in range(len(eigenvalues_fa))],
                y=eigenvalues_fa,
                mode='lines+markers',
                name='Eigenvalues'
            ))
            scree_fig_fa.add_shape(type="line", x0=-0.5, y0=1, x1=len(eigenvalues_fa)-0.5, y1=1,
                                   line=dict(color="Red", width=2, dash="dash"), name='Kaiser Criterion (Eigenvalue > 1)')
    
            scree_fig_fa.update_layout(
                title=dict(text='<b>Factor Analysis Scree Plot (Eigenvalues)</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
                xaxis_title='Factor Number',
                yaxis_title='Eigenvalue',
                showlegend=False,
                margin={"r":0,"t":40,"l":0,"b":0}
            )
        else:
             scree_fig_fa = go.Figure().update_layout(title=dict(text="<b>Factor Analysis Scree Plot: No eigenvalues calculated.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


    # --- Update FA Variance Plot ---
    # Ensure fa_variance has expected structure and enough data for valid_n_factors
        if fa_variance is not None and len(fa_variance) > 2 and len(fa_variance[1]) >= valid_n_factors and len(fa_variance[2]) >= valid_n_factors:
            variance_fig_fa = go.Figure()
            variance_fig_fa.add_trace(go.Bar(
                x=[f'Factor {i+1}' for i in range(valid_n_factors)],
                y=fa_variance[1][:valid_n_factors], # Proportion of variance explained
                name='Proportion<br>Explained Variance' # Legend text on two lines
            ))
            variance_fig_fa.add_trace(go.Scatter(
                x=[f'Factor {i+1}' for i in range(valid_n_factors)],
                y=fa_variance[2][:valid_n_factors], # Cumulative proportion
                mode='lines+markers',
                name='Cumulative<br>Explained Variance' # Legend text on two lines
            ))
            variance_fig_fa.update_layout(
                title=dict(text=f'<b>Factor Analysis Explained Variance ({valid_n_factors} Factors)</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
                xaxis_title='Factor Number',
                yaxis_title='Proportion of Variance',
                legend_title='Variance Type',
                margin={"r":0,"t":40,"l":0,"b":0},
                legend=dict( # Move legend inside the plot
                    x=0.95, # X position (0 to 1)
                    y=0.95, # Y position (0 to 1)
                    xanchor='right', # Anchor point for x
                    yanchor='top', # Anchor point for y
                    bgcolor='rgba(255, 255, 255, 0.5)' # Optional: add background for readability
                )
            )
        else:
             variance_fig_fa = go.Figure().update_layout(title=dict(text="<b>Factor Analysis Explained Variance: Variance data not available or incomplete.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


    # --- Update FA Loadings Heatmap ---
        heatmap_fig_fa = go.Figure()
        if fa_loadings is not None and not data_for_analysis.empty and fa_loadings.shape[0] == len(data_for_analysis.columns) and fa_loadings.shape[1] == valid_n_factors:
             loadings_df_fa = pd.DataFrame(
                 fa_loadings,
                 index=data_for_analysis.columns,
                 columns=[f'Factor {i+1}' for i in range(fa_loadings.shape[1])]
             )
             heatmap_fig_fa = go.Figure(data=go.Heatmap(
                                z=loadings_df_fa.values,
                                x=loadings_df_fa.columns,
                                y=loadings_df_fa.index,
                                colorscale='RdBu',
                                zmin=-1,
                                zmax=1,
                                colorbar=dict(
                                    title='<b>Loading Value</b>',
                                    titleside='right'
                                )))
            
             heatmap_fig_fa.update_layout(
                 title=dict(text='<b>Factor Analysis Loadings Heatmap</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
                 xaxis_title='Factor',
                 yaxis_title='Variable',
                 margin={"r":0,"t":40,"l":0,"b":0}
             )
        else:
            heatmap_fig_fa.update_layout(title=dict(text="<b>Factor Analysis Loadings Heatmap: Loadings data not available or shape mismatch.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
        

    # --- Generate Factor Score Maps ---
        map_rows = []
        # Check if original df and coordinates are available and fa_scores_df is not empty
        if df.empty or 'x_utm' not in df.columns or 'y_utm' not in df.columns or data_for_analysis.empty or fa_scores_df.empty:
             print("Error: Not enough data or missing coordinates in original df or fa_scores_df for factor score maps.")
             # Return empty maps and a message
             map_rows = [dbc.Row(dbc.Col(html.Div("Not enough data or missing coordinates for factor score maps."), width=12))]
             return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
    
    
        # Add x_utm and y_utm to the scores dataframe using indices from data_for_analysis
        fa_scores_df_with_coords = fa_scores_df.copy()
        # Ensure indices match before adding coordinates
        if len(data_for_analysis) == len(fa_scores_df_with_coords):
            try:
                fa_scores_df_with_coords['x_utm'] = df.loc[data_for_analysis.index, 'x_utm'].values
                fa_scores_df_with_coords['y_utm'] = df.loc[data_for_analysis.index, 'y_utm'].values
            except Exception as e:
                print(f"Error adding coordinates to factor scores dataframe: {e}")
                map_rows = [dbc.Row(dbc.Col(html.Div(f"Error adding coordinates for factor score maps: {e}"), width=12))]
                return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows


        # Create GeoDataFrame from the scores dataframe with added coordinates
        try:
            gdf_fa_scores = gpd.GeoDataFrame(
                fa_scores_df_with_coords,
                geometry=gpd.points_from_xy(fa_scores_df_with_coords.x_utm, fa_scores_df_with_coords.y_utm),
                crs="EPSG:32721"  # Assuming UTM zone 21S, adjust if needed
            )
        except Exception as e:
            print(f"Error creating GeoDataFrame for factor scores: {e}")
            map_rows = [dbc.Row(dbc.Col(html.Div(f"Error creating geographic data for factor scores: {e}"), width=12))]
            return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
        
        # Check if gdf_fa_scores is empty after creation
        if gdf_fa_scores.empty:
             print("Error: GeoDataFrame for factor scores is empty.")
             map_rows = [dbc.Row(dbc.Col(html.Div("No geographic data for factor scores."), width=12))]
             return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
        
        
        # Convert coordinates to numpy array for interpolation
        points = np.array([gdf_fa_scores.geometry.x, gdf_fa_scores.geometry.y]).T
        
        # Create map for each factor score
        map_cols = []
        for i in range(valid_n_factors):
            factor_score_col = f'Factor_{i+1}_Score'
            if factor_score_col in gdf_fa_scores.columns: # Check if the factor score column exists
                values = gdf_fa_scores[factor_score_col].values # Use values from gdf_fa_scores
        
                # Check if there are enough points for interpolation (at least 4 points and variation in both x and y)
                if len(points) < 4 or len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
                    print(f"Warning: Not enough unique points for interpolation for Factor {i+1}.")
                    map_fig = go.Figure().update_layout(title=dict(text=f'<b>Factor {i+1} Score Map: Not enough data for interpolation.</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
                else:
                    # Create a regular grid for interpolation (adjust density for performance)
                    grid_density = 100
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    xi = np.linspace(x_min, x_max, grid_density)
                    yi = np.linspace(y_min, y_max, grid_density)
                    xi, yi = np.meshgrid(xi, yi)
        
                    # Interpolation using the cubic method
                    # Handle potential errors during interpolation
                    try:
                        grid_values = griddata(points, values, (xi, yi), method='cubic')
        
                        # Create contour map for the factor score
                        map_fig = go.Figure(data=go.Contour(
                            z=grid_values,
                            x=xi[0, :],
                            y=yi[:, 0],
                            ncontours=25,
                            colorscale='RdBu', # Use a divergent colorscale for scores
                            contours=dict(
                                coloring='fill',
                                showlabels=False
                            ),
                            hoverinfo='z',
                            hovertemplate='<b>%{z:.2f}</b><extra></extra>',
                            colorbar=dict(
                                title=f'<b>Factor {i+1} Score</b>',
                                titleside='right'
                            )
                        ))
            
                        map_fig.update_layout(
                            title=dict(text=f'<b>Interpolated Factor {i+1} Score Map</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
                             xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
                             yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, showline=False, ticks=''),
                            height=400
                        )
                    except Exception as e:
                         print(f"Error during interpolation for Factor {i+1}: {e}")
                         map_fig = go.Figure().update_layout(title=dict(text=f'<b>Factor {i+1} Score Map: Error during interpolation.</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
            
            else:
                 # Return empty figure if the factor score column doesn't exist (shouldn't happen with correct n_factors)
                 map_fig = go.Figure().update_layout(title=dict(text=f'<b>Factor {i+1} Score Map: Data not available.</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
            
            
            map_cols.append(dbc.Col(dcc.Graph(figure=map_fig), width=6)) # Arrange maps in two columns
        
        # Group maps into rows (two maps per row)
        
        for i in range(0, len(map_cols), 2):
            map_rows.append(dbc.Row(map_cols[i:i+2], className="mb-4"))
        
        # else:
        #  print("Error: Index mismatch when adding coordinates to factor scores.")
        #  map_rows = [dbc.Row(dbc.Col(html.Div("Index mismatch when adding coordinates for factor score maps."), width=12))]

        
        return scree_fig_fa, variance_fig_fa, heatmap_fig_fa, map_rows
        
