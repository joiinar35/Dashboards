"""
Data Visualization page and callbacks for interactive geochemical data exploration.
"""

import numpy as np
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

# Import from shared_data module in the same directory
try:
    from shared_data import df, gdf, column_title_map, numeric_cols
except ImportError:
    # Fallback: define the data directly (for standalone testing)
    import pandas as pd
    import geopandas as gpd
    
    df = pd.DataFrame()
    gdf = pd.DataFrame()
    column_title_map = {}
    numeric_cols = []

# Helper: get element columns for dropdown (robust to missing columns)
def get_element_columns():
    """Return columns representing elements for dropdown selection."""
    if df.empty:
        return []
    
    if 'ba_ppm' in df.columns and 'zn_ppm' in df.columns:
        start = df.columns.get_loc('ba_ppm')
        end = df.columns.get_loc('zn_ppm') + 1
        return df.columns[start:end]
    
    # fallback: use numeric_cols if defined, or all numeric columns
    if numeric_cols and len(numeric_cols) > 0:
        return numeric_cols
    
    return df.select_dtypes(include=[np.number]).columns.tolist()

element_columns = get_element_columns()
dropdown_options = [
    {'label': column_title_map.get(col, col), 'value': col}
    for col in element_columns
]

# Layout for Data Visualization page
data_viz_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("This tab provides interactive visualizations of the geochemical data from a given geographical location and sample set."),
            html.P("Select an element to explore its distribution and correlations."),
            html.P(html.Ul([
                html.Li([
                    "The ", html.Strong("Distribution of Selected Element"), " plot shows the distribution using violin and box plot."
                ]),
                html.Li([
                    "The ", html.Strong("Interpolated Contour Map"), " visualizes the spatial distribution with interpolated contours."
                ]),
                html.Li([
                    "The ", html.Strong("Correlation Matrix"), " heatmap shows the correlation matrix for all geochemical elements."
                ])
            ])),
            html.P("The white dots mark the location of the samples in the map"),
        ], className="explanation-text"), width=12),
    ]),
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Menu", className="mb-3"),
            dcc.Dropdown(
                id='column-dropdown',
                options=dropdown_options,
                value=element_columns[0] if len(element_columns) > 0 else None,
                clearable=False,
                style={"margin-bottom": "1rem"}
            ),
            
        # DISCLAIMER BOX
        dbc.Card([
    dbc.CardHeader("Data Source", className="text-center fw-bold"),
    dbc.CardBody([
        html.P("The data used in this dashboard comes from the 'Inventario Minero del Uruguay', which is freely available in the DINAMIGE catalog hosted on the GeoNetwork of MIEM",
               className="small mb-1"),  # Añade 'small' aquí
        html.A(
            "(https://geonetwork.miem.gub.uy/)",
            href="https://geonetwork.miem.gub.uy/",
            target="_blank",
            className="text-decoration-none small"  # Añade 'small' aquí también
        )
    ], className="small")  # O aplica a todo el CardBody
], className="mt-3 border-info", style={"border-width": "2px"}),
            
            html.Hr(),
            html.Div(id='controls')
        ], width=2, className="sidebar"),
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='contour-map', config={"displayModeBar": False}), width=12),
            ], className="graph-container"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='violin-boxplot-plot', config={"displayModeBar": False}), width=12),
            ], className="graph-container"),
            dbc.Row([
                dbc.Col(dcc.Graph(id='correlation-matrix', config={"displayModeBar": False}), width=12),
            ], className="graph-container"),
        ], width=10)
    ])
], fluid=True)

# --- Callbacks ---
def data_viz_callbacks(app):
    """Register all callbacks for the data visualization page."""
    
    @app.callback(
        Output('contour-map', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_contour_map(selected_column):
        """Interactive map of the selected element using spatial interpolation (contour plot)."""
        # Check for required columns and data
        if (
            df.empty or
            selected_column is None or
            'x_utm' not in df.columns or
            'y_utm' not in df.columns or
            selected_column not in df.columns or
            df[selected_column].isnull().all()
        ):
            return go.Figure().update_layout(
                title=dict(
                    text="<b>Interpolated Contour Map: Not enough data.</b>",
                    x=0.5, xanchor="center", y=0.9, yanchor="top",
                    font=dict(size=16, color="black", family="Arial"))
            )
        
        # Clean data
        mask = df[['x_utm', 'y_utm', selected_column]].dropna()
        if mask.empty:
            return go.Figure().update_layout( width=800, height=600,
                title=dict(
                    text="<b>Interpolated Contour Map: No valid sample locations.</b>",
                    x=0.5, xanchor="center", y=0.9, yanchor="top",
                    font=dict(size=16, color="black", family="Arial"))
            )
        
        x = mask['x_utm'].values
        y = mask['y_utm'].values
        values = mask[selected_column].values
        
        # Interpolation grid
        grid_density = 100
        try:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            if x_min == x_max or y_min == y_max:
                raise ValueError("Not enough spatial spread for interpolation.")
            
            xi = np.linspace(x_min, x_max, grid_density)
            yi = np.linspace(y_min, y_max, grid_density)
            xi, yi = np.meshgrid(xi, yi)
            grid_values = griddata((x, y), values, (xi, yi), method='cubic')
            
            # Remove negative interpolations
            if grid_values is not None:
                grid_values = np.where(grid_values < 0, 0, grid_values)
        except Exception as e:
            return go.Figure().update_layout(
                title=dict(
                    text="<b>Interpolated Contour Map: Interpolation failed.</b>",
                    x=0.5, xanchor="center", y=0.9, yanchor="top",
                    font=dict(size=16, color="black", family="Arial"))
            )

        title = column_title_map.get(selected_column, selected_column)
        fig = go.Figure(data=go.Contour(
            z=grid_values,
            x=xi[0, :],
            y=yi[:, 0],
            ncontours=25,
            colorscale='Viridis',
            contours=dict(coloring='fill', showlabels=False),
            hoverinfo='z',
            hovertemplate='<b>%{z:.1f} ppm</b><extra></extra>',
            colorbar=dict(title=f'<b>{title}</b>', titleside='right')
        ))
        
        # Overlay sample locations (dots)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(color='white', size=4, line=dict(width=0.5, color='black')),
            name='Samples',
            hoverinfo='skip',
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>Interpolated Contour Map - {title}</b>",
                x=0.5, xanchor="center", y=0.9, yanchor="top",
                font=dict(size=16, color="black", family="Arial")),
            xaxis=dict(
                showticklabels=False, showgrid=False, zeroline=False, 
                showline=False, ticks=''),
            yaxis=dict(
                showticklabels=False, showgrid=False, zeroline=False, 
                showline=False, ticks=''),
            height=800,
            margin=dict(l=40, r=20, t=50, b=30),
        )
        return fig

    @app.callback(
        Output('violin-boxplot-plot', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_violin_boxplot(selected_column):
        """Violin & Box plot for the selected element."""
        if (df.empty or selected_column is None or selected_column not in df.columns or
            df[selected_column].dropna().empty):
            return go.Figure().update_layout(
                title=dict(
                    text="<b>Distribution Plots: Not enough data.</b>",
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")
                )
            )
        
        clean_vals = df[selected_column].dropna()
        clean_vals = clean_vals[np.isfinite(clean_vals)]
        
        if clean_vals.empty:
            return go.Figure().update_layout(
                title=dict(
                    text="<b>Distribution Plots: No valid values.</b>",
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")
                )
            )
        
        title = column_title_map.get(selected_column, selected_column)
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                f'<b>Violin Plot of {title}</b>',
                f'<b>Box Plot of {title}</b>'
            )
        )
        
        fig.add_trace(
            go.Violin(y=clean_vals, name='Violin', box_visible=True, 
                     meanline_visible=True, line_color='royalblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Box(y=clean_vals, name='Boxplot', marker_color='indianred'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text=f"<b>Distribution of {title}</b>",
            showlegend=False,
            height=400,
            title_x=0.5,
            title_y=1,
            margin=dict(l=40, r=20, t=50, b=30),
        )
        return fig

    @app.callback(
        Output('correlation-matrix', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_full_correlation_matrix(_):
        """Correlation matrix for all available numeric columns (elements)."""
        if df.empty:
            return go.Figure().update_layout(
                title=dict(
                    text="<b>Correlation Matrix of Geochemical Elements</b>",
                    x=0.5, y=1, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")
                )
            )
        
        elementos = df.select_dtypes(include=[np.number])
        for col in ['x_utm', 'y_utm']:
            if col in elementos.columns:
                elementos = elementos.drop(columns=[col])
        
        # Use only columns with at least some non-null values
        elementos = elementos.dropna(axis=1, how='all')
        
        # Sample rows if too large (performance)
        if elementos.shape[0] > 1000:
            elementos = elementos.sample(n=1000, random_state=42)
        
        if elementos.shape[1] == 0:
            return go.Figure().update_layout(
                title=dict(
                    text="<b>No numeric geochemical columns available for correlation.</b>",
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")
                )
            )
        
        corr_matrix = elementos.corr().round(2)
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[column_title_map.get(c, c) for c in corr_matrix.columns],
            y=[column_title_map.get(c, c) for c in corr_matrix.index],
            colorscale='RdBu',
            zmin=-1, zmax=1,
            hoverongaps=False,
            hovertemplate='Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>',
            colorbar=dict(title='<b>Correlation Coefficient</b>', titleside='right')
        ))
        
        # Annotations: show values on the heatmap
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                font_color = 'white' if abs(value) > 0.7 else 'black'
                annotations.append(
                    dict(
                        x=column_title_map.get(corr_matrix.columns[j], corr_matrix.columns[j]),
                        y=column_title_map.get(corr_matrix.index[i], corr_matrix.index[i]),
                        text=f'{value:.2f}',
                        showarrow=False,
                        font=dict(color=font_color, size=10),
                        bgcolor='rgba(255,255,255,0.5)' if abs(value) < 0.3 else 'rgba(0,0,0,0)'
                    )
                )
        
        fig.update_layout(
            title=dict(
                text='<b>Correlation Matrix of Geochemical Elements</b>',
                x=0.5, y=0.9, xanchor="center", yanchor="top",
                font=dict(size=16, color="black", family="Arial")),
            xaxis=dict(title='Elements', tickangle=-45),
            yaxis=dict(title='Elements'),
            annotations=annotations,
            height=600,
            margin=dict(l=100, r=50, t=80, b=100),
            title_x=0.5,
            title_y=0.9
        )
        return fig

# Nueva función para lazy loading (compatible con app.py optimizado)
def register_data_viz_callbacks(app):
    """Register callbacks for lazy loading support."""
    return data_viz_callbacks(app)
