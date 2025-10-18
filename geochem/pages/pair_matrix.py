"""
Pair Matrix page and callbacks for comprehensive geochemical element relationships visualization.
"""

from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Import from shared_data module
try:
    from shared_data import df, column_title_map, numeric_cols
except ImportError:
    # Fallback: define empty data structures
    df = pd.DataFrame()
    column_title_map = {}
    numeric_cols = []

# Helper function to get element columns
def get_element_columns():
    """Return columns representing elements for selection."""
    if df.empty:
        return []
    
    # Try to get element columns from ba_ppm to zn_ppm
    if 'ba_ppm' in df.columns and 'zn_ppm' in df.columns:
        start_idx = df.columns.get_loc('ba_ppm')
        end_idx = df.columns.get_loc('zn_ppm') + 1
        return df.columns[start_idx:end_idx].tolist()
    
    # Fallback to numeric columns excluding coordinates
    element_cols = [col for col in numeric_cols if col not in ['x_utm', 'y_utm']]
    return element_cols

# Get element columns for dropdown
element_columns = get_element_columns()

# Layout for Pair Matrix page
pair_matrix_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("The Pair Matrix provides a comprehensive view of relationships between multiple geochemical elements:"),
            html.Ul([
                html.Li([html.Strong("Upper Triangle: "), "Scatter plots showing relationships between element pairs"]),
                html.Li([html.Strong("Diagonal: "), "Histograms with KDE showing distribution of each element"]),
                html.Li([html.Strong("Lower Triangle: "), "2D Density plots showing density relationships"]),
                html.Li([html.Strong("Correlation coefficients (r) "), "are displayed in the upper triangle"])
            ]),
            html.P("This interactive visualization helps identify patterns, correlations, and potential outliers in your geochemical data.")
        ], className="explanation-text"), width=12),
    ]),

    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Pair Matrix Controls", className="mb-6"),
            
            # Pair Matrix Controls
            html.H5("Pair Matrix", className="mt-4 mb-2"),
            html.Label("Sample Size (for performance):"),
            dcc.Dropdown(
                id='sample-size-dropdown',
                options=[
                    {'label': 'Full Dataset', 'value': 'full'},
                    {'label': '100 samples', 'value': 100},
                    {'label': '200 samples', 'value': 200},
                    {'label': '500 samples', 'value': 500}
                ],
                value=200,
                clearable=False,
                disabled=df.empty
            ),
            html.Label("Select Elements for Pair Matrix:"),
            dcc.Dropdown(
                id='pair-matrix-element-selector',
                options=[{'label': column_title_map.get(col, col), 'value': col} for col in element_columns],
                value=element_columns[:4] if len(element_columns) >= 4 else element_columns,
                multi=True,
                placeholder="Select elements for pair matrix",
                disabled=df.empty
            ),
            dbc.Button(
                'Generate Pair Matrix', 
                id='generate-pair-matrix-btn', 
                n_clicks=0, 
                color="success", 
                className="mt-2",
                disabled=df.empty
            ),
            
            html.Div([
                html.P("Note: For better performance, limit to 4-6 elements for Pair Matrix.", 
                      className="text-muted small mt-3")
            ]),
            html.Div(id='pair-matrix-controls-placeholder')
        ], width=3, className="sidebar"),

        # Main content
        dbc.Col([
            # Pair Matrix Section
            dbc.Row([
                dbc.Col(html.H4("Pair Matrix", className="mb-3 mt-4"), width=12),
                dbc.Col(dcc.Graph(id='pair-matrix-plot'), width=12, className="graph-container"),
            ], id='pair-matrix-section'),
        ], width=9)
    ])
], fluid=True)

def create_plotly_pair_matrix(elements_df, selected_elements_titles):
    """Create an interactive pair matrix using Plotly."""
    
    elements = elements_df.columns.tolist()
    n_elements = len(elements)
    
    if n_elements == 0:
        return go.Figure()
    
    # Create subplots
    fig = make_subplots(
        rows=n_elements, 
        cols=n_elements,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        subplot_titles=[]  # We'll add titles manually
    )
    
    # Calculate correlations once
    corr_matrix = elements_df.corr()
    
    for i, y_col in enumerate(elements):
        for j, x_col in enumerate(elements):
            row = i + 1
            col = j + 1
            
            # Get clean data for this pair
            data = elements_df[[x_col, y_col]].dropna()
            x_data = data[x_col]
            y_data = data[y_col]
            
            if len(data) < 2:
                # Empty subplot if not enough data
                fig.add_trace(
                    go.Scatter(x=[], y=[], showlegend=False),
                    row=row, col=col
                )
                continue
            
            if i == j:
                # Diagonal: Histogram with KDE
                fig.add_trace(
                    go.Histogram(
                        x=x_data,
                        nbinsx=180,
                        name=f'{selected_elements_titles[i]}',
                        showlegend=False,
                        marker_color='steelblue',
                        opacity=0.5,
                        histnorm='probability density'
                    ),
                    row=row, col=col
                )
                
                # Add KDE line (simulated)
                hist, bin_edges = np.histogram(x_data, bins=180, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers,
                        y=hist,
                        mode='lines',
                        line=dict(color='darkblue', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
            elif i < j:
                # Upper triangle: Scatter plot with correlation
                correlation = corr_matrix.loc[y_col, x_col]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode='markers',
                        marker=dict(
                            size=4,
                            color='darkorange',
                            opacity=0.6,
                            line=dict(width=0)
                        ),
                        showlegend=False,
                        hovertemplate=f'{selected_elements_titles[j]}: %{{x}}<br>{selected_elements_titles[i]}: %{{y}}<extra></extra>'
                    ),
                    row=row, col=col
                )
                
                # Add correlation annotation
                fig.add_annotation(
                    xref=f'x{"" if n_elements == 1 else col}',
                    yref=f'y{"" if n_elements == 1 else row}',
                    x=0.95, y=0.95,
                    xanchor='right',
                    yanchor='top',
                    text=f'r = {correlation:.2f}',
                    showarrow=False,
                    bgcolor='white',
                    bordercolor='gray',
                    borderwidth=1,
                    borderpad=2,
                    opacity=0.8,
                    row=row, col=col
                )
                
            else:
                # Lower triangle: 2D Density plot
                fig.add_trace(
                    go.Histogram2dContour(
                        x=x_data,
                        y=y_data,
                        colorscale='Viridis',
                        showscale=False,
                        ncontours=20,
                        line=dict(width=0),
                        hoverinfo='none'
                    ),
                    row=row, col=col
                )
    
    # Update layout and axes
    fig.update_layout(
        title_text="<b>Pair Matrix - Geochemical Elements Relationships</b>",
        title_x=0.5,
        title_font=dict(size=24, color='black'),
        height=200 * n_elements + 100,  # Dynamic height based on number of elements
        width=200 * n_elements + 100,   # Dynamic width
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    # Set axis labels
    for i, element in enumerate(elements):
        # X-axis labels (bottom row)
        fig.update_xaxes(
            title_text=selected_elements_titles[i],
            row=n_elements, col=i+1,
            title_font=dict(size=10)
        )
        # Y-axis labels (first column)
        fig.update_yaxes(
            title_text=selected_elements_titles[i],
            row=i+1, col=1,
            title_font=dict(size=10)
        )
    
    # Hide axis labels for inner subplots to reduce clutter
    for i in range(1, n_elements + 1):
        for j in range(1, n_elements + 1):
            if i != n_elements:  # Not bottom row
                fig.update_xaxes(showticklabels=False, row=i, col=j)
            if j != 1:  # Not first column
                fig.update_yaxes(showticklabels=False, row=i, col=j)
    
    return fig

def create_simple_pair_matrix(elements_df, selected_elements_titles):
    """Create a simpler pair matrix for better performance with many elements."""
    
    elements = elements_df.columns.tolist()
    n_elements = len(elements)
    
    if n_elements == 0:
        return go.Figure()
    
    # Create scatter plot matrix using plotly express
    fig = px.scatter_matrix(
        elements_df,
        dimensions=elements,
        title="Pair Matrix - Geochemical Elements Relationships",
        labels={col: title for col, title in zip(elements, selected_elements_titles)},
        height=150 * n_elements + 100,
        width=150 * n_elements + 100
    )
    
    # Update marker style
    fig.update_traces(
        diagonal_visible=False,
        showupperhalf=True,
        showlowerhalf=True,
        marker=dict(
            size=3,
            opacity=0.6,
            color='darkorange',
            line=dict(width=0)
        )
    )
    
    # Add histograms on diagonal
    for i in range(len(n.elements)):
        if fig.data[i].name == fig.data[i].legendgroup:  # Diagonal
            fig.data[i].update(
                type='histogram',
                marker_color='steelblue',
                opacity=0.7
            )
    
    fig.update_layout(
        title_x=0.5,
        title_font=dict(size=16, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

# Callbacks for Pair Matrix page
def pair_matrix_callbacks(app):
    """Register all callbacks for the pair matrix page."""

    @app.callback(
        Output('pair-matrix-plot', 'figure'),
        [Input('generate-pair-matrix-btn', 'n_clicks')],
        [
            State('sample-size-dropdown', 'value'),
            State('pair-matrix-element-selector', 'value')
        ]
    )
    def update_pair_matrix(n_clicks, sample_size, selected_elements):
        """Update pair matrix visualization based on user selections."""
        if n_clicks == 0 or not selected_elements:
            return go.Figure().update_layout(
                title=dict(
                    text="Please select elements and click 'Generate Pair Matrix'",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    font=dict(size=16, color='black')
                )
            )

        if df.empty:
            return go.Figure().update_layout(
                title=dict(
                    text="No data available for pair matrix visualization",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    font=dict(size=16, color="black")
                )
            )

        # Select numeric columns and remove coordinate columns
        elements_df = df.select_dtypes(include=['float64', 'int64'])
        elements_df = elements_df.drop(
            columns=[col for col in ['x_utm', 'y_utm'] if col in elements_df.columns]
        )

        # Filter to only selected elements
        try:
            elements_df = elements_df[selected_elements]
        except KeyError as e:
            return go.Figure().update_layout(
                title=dict(
                    text=f"Error: Selected elements not found in data",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    font=dict(size=16, color="red")
                )
            )
        
        # Remove columns with all NaN values
        elements_df = elements_df.dropna(axis=1, how='all')
        
        if elements_df.empty:
            return go.Figure().update_layout(
                title=dict(
                    text="No valid numeric data available for selected elements",
                    x=0.5, y=0.5,
                    xanchor="center", yanchor="middle",
                    font=dict(size=16, color="orange")
                )
            )

        # Handle sampling for performance
        if sample_size != 'full' and len(elements_df) > int(sample_size):
            try:
                elements_df = elements_df.sample(n=int(sample_size), random_state=42)
            except ValueError:
                # If sample size is larger than dataset, use full dataset
                pass

        # Get readable titles for selected elements
        selected_elements_titles = [
            column_title_map.get(col, col) for col in selected_elements 
            if col in elements_df.columns
        ]
        
        # Choose matrix type based on number of elements
        n_elements = len(selected_elements_titles)
        
        if n_elements <= 6:
            # Use custom matrix for better control with few elements
            fig = create_plotly_pair_matrix(elements_df, selected_elements_titles)
        else:
            # Use plotly express for better performance with many elements
            fig = create_simple_pair_matrix(elements_df, selected_elements_titles)
        
        # Add sample count annotation
        fig.add_annotation(
            x=1, y=0,
            xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=f"Samples: {len(elements_df)} | Elements: {n_elements}",
            showarrow=False,
            bgcolor="white",
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            opacity=0.8
        )
        
        return fig