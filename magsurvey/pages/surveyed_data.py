from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shared_data import survey_df, survey_xi, survey_yi, survey_zi, add_observatory_markers

# Use the pre-loaded data from shared_data.py
df = survey_df

def create_grid_plot():
    """Create grid plot using shared data and functions"""
    # Use the pre-calculated grid data from shared_data.py
    xi = survey_xi
    yi = survey_yi
    zi = survey_zi
    
    # Flatten the grid for plotting
    lons_grid = xi.flatten()
    lats_grid = yi.flatten()
    z_grid = zi.flatten()
    
    # Create the plot
    fig = go.Figure()
    
    # Add heatmap (grid data)
    fig.add_trace(
        go.Heatmap(
            x=lons_grid, 
            y=lats_grid,
            z=z_grid,
            colorscale='Viridis',
            name='Magnetic Field',
            colorbar=dict(title="B (nT)"),
            hoverinfo='none'
        )
    )
    
    # Add contour lines
    fig.add_trace(
        go.Contour(
            x=lons_grid, 
            y=lats_grid,
            z=z_grid,
            showscale=False,
            line_width=2,
            contours=dict(
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=10, color='red')
            ),
            name='Contours'
        )
    )
    
    # Add original data points
    fig.add_trace(
        go.Scatter(
            x=df['Longitude (deg)'],
            y=df['Latitude (deg)'],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            name='Survey Stations',
            text=[f'Station: {idx}<br>Longitude: {x:.5f}°<br>Latitude: {y:.5f}°<br>B: {z:.1f} nT' 
                  for idx, x, y, z in zip(df.index, df['Longitude (deg)'], df['Latitude (deg)'], df['B(nT)'])],
            hovertemplate='%{text}<extra></extra>'
        )
    )
    
    # Add observatory and sensor hut markers using shared function
    fig = add_observatory_markers(fig)
    
    # Configure layout 
    fig.update_layout(
        title=dict(
            text='<b>Magnetic Field (Total Intensity)</b>',
            x=0.5,
            font=dict(size=16)),
        height=1000,
        showlegend=False,
        margin=dict(t=30, b=60, l=60, r=80),
    )
    
    return fig

layout = html.Div([
    html.H2("Survey Data", className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Summary"),
                dbc.CardBody([
                    html.P(f"Total Stations: {len(df)}"),
                    html.P(f"Magnetic Field Range: {df['B(nT)'].min():.1f} - {df['B(nT)'].max():.1f} nT"),
                    html.P(f"Latitude Range: {df['Latitude (deg)'].min():.5f} - {df['Latitude (deg)'].max():.5f}°"),
                    html.P(f"Longitude Range: {df['Longitude (deg)'].min():.5f} - {df['Longitude (deg)'].max():.5f}°"),
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Survey Details"),
                dbc.CardBody([
                    dcc.Markdown(r""" **The Problem:**    
                            Magnetic survey of a property of approximately one hectare to
                            evaluate the feasibility of installing a magnetic station on the site.
                            We analyze data from unevenly spaced data points along the study area.
                           The datasets includes a simple magnetic survey plus a gradiometric survey.  
                           The survey was performed with a portable proton magnetometer and the
                           gradiometric one with two vertically stacked Overhauser sensors at 1m of separation""")
                ])
            ])
        ], width=8),
    ]),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Table"),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.reset_index().to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.reset_index().columns],
                        page_size=10,
                        style_table={'overflowX': 'auto'},
                        style_cell={
                            'textAlign': 'left',
                            'padding': '8px',
                            'minWidth': '100px'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'
                        },
                        style_data_conditional=[
                            {
                                'if': {'column_id': 'B(nT)'},
                                'backgroundColor': 'rgb(240, 240, 240)',
                                'fontWeight': 'bold'
                            }
                        ]
                    )
                ])
            ])
        ])
    ]),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Magnetic Field Grid Plot"),
                dbc.CardBody([
                    dcc.Graph(
                        figure=create_grid_plot(), 
                        style={
                            'display': 'block',
                            'margin-left': 'auto',
                            'margin-right': 'auto',
                            'width': '100%'
                        }
                    ),
                    html.P("Red markers show actual survey station locations", 
                          style={'textAlign': 'center', 'fontStyle': 'italic', 'color': 'red', 'marginTop': '10px'})
                ])
            ])
        ])
    ])
])