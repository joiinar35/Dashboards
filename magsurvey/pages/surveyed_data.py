from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load data
df = pd.read_csv('data/magnetometria2.csv')
df.set_index('station', inplace=True)

# Rename columns
df = df.rename(columns={
    'Latitude': 'Latitude (deg)',
    'Longitude': 'Longitude (deg)'
})

# Create radian columns and set to 5 decimal places
# df['Latitude (rad)'] = np.radians(df['Latitude (deg)']).round(5)
# df['Longitude (rad)'] = np.radians(df['Longitude (deg)']).round(5)

# Create grid data for plotting
def create_grid_plot():
    # Extract coordinates and values
    x = df['Longitude (deg)'].values
    y = df['Latitude (deg)'].values
    z = df['B(nT)'].values
    
    # Create a regular grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate using simple linear interpolation (more robust)
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    # Flatten the grid for Densitymapbox
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
            #colorscale='Viridis',
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
            name='',
            text=[f'Station: {idx}<br>Longitude: {x:.5f}°<br>Latitude: {y:.5f}°<br>B: {z:.1f} nT' 
                  for idx, x, y, z in zip(df.index, df['Longitude (deg)'], df['Latitude (deg)'], df['B(nT)'])],
            hovertemplate='%{text}<extra></extra>'
        )
    )
    # Add white square for Observatory Building
    observatory_lat = -34.33344
    observatory_lon = -54.71229
    square_size = 0.00002  # Adjust this value to change square size
    
    sensor_hut_lat = -34.33305
    sensor_hut_lon = -54.71218
    
    # Create square coordinates
    square_lats = [
        observatory_lat - square_size,
        observatory_lat - square_size,
        observatory_lat + square_size,
        observatory_lat + square_size,
        observatory_lat - square_size,
        None  # Break for separate label
    ]
    square_lons = [
        observatory_lon - square_size,
        observatory_lon + square_size,
        observatory_lon + square_size,
        observatory_lon - square_size,
        observatory_lon - square_size,
        None  # Break for separate label
    ]
    
    # Add white square
    fig.add_trace(
        go.Scatter(
            x=square_lons,
            y=square_lats,
            mode='lines+markers',
            line=dict(color='white', width=3),
            marker=dict(size=0),  # Hide markers for clean lines
            name='Observatory Building',
            hoverinfo='text',
            text='Observatory Building',
            hovertemplate='Observatory Building<extra></extra>'
        )
    )
    
    # Add label for Observatory Building
    fig.add_trace(
        go.Scatter(
            x=[observatory_lon],
            y=[observatory_lat + square_size * 1.5],  # Position label above the square
            mode='text',
            text=['Observatory Building'],
            textfont=dict(
                size=14,
                color='white',
                family='Arial, bold'
            ),
            showlegend=False,
            hoverinfo='none'
        )
    )
    
    # Add circle mark
    fig.add_trace(
        go.Scatter(
            x=[sensor_hut_lon],
            y=[sensor_hut_lat],
            mode='markers',
            marker=dict(
                size=12,
                color='yellow',
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name='Sensor Hut',
            hoverinfo='text',
            text='Sensor Hut',
            hovertemplate='Sensor Hut<extra></extra>'
        )
    )
    
    # Add label for Sensor Hut
    fig.add_trace(
        go.Scatter(
            x=[sensor_hut_lon],
            y=[sensor_hut_lat - 0.00005],  # Position label below the circle
            mode='text',
            text=['Sensor Hut'],
            textfont=dict(
                size=12,
                color='yellow',
                family='Arial, bold'
            ),
            showlegend=False,
            hoverinfo='none'
        )
    )
   
    
    # Configure layout 
    fig.update_layout(
        title=dict(
            text='<b>Magnetic Field (Total Intensity)</b>',
            x=0.5,
            font=dict(size=16)),
        # width=1000,  # Slightly wider for better map view
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