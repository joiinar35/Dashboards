from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from shared_data import survey_df, gradiometer_df, survey_xi, survey_yi, survey_zi, grad_zi, Delta_x, Delta_y, add_observatory_markers

# Use the pre-loaded data from shared_data.py
df = survey_df
gdf = gradiometer_df

# Extract coordinates and values from df
x = df['Longitude (deg)'].values
y = df['Latitude (deg)'].values
z = df['B(nT)'].values

# Extract coordinates and values from gdf
gx = gdf['Longitud (deg)'].values
gy = gdf['Latitud (deg)'].values
gz = gdf['dB (nT)'].values

# Use the pre-calculated grid from shared_data.py
xi = survey_xi
yi = survey_yi
zi = survey_zi
gzi = grad_zi

# Calculate grid spacing using shared constants
dx = Delta_x * abs(x.max()-x.min()) / len(xi[0])
dy = Delta_y * abs(y.max()-y.min()) / len(yi[:,0])

# Calculate gradients 
(dBx, dBy) = np.gradient(zi, dy, dx)

dBx = dBx/Delta_x    # Gradient in X
dBy = dBy/Delta_y    # Gradient in Y

# Calculate different gradient components
total_horizontal_gradient = np.sqrt(dBx**2 + dBy**2)  # Total horizontal gradient

vertical_gradient = gzi

# Create statistics
dBh_avg = np.nanmean(total_horizontal_gradient)
dBh_max = np.nanmax(total_horizontal_gradient)
dBz_avg = np.nanmean(vertical_gradient)
dBz_max = np.nanmax(vertical_gradient)

# Create coordinate arrays for hover information
xx, yy = np.meshgrid(xi[0], yi[:,0])

# Plotting gridded data
def create_grid_plot():
    # Create figure for horizontal gradient
    fig_horizontal = go.Figure()
    
    # Add contour plot for horizontal gradient
    fig_horizontal.add_trace(
        go.Contour(
            z=1e3*total_horizontal_gradient,
            x=xi[0],
            y=yi[:,0],
            colorscale='Viridis',
            colorbar=dict(
                title='dBh (×10⁻³ nT/m)',
                title_font={'size': 14},
                tickfont={'size': 12}
            ),
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont={'color': 'white', 'size': 10},
            ),
            line=dict(
                color='white',
                width=1
            ),
            # Enhanced hover information
            hovertemplate=(
                '<b>Horizontal Gradient</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                'dBh: %{z:.3f} ×10⁻³ nT/m<br>' +
                '<extra></extra>'
            ),
            hoverinfo='all'
        )
    )
    
    # Add original measurement points for context
    fig_horizontal.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(255, 255, 255, 0.7)',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Measurement Points',
            hovertemplate=(
                '<b>Measurement Point</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )
    
    # Add observatory and sensor hut markers using shared function
    fig_horizontal = add_observatory_markers(fig_horizontal)
    
    # Configure layout for horizontal gradient
    fig_horizontal.update_layout(
        xaxis=dict(
            title='Longitude',
            title_font={'size': 14},
            tickfont={'size': 12},
            gridcolor='white',
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Latitude',
            title_font={'size': 14},
            tickfont={'size': 12},
            gridcolor='white',
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        height=1000,
        margin=dict(t=30, b=60, l=60, r=80),
        title=dict(
            text='Total Horizontal Magnetic Gradient',
            x=0.5,
            font=dict(size=16, weight='bold')
        ),
        # Add hover configuration
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Create figure for vertical gradient
    fig_vertical = go.Figure()
    
    # Add contour plot for vertical gradient
    fig_vertical.add_trace(
        go.Contour(
            z=vertical_gradient,
            x=xi[0],
            y=yi[:,0],
            colorscale='Plasma',
            colorbar=dict(
                title='dBz (nT/m)',
                title_font={'size': 14},
                tickfont={'size': 12}
            ),
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont={'color': 'white', 'size': 10},
            ),
            line=dict(
                color='white',
                width=1
            ),
            # Enhanced hover information
            hovertemplate=(
                '<b>Vertical Gradient</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                'dBz: %{z:.3f} nT/m<br>' +
                '<extra></extra>'
            ),
            hoverinfo='all'
        )
    )
    
    # Add original measurement points for context
    fig_vertical.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                size=3,
                color='rgba(255, 255, 255, 0.7)',
                symbol='circle',
                line=dict(width=1, color='black')
            ),
            name='Measurement Points',
            hovertemplate=(
                '<b>Measurement Point</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )
    
    # Add gradiometer measurement points
    fig_vertical.add_trace(
        go.Scatter(
            x=gx,
            y=gy,
            mode='markers',
            marker=dict(
                size=4,
                color='rgba(0, 255, 0, 0.6)',
                symbol='diamond',
                line=dict(width=1, color='darkgreen')
            ),
            name='Gradiometer Points',
            hovertemplate=(
                '<b>Gradiometer Point</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )
    
    # Add observatory and sensor hut markers using shared function
    fig_vertical = add_observatory_markers(fig_vertical)
    
    # Configure layout for vertical gradient
    fig_vertical.update_layout(
        xaxis=dict(
            title='Longitude',
            title_font={'size': 14},
            tickfont={'size': 12},
            gridcolor='white',
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Latitude',
            title_font={'size': 14},
            tickfont={'size': 12},
            gridcolor='white',
            gridwidth=1,
            showgrid=True,
            zeroline=False
        ),
        height=1000,
        margin=dict(t=30, b=60, l=60, r=80),
        title=dict(
            text='Vertical Magnetic Gradient',
            x=0.5,
            font=dict(size=16, weight='bold')
        ),
        # Add hover configuration
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig_horizontal, fig_vertical

# Create the figures
fig_horizontal, fig_vertical = create_grid_plot()

layout = html.Div([
    html.H2("Magnetic Gradients", className="mb-4"),
    dcc.Markdown(r"""This tab displays calculated magnetic gradients from the survey data. This technique helps
                 to identify and locate buried sources of magnetization. **Hover over the plots to see detailed 
                 gradient values and measurement locations.**"""),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Gradient Statistics"),
                dbc.CardBody([
                    html.P(f"Average Horizontal Gradient: {dBh_avg*1e3:.3f} ×10⁻³ nT/m"),
                    html.P(f"Max Horizontal Gradient: {dBh_max*1e3:.3f} ×10⁻³ nT/m"),
                    html.P(f"Average Vertical Gradient: {dBz_avg:.3f} nT/m"),
                    html.P(f"Max Vertical Gradient: {dBz_max:.3f} nT/m"),
                    html.Hr(),
                    html.P("Labels:", style={'fontWeight': 'bold'}),
                    html.Ul([
                        html.Li("White dots: Measurement points"),
                        html.Li("Green diamonds: Gradiometer points"),
                        html.Li("White square: Observatory building"),
                        html.Li("Yellow circle: Sensor hut"),
                    ])
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Key Notes"),
                dbc.CardBody([
                   dcc.Markdown(r"""The maps show the spatial distribution of the total horizontal magnetic gradient
                          dBh and the gradiometric profile showing vertical gradient dBz, respectively.  
                          Both graphs show large anomalies close to the west and southeast edges of the
                          surveyed area.    
                          The main causes of such anomalies are due to buried ferromagnetic objects or the proximitiy
                          to building remains, fences, etc.                                
                          In general there is a good agreement between the three graphs denoting that 
                          the largest magnetic gradients occur in the vertical direction.   
                          The area of the gradiometric survey is slightly smaller than the magnetic one.
                          Despite some large vertical gradients in the area, the planed sensor hut is located over 
                          a low gradient area which makes the site suitable to take acceptable magnetic readings.  
                          Gradiometry is widely used in archaeological and mining  prospection to identify hidden
                          building foundations, buried unexploded ordnance, mineral deposits, etc.  """)       
                ])
            ])      
        ], width=6)           
    ]),
    
    html.Br(),
    
    # First row - Horizontal Gradient
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Total Horizontal Magnetic Gradient"),
                dbc.CardBody([
                    dcc.Markdown("*Hover over the contour plot to see exact horizontal gradient values and coordinates at any point.*", 
                                style={'margin-bottom': '10px'}),
                    dcc.Graph(figure=fig_horizontal,
                              style={
                                  'display': 'block',
                                  'margin-left': 'auto',
                                  'margin-right': 'auto',
                                  'width': '100%'
                              }
                        )
                ])
            ])
        ])
    ]),
    
    html.Br(),
    
    # Second row - Vertical Gradient
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Vertical Magnetic Gradient"),
                dbc.CardBody([
                    dcc.Markdown("*Hover over the contour plot to see exact vertical gradient values and coordinates at any point.*", 
                                style={'margin-bottom': '10px'}),
                    dcc.Graph(figure=fig_vertical,
                              style={
                                  'display': 'block',
                                  'margin-left': 'auto',
                                  'margin-right': 'auto',
                                  'width': '100%'
                              }
                        )
                ])
            ])
        ])
    ])
])