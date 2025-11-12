from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from shared_data import survey_df, survey_xi, survey_yi, survey_zi, dx_survey, dy_survey, decl, incl, add_observatory_markers
from shared_data import create_data_mask, extrapolate_nans, apply_mask_to_data, reduction_to_pole_improved

# Use the pre-loaded data from shared_data.py
df = survey_df

# Use the pre-calculated grid data from shared_data.py
xi = survey_xi
yi = survey_yi
tf = survey_zi

# Use pre-calculated grid spacing from shared_data.py
dx = dx_survey
dy = dy_survey

# Check for NaN values before processing
print(f"NaN values in original tf: {np.isnan(tf).sum()}")
print(f"Original data range: {np.nanmin(tf):.1f} to {np.nanmax(tf):.1f} nT")
print(f"Original data mean: {np.nanmean(tf):.1f} nT")

# Create mask for non-null data points using shared function
data_mask = create_data_mask(tf)
print(f"Valid data points: {np.sum(data_mask)} out of {tf.size}")

# Extrapolate NaNs with nearest neighbours algorithm using shared function
tf_clean = extrapolate_nans(tf)

# Improved reduction to pole calculation with better parameters using shared function
tf_red_improved = reduction_to_pole_improved(
    tf_clean, dx, dy, incl, decl,
    apply_tapering=True,
    apply_wiener_filter=True,
    wiener_noise_level=1e-5,  # Lower noise level for less aggressive filtering
    apply_smoothing=True,
    sigma=0.5,  # Less smoothing to preserve details
    alpha=0.2   # More tapering for better edge handling
)

print(f"Improved reduced data range: {np.nanmin(tf_red_improved):.1f} to {np.nanmax(tf_red_improved):.1f} nT")
print(f"Improved reduced data mean: {np.nanmean(tf_red_improved):.1f} nT")

# Apply mask to reduced field for display using shared function
tf_red_masked_improved = apply_mask_to_data(tf_red_improved, data_mask)

# Calculate RMS error (using cleaned data for comparison)
tf_err_improved = tf_clean - tf_red_improved

# Apply mask to error for display using shared function
tf_err_masked_improved = apply_mask_to_data(tf_err_improved, data_mask)

# Calculate RMS error only on valid data points
valid_errors_improved = tf_err_improved[data_mask]
rms_error_improved = np.sqrt(np.mean(valid_errors_improved**2))

print(f"Improved RMS Error: {rms_error_improved:.2f} nT")

# Create coordinate arrays for hover information
xx, yy = np.meshgrid(xi[0], yi[:,0])

# plotting gridded data
def create_grid_plot():
    # Create figure for reduced field
    fig = go.Figure()
    
    # Add contour plot for reduced field (using masked data)
    fig.add_trace(
        go.Contour(
            z=tf_red_masked_improved,
            x=xi[0],
            y=yi[:,0],
            colorscale='Viridis',
            colorbar=dict(
                title='B_red (nT)',
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
                '<b>Reduced Field</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                'Field: %{z:.1f} nT<br>' +
                '<extra></extra>'
            ),
            hoverinfo='all'
        )
    )
    
    # Add original data points for context
    fig.add_trace(
        go.Scatter(
            x=df['Longitude (deg)'],
            y=df['Latitude (deg)'],
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                symbol='circle',
                line=dict(width=1, color='white')
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
    fig = add_observatory_markers(fig)
    
    # Configure layout for reduced field
    fig.update_layout(
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
            text='<b>Reduced to Pole Magnetic Field </b>',
            x=0.5,
            font=dict(size=16)
        ),
        # Add hover configuration
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
       
    return fig

def create_error_map():
    """Create a mini map showing RMS error between surveyed and reduced field"""
    fig = go.Figure()
    
    # Add contour plot for error (using masked data)
    fig.add_trace(
        go.Contour(
            z=tf_err_masked_improved,
            x=xi[0],
            y=yi[:,0],
            colorscale='RdBu_r',
            colorbar=dict(
                title='Error (nT)',
                title_font={'size': 10},
                tickfont={'size': 8}
            ),
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont={'size': 8},
            ),
            # Enhanced hover information for error map
            hovertemplate=(
                '<b>Field Error</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                'Error: %{z:.1f} nT<br>' +
                '<extra></extra>'
            ),
            hoverinfo='all'
        )
    )
    
    # Create coordinate arrays for all grid points
    xx, yy = np.meshgrid(xi[0], yi[:,0])
    
    # Create mask for large errors (only on valid data points)
    large_errors_mask = (np.abs(tf_err_improved) > rms_error_improved) & data_mask
    
    # Get coordinates and error values for large errors
    large_error_coords_x = xx[large_errors_mask]
    large_error_coords_y = yy[large_errors_mask]
    large_error_values = tf_err_improved[large_errors_mask]
    
    # Add scatter points for grid points with larger errors
    if len(large_error_coords_x) > 0:
        fig.add_trace(
            go.Scatter(
                x=large_error_coords_x,
                y=large_error_coords_y,
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    symbol='x',
                    line=dict(width=1, color='darkred')
                ),
                showlegend=False,
                hovertemplate=(
                    '<b>Large Error Point</b><br>' +
                    'Longitude: %{x:.6f}°<br>' +
                    'Latitude: %{y:.6f}°<br>' +
                    'Error: %{text:.1f} nT<br>' +
                    '<extra></extra>'
                ),
                text=large_error_values,
                hoverinfo='text'
            )
        )
    
    # Add original data points as reference
    fig.add_trace(
        go.Scatter(
            x=df['Longitude (deg)'],
            y=df['Latitude (deg)'],
            mode='markers',
            marker=dict(
                size=3,
                color='green',
                symbol='circle',
                opacity=0.6
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
    
    fig.update_layout(
        title=dict(
            text=f'RMS Error Map (RMS: {rms_error_improved:.2f} nT)',
            x=0.5,
            font=dict(size=12, weight='bold')
        ),
        xaxis=dict(
            title='Longitude',
            title_font={'size': 10},
            tickfont={'size': 8}
        ),
        yaxis=dict(
            title='Latitude',
            title_font={'size': 10},
            tickfont={'size': 8}
        ),
        width=860,
        height=580,
        margin=dict(t=40, b=40, l=40, r=40),
        showlegend=True,
        # Add hover configuration
        hoverlabel=dict(
            bgcolor="white",
            font_size=10,
            font_family="Arial"
        )
    )
    
    return fig

layout = html.Div([
    html.H2("Reduction to the Pole (RTP)", className="mb-4"),
    dcc.Markdown(r"""This tab shows the magnetic data after apply an **improved reduction to the pole transformation**. 
           The enhanced algorithm includes better stabilization and filtering to produce more realistic magnetic field values.
           This technique maintains the regional magnetic field while transforming anomalies, resulting in more physically 
           accurate results."""),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(" RTP Key Features:"),
                dbc.CardBody([
                    dcc.Markdown(r"""                               
- **Wiener filtering** for frequency-domain stabilization
- **Regional field preservation** to maintain realistic field magnitudes  
- **Controlled filter amplification** to prevent unrealistic values
- **Enhanced edge tapering** to reduce boundary effects
- **Optimized smoothing** to preserve geological features."""),
                    html.P(html.Strong('RTP Summary')),              
                    html.Ul([
                        html.Li(f"Inclination: {incl}°"),
                        html.Li(f"Declination: {decl}°"),
                        html.Li(f"Original field range: {np.nanmin(tf):.1f} to {np.nanmax(tf):.1f} nT"),
                        html.Li(f"Reduced field range: {np.nanmin(tf_red_improved):.1f} to {np.nanmax(tf_red_improved):.1f} nT"),
                        html.Li(f"Regional field preserved: {np.nanmean(tf):.1f} nT"),
                        html.Li(f"RMS Error: {rms_error_improved:.2f} nT"),
                        html.Li(f"Valid data points: {np.sum(data_mask)}/{tf.size}"),
                    ])
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("RMS Error: Original vs Reduced Field"),
                dbc.CardBody([
                    dbc.Row([           
                        dbc.Col([
                            dcc.Graph(
                                figure=create_error_map(),
                                style={'height': '560px'}
                            )
                        ], width=12)
                    ])
                ])
            ])
        ], width=8),
    ]),
    
    html.Br(),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Reduced Field Map"),
                dbc.CardBody([
                    dcc.Markdown("*Hover over the contour plot to see exact magnetic field values and coordinates at any point.*", 
                                style={'margin-bottom': '10px'}),
                    dcc.Graph(
                        figure=create_grid_plot(),
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