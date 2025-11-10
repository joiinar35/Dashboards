from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.signal import windows
from scipy.interpolate import griddata

# Load data
df = pd.read_csv('data/magnetometria2.csv')
df.set_index('station', inplace=True)

# Extract coordinates and values from df
x = df['Longitude'].values
y = df['Latitude'].values
z = df['B(nT)'].values

# Create a regular grid
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Constants for the derivative, dx != dy at Earth's surface (it is a Geoid)
Delta_x = 106e3   # m/deg  in N-S
Delta_y = 110e3   # m/deg  in E-W

dx = Delta_x * abs(x.max()-x.min()) / len(xi)
dy = Delta_y * abs(y.max()-y.min()) / len(yi)

# Interpolate using simple linear interpolation (more robust)
tf = griddata((x, y), z, (xi, yi), method='linear')

decl = -11.37
incl = -40.3

def create_data_mask(data, threshold=1e-10):
    """
    Create a mask for non-null data points.
    Returns a boolean mask where True indicates valid (non-null) data points.
    """
    # Create mask for non-NaN and non-zero values
    non_nan_mask = ~np.isnan(data)
    
    # For the original data, also exclude near-zero values if they represent gaps
    non_zero_mask = np.abs(data) > threshold
    
    # Combine masks
    valid_mask = non_nan_mask & non_zero_mask
    
    return valid_mask

def extrapolate_nans(data):
    """
    Extrapolate NaN values in a 2D array using nearest neighbor interpolation.
    """
    if not np.isnan(data).any():
        return data
    
    data_extrapolated = data.copy()
    ny, nx = data.shape
    
    # Create coordinate arrays
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Get valid (non-NaN) points
    valid_mask = ~np.isnan(data)
    valid_points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask]))
    valid_values = data[valid_mask]
    
    # Get NaN points
    nan_points = np.column_stack((x_coords[~valid_mask], y_coords[~valid_mask]))
    
    if len(nan_points) > 0:
        # Use nearest neighbor interpolation for NaN points
        from scipy.interpolate import NearestNDInterpolator
        interpolator = NearestNDInterpolator(valid_points, valid_values)
        nan_values = interpolator(nan_points)
        
        # Fill NaN values
        data_extrapolated[~valid_mask] = nan_values
    
    return data_extrapolated

def apply_mask_to_data(data, mask, fill_value=np.nan):
    """
    Apply mask to data, setting masked values to fill_value.
    """
    masked_data = data.copy()
    masked_data[~mask] = fill_value
    return masked_data

def apply_tukey_tapering(data, alpha=0.2):
    """
    Apply Tukey 2D window to reduce edge effects with larger tapering.
    """
    ny, nx = data.shape
    # Create 1D Tukey windows
    window_x = windows.tukey(nx, alpha)
    window_y = windows.tukey(ny, alpha)
    
    # Create 2D window
    window_2d = np.outer(window_y, window_x)
    
    # Apply window to data
    return data * window_2d

def create_wiener_filter(kx, ky, noise_level=1e-3, signal_level=1.0):
    """
    Create Wiener filter for stabilization in frequency domain.
    """
    k_squared = kx**2 + ky**2
    k = np.sqrt(k_squared)
    
    # Avoid division by zero
    k_squared[k_squared == 0] = 1e-10
    
    # Wiener filter: signal/(signal + noise)
    # For magnetic data, noise increases with frequency
    signal_power = signal_level / (1 + k_squared)
    noise_power = noise_level * k_squared
    
    wiener_filter = signal_power / (signal_power + noise_power)
    
    return wiener_filter

def reduction_to_pole_improved(tf, dx, dy, inc, dec, 
                              apply_tapering=True, 
                              apply_wiener_filter=True,
                              wiener_noise_level=1e-4,
                              apply_smoothing=True,
                              sigma=0.8,
                              alpha=0.2):
    """
    Improved RTP with better stabilization and filtering.
    """
    # First, extrapolate any NaN values
    tf_clean = extrapolate_nans(tf)
    
    # Remove regional field to work with anomalies
    regional_field = np.nanmean(tf_clean)
    tf_anomaly = tf_clean - regional_field
    
    print(f"Regional field: {regional_field:.1f} nT")
    print(f"Anomaly range: {np.nanmin(tf_anomaly):.1f} to {np.nanmax(tf_anomaly):.1f} nT")
    
    # Apply tapering to reduce edge effects
    if apply_tapering:
        tf_processed = apply_tukey_tapering(tf_anomaly, alpha=alpha)
    else:
        tf_processed = tf_anomaly.copy()
    
    # Convert angles to radians
    inc_rad = np.deg2rad(inc)
    dec_rad = np.deg2rad(dec)
    
    # Grid dimensions
    ny, nx = tf_processed.shape
    
    # Calculate spatial frequencies (rad/m)
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    
    # Create frequency meshes
    Kx, Ky = np.meshgrid(kx, ky)
    K_squared = Kx**2 + Ky**2
    K = np.sqrt(K_squared)
    
    # Avoid division by zero
    K_squared[K_squared == 0] = 1e-10
    
    # Direction cosines for geomagnetic field
    X0 = np.cos(inc_rad) * np.cos(dec_rad)
    Y0 = np.cos(inc_rad) * np.sin(dec_rad) 
    Z0 = np.sin(inc_rad)
    
    # Direction cosines for pole (vertical magnetization)
    Xp = 0.0  # At pole, field is vertical
    Yp = 0.0
    Zp = 1.0
    
    # Calculate the RTP filter with stabilization
    with np.errstate(divide='ignore', invalid='ignore'):
        # Original field direction filter
        F_original = 1j * (X0 * Kx + Y0 * Ky) - Z0 * K
        # Pole direction filter  
        F_pole = 1j * (Xp * Kx + Yp * Ky) - Zp * K
        
        # RTP filter
        rtp_filter = F_pole / F_original
        
        # Apply stabilization to avoid extreme values
        max_filter_magnitude = 10.0  # Limit filter amplification
        filter_magnitude = np.abs(rtp_filter)
        rtp_filter = np.where(filter_magnitude > max_filter_magnitude, 
                             rtp_filter * max_filter_magnitude / filter_magnitude, 
                             rtp_filter)
        
        # Handle remaining invalid values
        rtp_filter = np.nan_to_num(rtp_filter, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Apply Wiener filter for additional stabilization
    if apply_wiener_filter:
        wiener_filter = create_wiener_filter(Kx, Ky, noise_level=wiener_noise_level)
        rtp_filter = rtp_filter * wiener_filter
    
    # FFT of processed data
    tf_fft = fft2(tf_processed)
    
    # Apply the stabilized RTP filter
    rtp_fft = tf_fft * rtp_filter
    
    # Inverse FFT to real space
    rtp_anomaly = np.real(ifft2(rtp_fft))
    
    # Add back the regional field
    rtp_result = rtp_anomaly + regional_field
    
    # Apply Gaussian smoothing if enabled
    if apply_smoothing:
        rtp_result = gaussian_filter(rtp_result, sigma=sigma)
    
    return rtp_result

def process_magnetic_data_improved(data, dx, dy, inc, dec, **kwargs):
    """
    Improved main function for processing magnetic data with RTP.
    """
    return reduction_to_pole_improved(data, dx, dy, inc, dec, **kwargs)

# Check for NaN values before processing
print(f"NaN values in original tf: {np.isnan(tf).sum()}")
print(f"Original data range: {np.nanmin(tf):.1f} to {np.nanmax(tf):.1f} nT")
print(f"Original data mean: {np.nanmean(tf):.1f} nT")

# Create mask for non-null data points
data_mask = create_data_mask(tf)
print(f"Valid data points: {np.sum(data_mask)} out of {tf.size}")

# Extrapolate NaNs with nearest neighbours algorithm
tf_clean = extrapolate_nans(tf)

# Improved reduction to pole calculation with better parameters
tf_red_improved = process_magnetic_data_improved(
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

# Apply mask to reduced field for display
tf_red_masked_improved = apply_mask_to_data(tf_red_improved, data_mask)

# Calculate RMS error (using cleaned data for comparison)
tf_err_improved = tf_clean - tf_red_improved

# Apply mask to error for display
tf_err_masked_improved = apply_mask_to_data(tf_err_improved, data_mask)

# Calculate RMS error only on valid data points
valid_errors_improved = tf_err_improved[data_mask]
rms_error_improved = np.sqrt(np.mean(valid_errors_improved**2))

print(f"Improved RMS Error: {rms_error_improved:.2f} nT")

# Create coordinate arrays for hover information
xx, yy = np.meshgrid(xi[0], yi[:,0])

# plotting gridded data
def create_grid_plot():
    # Coordinates for markers
    observatory_lat = -34.33344
    observatory_lon = -54.71229
    sensor_hut_lat = -34.33306
    sensor_hut_lon = -54.71218
    
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
    
    # Add Observatory Building marker
    fig.add_trace(
        go.Scatter(
            x=[observatory_lon],
            y=[observatory_lat],
            mode='markers+text',
            marker=dict(
                size=12,
                color='white',
                symbol='square',
                line=dict(width=2, color='black')
            ),
            text=['Observatory'],
            textposition='top center',
            textfont=dict(
                size=12,
                color='white',
                family='Arial, bold'
            ),
            name='Observatory Building',
            hovertemplate=(
                '<b>Observatory Building</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )
    
    # Add Sensor Hut marker
    fig.add_trace(
        go.Scatter(
            x=[sensor_hut_lon],
            y=[sensor_hut_lat],
            mode='markers+text',
            marker=dict(
                size=12,
                color='yellow',
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            text=['Sensor Hut'],
            textposition='top center',
            textfont=dict(
                size=12,
                color='yellow',
                family='Arial, bold'
            ),
            name='Sensor Hut',
            hovertemplate=(
                '<b>Sensor Hut</b><br>' +
                'Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>' +
                '<extra></extra>'
            ),
            showlegend=False
        )
    )
    
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
            x=x,
            y=y,
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
    dcc.Markdown(r"""This tab shows the magnetic data after **improved** reduction to the pole transformation. 
           The enhanced algorithm includes better stabilization and filtering to produce more realistic magnetic field values.
           This technique maintains the regional magnetic field while transforming anomalies, resulting in more physically realistic results."""),
    
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
                        html.Li("Inclination: -40.3° "),
                        html.Li("Declination: -11.37° "),
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