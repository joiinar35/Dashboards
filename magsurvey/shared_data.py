import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2
from scipy.ndimage import gaussian_filter
from scipy.signal import windows
from scipy.interpolate import NearestNDInterpolator
import plotly.graph_objects as go

magnetometer_data = 'https://github.com/joiinar35/Dashboards/blob/6dc680ddae492d25074be450625608e4d54b1ab8/magsurvey/data/magnetometria2.csv'
gradiometer_data = 'https://github.com/joiinar35/Dashboards/blob/6dc680ddae492d25074be450625608e4d54b1ab8/magsurvey/data/gradiometria.csv'
# Load all datasets
def load_survey_data():
    """Load main survey data"""
    df = pd.read_csv(magnetometer_data)
    df.set_index('station', inplace=True)
    df = df.rename(columns={
        'Latitude': 'Latitude (deg)',
        'Longitude': 'Longitude (deg)'
    })
    return df

def load_gradiometer_data():
    """Load gradiometer data"""
    gdf = pd.read_csv(gradiometer_data)
    gdf = gdf.iloc[:, [0, 1, 3, 4, 5]]  # Select relevant columns
    return gdf

# Constants
Delta_x = 106e3   # m/deg in N-S
Delta_y = 110e3   # m/deg in E-W
decl = -11.37     # Declination
incl = -40.3      # Inclination

# Observatory and sensor hut coordinates
OBSERVATORY_LAT = -34.33344
OBSERVATORY_LON = -54.71229
SENSOR_HUT_LAT = -34.33305
SENSOR_HUT_LON = -54.71218

# Common grid creation function
def create_interpolation_grid(df, grid_points=100):
    """Create interpolation grid for survey data"""
    if 'Longitude (deg)' in df.columns:
        x = df['Longitude (deg)'].values 
        y = df['Latitude (deg)'].values
    else:
        x = df['Longitud (deg)'].values
        y = df['Latitud (deg)'].values
    
    xi = np.linspace(x.min(), x.max(), grid_points)
    yi = np.linspace(y.min(), y.max(), grid_points)
    xi, yi = np.meshgrid(xi, yi)
    
    return x, y, xi, yi

def interpolate_data(x, y, z, xi, yi, method='cubic'):
    """Interpolate data onto regular grid"""
    return griddata((x, y), z, (xi, yi), method=method)

# RTP processing functions
def create_data_mask(data, threshold=1e-10):
    """Create a mask for non-null data points"""
    non_nan_mask = ~np.isnan(data)
    non_zero_mask = np.abs(data) > threshold
    valid_mask = non_nan_mask & non_zero_mask
    return valid_mask

def extrapolate_nans(data):
    """Extrapolate NaN values using nearest neighbor interpolation"""
    if not np.isnan(data).any():
        return data
    
    data_extrapolated = data.copy()
    ny, nx = data.shape
    
    x_coords, y_coords = np.meshgrid(np.arange(nx), np.arange(ny))
    valid_mask = ~np.isnan(data)
    valid_points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask]))
    valid_values = data[valid_mask]
    nan_points = np.column_stack((x_coords[~valid_mask], y_coords[~valid_mask]))
    
    if len(nan_points) > 0:
        interpolator = NearestNDInterpolator(valid_points, valid_values)
        nan_values = interpolator(nan_points)
        data_extrapolated[~valid_mask] = nan_values
    
    return data_extrapolated

def apply_tukey_tapering(data, alpha=0.2):
    """Apply Tukey 2D window to reduce edge effects"""
    ny, nx = data.shape
    window_x = windows.tukey(nx, alpha)
    window_y = windows.tukey(ny, alpha)
    window_2d = np.outer(window_y, window_x)
    return data * window_2d

def create_wiener_filter(kx, ky, noise_level=1e-3, signal_level=1.0):
    """Create Wiener filter for stabilization in frequency domain"""
    k_squared = kx**2 + ky**2
    k_squared[k_squared == 0] = 1e-10
    
    signal_power = signal_level / (1 + k_squared)
    noise_power = noise_level * k_squared
    wiener_filter = signal_power / (signal_power + noise_power)
    
    return wiener_filter

def reduction_to_pole_improved(tf, dx, dy, inc, dec, apply_tapering=True, 
                              apply_wiener_filter=True, wiener_noise_level=1e-4,
                              apply_smoothing=True, sigma=0.8, alpha=0.2):
    """Improved RTP with better stabilization and filtering"""
    # Clean and prepare data
    tf_clean = extrapolate_nans(tf)
    regional_field = np.nanmean(tf_clean)
    tf_anomaly = tf_clean - regional_field
    
    # Apply tapering
    if apply_tapering:
        tf_processed = apply_tukey_tapering(tf_anomaly, alpha=alpha)
    else:
        tf_processed = tf_anomaly.copy()
    
    # Convert angles to radians
    inc_rad = np.deg2rad(inc)
    dec_rad = np.deg2rad(dec)
    
    # Grid dimensions
    ny, nx = tf_processed.shape
    
    # Calculate spatial frequencies
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    Kx, Ky = np.meshgrid(kx, ky)
    K_squared = Kx**2 + Ky**2
    K = np.sqrt(K_squared)
    K_squared[K_squared == 0] = 1e-10
    
    # Direction cosines
    X0 = np.cos(inc_rad) * np.cos(dec_rad)
    Y0 = np.cos(inc_rad) * np.sin(dec_rad) 
    Z0 = np.sin(inc_rad)
    Xp, Yp, Zp = 0.0, 0.0, 1.0  # Pole direction
    
    # Calculate RTP filter with stabilization
    with np.errstate(divide='ignore', invalid='ignore'):
        F_original = 1j * (X0 * Kx + Y0 * Ky) - Z0 * K
        F_pole = 1j * (Xp * Kx + Yp * Ky) - Zp * K
        rtp_filter = F_pole / F_original
        
        max_filter_magnitude = 10.0
        filter_magnitude = np.abs(rtp_filter)
        rtp_filter = np.where(filter_magnitude > max_filter_magnitude, 
                             rtp_filter * max_filter_magnitude / filter_magnitude, 
                             rtp_filter)
        rtp_filter = np.nan_to_num(rtp_filter, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Apply Wiener filter
    if apply_wiener_filter:
        wiener_filter = create_wiener_filter(Kx, Ky, noise_level=wiener_noise_level)
        rtp_filter = rtp_filter * wiener_filter
    
    # Apply FFT and filter
    tf_fft = fft2(tf_processed)
    rtp_fft = tf_fft * rtp_filter
    rtp_anomaly = np.real(ifft2(rtp_fft))
    rtp_result = rtp_anomaly + regional_field
    
    # Apply smoothing
    if apply_smoothing:
        rtp_result = gaussian_filter(rtp_result, sigma=sigma)
    
    return rtp_result

def apply_mask_to_data(data, mask, fill_value=np.nan):
    """Apply mask to data, setting masked values to fill_value"""
    masked_data = data.copy()
    masked_data[~mask] = fill_value
    return masked_data

# Common plotting utilities
def add_observatory_markers(fig):
    """Add observatory markers to plotly figure - modified for Streamlit"""
    # Observatory building coordinates
    obs_lon = OBSERVATORY_LON
    obs_lat = OBSERVATORY_LAT
    
    # Sensor hut coordinates  
    hut_lon = SENSOR_HUT_LON
    hut_lat = SENSOR_HUT_LAT
    
    # Add observatory building marker
    fig.add_trace(
        go.Scatter(
            x=[obs_lon],
            y=[obs_lat],
            mode='markers',
            marker=dict(
                size=12,
                color='white',
                symbol='square',
                line=dict(width=2, color='black')
            ),
            name='Observatory',
            hovertemplate='<b>Observatory Building</b><br>Longitude: -54.71229°<br>Latitude: -34.33344°<extra></extra>'
        )
    )
    
    # Add sensor hut marker
    fig.add_trace(
        go.Scatter(
            x=[hut_lon],
            y=[hut_lat],
            mode='markers',
            marker=dict(
                size=10,
                color='yellow', 
                symbol='circle',
                line=dict(width=2, color='black')
            ),
            name='Sensor Hut',
            hovertemplate='<b>Sensor Hut</b><br>Longitude: -54.71218°<br>Latitude: -34.33305°<extra></extra>'
        )
    )
    
    return fig


# Pre-load data for all pages
survey_df = load_survey_data()
gradiometer_df = load_gradiometer_data()

# Create common grid for survey data
survey_x, survey_y, survey_xi, survey_yi = create_interpolation_grid(survey_df)
survey_z = survey_df['B(nT)'].values
survey_zi = interpolate_data(survey_x, survey_y, survey_z, survey_xi, survey_yi, method='cubic')

# Create common grid for gradiometer data
grad_x = gradiometer_df['Longitud (deg)'].values
grad_y = gradiometer_df['Latitud (deg)'].values  
grad_z = gradiometer_df['dB (nT)'].values
grad_zi = interpolate_data(grad_x, grad_y, grad_z, survey_xi, survey_yi, method='cubic')

# Calculate grid spacing
dx_survey = Delta_x * abs(survey_x.max()-survey_x.min()) / len(survey_xi[0])
dy_survey = Delta_y * abs(survey_y.max()-survey_y.min()) / len(survey_yi[:,0])
