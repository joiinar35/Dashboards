import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from scipy.interpolate import griddata

# Data loading and preprocessing
def load_and_preprocess_data(file_path='geochem_clean.csv'):
    try:
        df = pd.read_csv(file_path)
        if 'x_utm' in df.columns and 'y_utm' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x_utm, df.y_utm))
            gdf = gdf.set_crs("EPSG:32721")
        else:
            gdf = df
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        df = pd.DataFrame()
        gdf = pd.DataFrame()
    
    return df, gdf

# Prepare data for analysis
def prepare_analysis_data(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'x_utm' in numeric_cols:
        numeric_cols.remove('x_utm')
    if 'y_utm' in numeric_cols:
        numeric_cols.remove('y_utm')
    
    data_for_analysis = df[numeric_cols].dropna().copy()
    
    if not data_for_analysis.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_analysis)
        scaled_data_df = pd.DataFrame(scaled_data, columns=data_for_analysis.columns, index=data_for_analysis.index)
    else:
        scaled_data_df = pd.DataFrame()
    
    return data_for_analysis, scaled_data_df, numeric_cols

# Initialize data (call this once at startup)
df, gdf = load_and_preprocess_data()
data_for_analysis, scaled_data_df, numeric_cols = prepare_analysis_data(df)

# Column title mapping
column_title_map = {
    'ba_ppm': 'Ba (ppm)',
    'fe2o3_pct': 'Fe₂O₃ (%)',
    'co_ppm': 'Co (ppm)',
    'cr_ppm': 'Cr (ppm)',
    'ni_ppm': 'Ni (ppm)',
    'cu_ppm': 'Cu (ppm)',
    'zn_ppm': 'Zn (ppm)',
    'pb_ppm': 'Pb (ppm)',
    'p_ppm': 'P (ppm)',
    'v_ppm': 'V (ppm)',
    'y_ppm': 'Y (ppm)',
}
