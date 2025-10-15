import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from scipy.interpolate import griddata
from typing import Tuple, List, Dict, Any, Union

def load_and_preprocess_data(file_path: str = 'geochem_clean.csv') -> Tuple[pd.DataFrame, Union[gpd.GeoDataFrame, pd.DataFrame]]:
    """
    Loads geochemical data from a CSV file and returns both a standard DataFrame and a GeoDataFrame (if coordinates exist).
    Returns empty DataFrames if file not found.

    Args:
        file_path (str): Path to CSV file.

    Returns:
        Tuple[pd.DataFrame, Union[gpd.GeoDataFrame, pd.DataFrame]]: (df, gdf)
    """
    try:
        df = pd.read_csv(file_path)
        if {'x_utm', 'y_utm'}.issubset(df.columns):
            gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df['x_utm'], df['y_utm']))
            gdf = gdf.set_crs("EPSG:32721")
        else:
            gdf = df.copy()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        df = pd.DataFrame()
        gdf = pd.DataFrame()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        df = pd.DataFrame()
        gdf = pd.DataFrame()
    return df, gdf

def prepare_analysis_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Selects numeric columns (excluding x_utm and y_utm), drops rows with NaN,
    and standardizes the data.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple containing:
          - data_for_analysis (pd.DataFrame): Numeric data, NaNs dropped.
          - scaled_data_df (pd.DataFrame): Standardized numeric data.
          - numeric_cols (List[str]): Names of the numeric columns used.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for coord in ['x_utm', 'y_utm']:
        if coord in numeric_cols:
            numeric_cols.remove(coord)

    data_for_analysis = df[numeric_cols].dropna().copy()
    if not data_for_analysis.empty:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_for_analysis)
        scaled_data_df = pd.DataFrame(scaled_data, columns=data_for_analysis.columns, index=data_for_analysis.index)
    else:
        scaled_data_df = pd.DataFrame(columns=numeric_cols)

    return data_for_analysis, scaled_data_df, numeric_cols

# Initialize data once at module import
df, gdf = load_and_preprocess_data()
data_for_analysis, scaled_data_df, numeric_cols = prepare_analysis_data(df)

# Consistent column title mapping
column_title_map: Dict[str, str] = {
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
