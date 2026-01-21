"""
Shared data for all pages - Loads and caches data once
"""
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_and_preprocess_data(file_path: str | None = None):
    """Load and preprocess data for the dashboard.

    If file_path is None, use the default raw GitHub URL.
    """
    if file_path is None:
        file_path = "https://raw.githubusercontent.com/joiinar35/Dashboards/main/geochem_st/data/geochem_clean.csv"

    try:
        df = pd.read_csv(file_path)
        if 'x_utm' in df.columns and 'y_utm' in df.columns:
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x_utm, df.y_utm))
            gdf = gdf.set_crs("EPSG:32721")
        else:
            gdf = df
    except FileNotFoundError:
        st.error(f"Error: {file_path} not found.")
        df = pd.DataFrame()
        gdf = pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error(f"Error: {file_path} contains no data.")
        df = pd.DataFrame()
        gdf = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        df = pd.DataFrame()
        gdf = pd.DataFrame()

    return df, gdf

@st.cache_data
def prepare_analysis_data(df):
    """Prepare data for analysis."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

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

# Helper functions
def get_element_columns(df, numeric_cols):
    """Return columns representing elements for dropdown selection."""
    if df is None or df.empty:
        return []

    if 'ba_ppm' in df.columns and 'zn_ppm' in df.columns:
        start = df.columns.get_loc('ba_ppm')
        end = df.columns.get_loc('zn_ppm') + 1
        return df.columns[start:end].tolist()

    if numeric_cols and len(numeric_cols) > 0:
        return numeric_cols

    return df.select_dtypes(include=[np.number]).columns.tolist()

# Load global data once
df, gdf = load_and_preprocess_data()
data_for_analysis, scaled_data_df, numeric_cols = prepare_analysis_data(df)
element_columns = get_element_columns(df, numeric_cols)
