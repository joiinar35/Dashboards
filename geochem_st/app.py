import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Geochemistry Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🧪 Geochemistry Data Analysis Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=100)
    st.title("Navigation")
    st.markdown("---")
    
    # File uploader
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your Excel file", 
        type=['xlsx'],
        help="Upload an Excel file with geochemical data"
    )
    
    # Sample datasets
    st.subheader("📊 Sample Datasets")
    dataset_choice = st.selectbox(
        "Choose a sample dataset:",
        ["Elements Data", "PPM Data", "Samples Data"]
    )
    
    # Analysis parameters
    st.subheader("⚙️ Analysis Parameters")
    confidence_level = st.slider(
        "Confidence Level (%)",
        min_value=90,
        max_value=99,
        value=95,
        step=1
    )
    
    st.markdown("---")
    st.caption("Built with Streamlit • Geochemistry Analysis Tool")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview", 
    "📊 Element Distribution", 
    "📈 Correlation Analysis", 
    "🗺️ Spatial Analysis"
])

with tab1:
    from pages import _1_🏠_Overview
    _1_🏠_Overview.show()

with tab2:
    from pages import _2_📊_Element_Distribution
    _2_📊_Element_Distribution.show()

with tab3:
    from pages import _3_📈_Correlation_Analysis
    _3_📈_Correlation_Analysis.show()

with tab4:
    from pages import _4_🗺️_Spatial_Analysis
    _4_🗺️_Spatial_Analysis.show()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    st.caption("© 2024 Geochemistry Dashboard | Streamlit Version")
