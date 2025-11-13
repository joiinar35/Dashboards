import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from shared_data import survey_df, gradiometer_df, survey_xi, survey_yi, survey_zi, add_observatory_markers
from shared_data import dx_survey, dy_survey, decl, incl, create_data_mask, extrapolate_nans, reduction_to_pole_improved

# Set page config
st.set_page_config(
    page_title="Magnetic Survey Analysis",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS from assets folder
def load_css():
    css_url = 'https://raw.githubusercontent.com/joiinar35/Dashboards/main/magsurvey/assets/style.css'
    try:
        with open(css_url, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # Fallback CSS if file doesn't exist
        st.markdown("""
        <style>
            .main-header {
                font-size: 2.5rem;
                color: #2c3e50;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #3498db;
                margin-bottom: 1rem;
            }
            .custom-tab {
                background-color: #f8f9fa;
                border-radius: 0.5rem;
                padding: 0.5rem;
                margin-bottom: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)

# Load the CSS
load_css()

# Sidebar navigation
st.sidebar.title("Magnetic Survey Analysis")
page = st.sidebar.radio("Navigate to:", [
    "Survey Data", 
    "Magnetic Gradients", 
    "Reduction to the Pole"
])

# Survey Data Page
if page == "Survey Data":
    st.title("Survey Data")
    
    # Data summary cards
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Summary")
        st.metric("Total Stations", len(survey_df))
        st.metric("Magnetic Field Range", f"{survey_df['B(nT)'].min():.1f} - {survey_df['B(nT)'].max():.1f} nT")
        st.metric("Latitude Range", f"{survey_df['Latitude (deg)'].min():.5f} - {survey_df['Latitude (deg)'].max():.5f}°")
        st.metric("Longitude Range", f"{survey_df['Longitude (deg)'].min():.5f} - {survey_df['Longitude (deg)'].max():.5f}°")
    
    with col2:
        st.subheader("Survey Details")
        st.markdown("""
        **The Problem:**  
        Magnetic survey of a property of approximately one hectare to evaluate the feasibility 
        of installing a magnetic station on the site. We analyze data from unevenly spaced data 
        points along the study area. The datasets includes a simple magnetic survey plus a 
        gradiometric survey. The survey was performed with a portable proton magnetometer and 
        the gradiometric one with two vertically stacked Overhauser sensors at 1m of separation.
        """)
    
    # Data table
    st.subheader("Data Table")
    st.dataframe(survey_df, use_container_width=True)
    
    # Grid plot
    st.subheader("Magnetic Field Grid Plot")
    
    # Recreate the grid plot (simplified version)
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(
        go.Heatmap(
            x=survey_xi.flatten(),
            y=survey_yi.flatten(), 
            z=survey_zi.flatten(),
            colorscale='Viridis',
            colorbar=dict(title="B (nT)")
        )
    )
    
    # Add data points
    fig.add_trace(
        go.Scatter(
            x=survey_df['Longitude (deg)'],
            y=survey_df['Latitude (deg)'],
            mode='markers',
            marker=dict(color='red', size=6),
            name='Survey Stations'
        )
    )
    
    fig.update_layout(
        title="Magnetic Field (Total Intensity)",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red markers show actual survey station locations")

# Magnetic Gradients Page
elif page == "Magnetic Gradients":
    st.title("Magnetic Gradients")
    
    # Your magnetic gradients code here (simplified)
    st.info("Magnetic gradients analysis would go here...")

# Reduction to the Pole Page  
elif page == "Reduction to the Pole":
    st.title("Reduction to the Pole (RTP)")
    
    # Your RTP code here (simplified)
    st.info("Reduction to the pole analysis would go here...")

