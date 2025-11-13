# pages/surveyed_data.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from shared_data import survey_df, survey_xi, survey_yi, survey_zi, add_observatory_markers

def create_grid_plot():
    """Create grid plot using shared data and functions"""
    # Use the pre-calculated grid data from shared_data.py
    xi = survey_xi
    yi = survey_yi
    zi = survey_zi
    
    # Flatten the grid for plotting
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
            x=survey_df['Longitude (deg)'],
            y=survey_df['Latitude (deg)'],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='circle',
                line=dict(width=2, color='white')
            ),
            name='Survey Stations',
            text=[f'Station: {idx}<br>Longitude: {x:.5f}°<br>Latitude: {y:.5f}°<br>B: {z:.1f} nT' 
                  for idx, x, y, z in zip(survey_df.index, survey_df['Longitude (deg)'], survey_df['Latitude (deg)'], survey_df['B(nT)'])],
            hovertemplate='%{text}<extra></extra>'
        )
    )
    
    # Add observatory and sensor hut markers using shared function
    fig = add_observatory_markers(fig)
    
    # Configure layout 
    fig.update_layout(
        title=dict(
            text='<b>Magnetic Field (Total Intensity)</b>',
            x=0.5,
            font=dict(size=16)),
        height=600,
        showlegend=False,
        margin=dict(t=30, b=60, l=60, r=80),
    )
    
    return fig

def render_survey_data():
    """Render the survey data page"""
    st.markdown('<h1 class="main-header">Survey Data</h1>', unsafe_allow_html=True)
    
    # Data Summary and Survey Details in columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Data Summary")
        st.markdown(f"""
        <div class="metric-card">
            <b>Total Stations:</b> {len(survey_df)}<br>
            <b>Magnetic Field Range:</b> {survey_df['B(nT)'].min():.1f} - {survey_df['B(nT)'].max():.1f} nT<br>
            <b>Latitude Range:</b> {survey_df['Latitude (deg)'].min():.5f} - {survey_df['Latitude (deg)'].max():.5f}°<br>
            <b>Longitude Range:</b> {survey_df['Longitude (deg)'].min():.5f} - {survey_df['Longitude (deg)'].max():.5f}°
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔍 Survey Details")
        st.markdown("""
        **The Problem:**    
        Magnetic survey of a property of approximately one hectare to evaluate the feasibility 
        of installing a magnetic station on the site. We analyze data from unevenly spaced data 
        points along the study area. The datasets includes a simple magnetic survey plus a 
        gradiometric survey. The survey was performed with a portable proton magnetometer and 
        the gradiometric one with two vertically stacked Overhauser sensors at 1m of separation.
        """)
    
    st.markdown("---")
    
    # Data Table
    st.subheader("📋 Data Table")
    st.dataframe(survey_df, use_container_width=True)
    
    # Grid Plot
    st.subheader("🗺️ Magnetic Field Grid Plot")
    fig = create_grid_plot()
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🔴 Red markers show actual survey station locations")
