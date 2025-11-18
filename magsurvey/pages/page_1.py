# pages/surveyed_data.py
import streamlit as st
import plotly.graph_objects as go
from shared_data import survey_df, survey_xi, survey_yi, survey_zi, add_observatory_markers, add_scale_bar

def render_survey_data():
    st.markdown('<h1 class="main-header" style="color: #FF5e38; font-weight: 800;">Survey Data</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card" style="font-size: 1.3rem;" "text-align: justify">           
    The datatable present the raw magnetic data from the survey. Additionally, we show a summary of the data collected and 
    a contour plot of the magnetic intensities measured along the surveyed area. <br>
    A suitable place for a magnetic station should encompass very low magnetic gradients, absence of magnetic anomalies and
    cultural magnetic noise (e.g. power lines, trafic, fences, buildings, etc.). All these requirements are difficut
    to accomplish in urban areas, therefore, we surveyed a distant location far away of most antropic noise.
    </div> """, unsafe_allow_html=True)

    
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
        <div class="metric-card" style="text-align: justify">
        <strong>The Problem:</strong> Magnetic survey of a property of approximately one hectare to evaluate the feasibility 
        of installing a magnetic station on the site. We analyze data from unevenly spaced data 
        points along the study area. The datasets includes a simple magnetic survey plus a 
        gradiometric survey. The survey was performed with a portable proton magnetometer and 
        the gradiometric one with two vertically stacked Overhauser sensors at 1m of separation.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Table
    st.subheader("Data Table")
    st.dataframe(survey_df, width='stretch')
    
    # Grid Plot
    st.subheader("Magnetic Field Grid Plot")
    
    # Use the pre-calculated grid data
    xi = survey_xi
    yi = survey_yi
  
    
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=survey_xi.flatten(),
            y=survey_yi.flatten(), 
            z=survey_zi.flatten(),
            colorscale='Viridis',
            colorbar=dict(
                title="B (nT)",
                len=0.85,  # Reduced to 3/4 of original height
                y=0.425,   # Adjusted to center the shorter colorbar
                yanchor='middle'
            )
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=survey_df['Longitude (deg)'],
            y=survey_df['Latitude (deg)'],
            mode='markers',
            marker=dict(color='red', size=6),
            name='Survey Stations',
            showlegend=True
        )
    )
    
    # Add contour lines
    fig.add_trace(
        go.Contour(
            x=survey_xi.flatten(),
            y=survey_yi.flatten(), 
            z=survey_zi.flatten(),
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
    
    fig = add_observatory_markers(fig)
    
    # Add scale bar to the field map (yellow, top left corner)
    x_range = [xi[0].min(), xi[0].max()]
    y_range = [yi[:,0].min(), yi[:,0].max()]
    fig = add_scale_bar(fig , x_range, y_range, scale_length_meters=25)
    
    # Update layout with grid lines
    fig.update_layout(
        title="Magnetic Field (Total Intensity)", 
        height=1200,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            griddash='dot'
        ),
        showlegend=True
    )
    st.plotly_chart(fig, width='stretch')