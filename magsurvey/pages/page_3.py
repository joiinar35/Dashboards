# pages/reduction_to_pole.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from shared_data import survey_df, survey_xi, survey_yi, survey_zi, dx_survey, dy_survey, decl, incl
from shared_data import create_data_mask, extrapolate_nans, apply_mask_to_data, reduction_to_pole_improved, add_observatory_markers, add_scale_bar

def render_reduction_to_pole():
    """Render the reduction to pole page"""
    st.markdown('<h1 class="main-header" style="color: #FF5e38; font-weight: 800;">Reduction to the Pole (RTP)</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card" style="font-size: 1.3rem;" "text-align: justify">  
    This tab shows the magnetic data after applying an <font color="#FF5e38;"> improved reduction to the pole transformation</font>. 
    The enhanced algorithm includes better stabilization and filtering to produce more realistic magnetic field values.
    This technique maintains the regional magnetic field while transforming anomalies, resulting in more physically 
    accurate results.</div> """, unsafe_allow_html=True)
    
    # Use the pre-calculated grid data
    xi = survey_xi
    yi = survey_yi
    tf = survey_zi
    
    # Use pre-calculated grid spacing
    dx = dx_survey
    dy = dy_survey
    
    # Process data
    data_mask = create_data_mask(tf)
    tf_clean = extrapolate_nans(tf)
    
    # Improved reduction to pole calculation
    tf_red_improved = reduction_to_pole_improved(
        tf_clean, dx, dy, incl, decl,
        apply_tapering=True,
        apply_wiener_filter=True,
        wiener_noise_level=1e-5,
        apply_smoothing=True,
        sigma=0.5,
        alpha=0.2
    )
    
    # Apply mask to reduced field for display
    tf_red_masked_improved = apply_mask_to_data(tf_red_improved, data_mask)
    
    # Calculate RMS error
    tf_err_improved = tf_clean - tf_red_improved
    tf_err_masked_improved = apply_mask_to_data(tf_err_improved, data_mask)
    valid_errors_improved = tf_err_improved[data_mask]
    rms_error_improved = np.sqrt(np.mean(valid_errors_improved**2))
    
    # Display RTP features and error map in columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 RTP Key Features")
        st.markdown("""
        <div class="metric-card"> 
        - <b>Wiener filtering</b> for frequency-domain stabilization <br>
        - <b>Regional field preservation</b> to maintain realistic field magnitudes <br>  
        - <b>Controlled filter amplification</b> to prevent unrealistic values <br>
        - <b>Enhanced edge tapering</b> to reduce boundary effects <br>
        - <b>Optimized smoothing</b> to preserve geological features  <br>
        </div>
        """, unsafe_allow_html=True)
       
        st.subheader("📋 RTP Summary")
        st.markdown(f"""         
        <div style="font-size: 1.3rem; line-height: 2.0;">
        - <b>Inclination:</b> {incl}°<br>
        - <b>Declination:</b> {decl}°<br>
        - <b>Original field range:</b> {np.nanmin(tf):.1f} to {np.nanmax(tf):.1f} nT<br>
        - <b>Reduced field range:</b> {np.nanmin(tf_red_improved):.1f} to {np.nanmax(tf_red_improved):.1f} nT<br>
        - <b>Regional field preserved:</b> {np.nanmean(tf):.1f} nT<br>
        - <b>RMS Error:</b> {rms_error_improved:.2f} nT<br>
        - <b>Valid data points:</b> {np.sum(data_mask)}/{tf.size}
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader(f"📊 RMS Error Map (RMS: {rms_error_improved:.2f} nT)")
        
        # Create error map
        fig_error = go.Figure()
        fig_error.add_trace(
            go.Contour(
                z=tf_err_masked_improved,
                x=xi[0],
                y=yi[:,0],
                colorscale='RdBu_r',
                colorbar=dict(title='Error (nT)'),
                contours=dict(coloring='heatmap', showlabels=True),
                hovertemplate=(
                    '<b>Field Error</b><br>Longitude: %{x:.6f}°<br>' +
                    'Latitude: %{y:.6f}°<br>Error: %{z:.1f} nT<br><extra></extra>'
                )
            )
        )
        
        # Add original data points
        fig_error.add_trace(
            go.Scatter(
                x=survey_df['Longitude (deg)'],
                y=survey_df['Latitude (deg)'],
                mode='markers',
                marker=dict(size=6, color='green', symbol='circle', opacity=0.6),
                name='Survey Stations',
                showlegend=False
            )
        )
        
        # Update layout with grid lines
        fig_error.update_layout(
            height=800,
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
            )
        )
        st.plotly_chart(fig_error, width='stretch')
    
    st.markdown("---")
    
    # Reduced Field Map
    st.subheader("🗺️ Reduced Field Map")
    st.markdown("*Hover over the contour plot to see exact magnetic field values and coordinates at any point.*")
    
    fig_reduced = go.Figure()
    fig_reduced.add_trace(
        go.Contour(
            z=tf_red_masked_improved,
            x=xi[0],
            y=yi[:,0],
            colorscale='Viridis',
            colorbar=dict(title='B_red (nT)',
                          len=0.85,  # Reduced to 3/4 of original height
                          y=0.425,   # Adjusted to center the shorter colorbar
                          yanchor='middle'
                      ),
            contours=dict(coloring='heatmap', showlabels=True),
            line=dict(color='white', width=1),
            hovertemplate=(
                '<b>Reduced Field</b><br>Longitude: %{x:.6f}°<br>' +
                'Latitude: %{y:.6f}°<br>Field: %{z:.1f} nT<br><extra></extra>'
            )
        )
    )
    
    # Add original data points
    fig_reduced.add_trace(
        go.Scatter(
            x=survey_df['Longitude (deg)'],
            y=survey_df['Latitude (deg)'],
            mode='markers',
            marker=dict(size=8, color='red', symbol='circle', line=dict(width=1, color='white')),
            name='Survey Stations',
            showlegend=True
        )
    )
    
    fig_reduced = add_observatory_markers(fig_reduced)
    
    # Add scale bar to the reduced field map (yellow, top left corner)
    x_range = [xi[0].min(), xi[0].max()]
    y_range = [yi[:,0].min(), yi[:,0].max()]
    fig_reduced = add_scale_bar(fig_reduced, x_range, y_range, scale_length_meters=25)
    
    # Update layout with grid lines
    fig_reduced.update_layout(
        height=1200, 
        title="Reduced to Pole Magnetic Field",
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
    st.plotly_chart(fig_reduced, width='stretch')