# pages/magnetic_gradients.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from shared_data import survey_df, gradiometer_df, survey_xi, survey_yi, survey_zi, grad_zi, Delta_x, Delta_y, add_observatory_markers

def render_magnetic_gradients():
    st.markdown('<h1 class="main-header" style="color: #FF5e38; font-weight: 800;">Magnetic Gradients</h1>', 
                unsafe_allow_html=True)
    
    
    st.markdown("""
    <div class="metric-card" style="font-size: 1.3rem;" "text-align: justify">           
    This tab displays calculated magnetic gradients from the survey data. This technique helps
    to identify and locate buried sources of magnetization. <b>Hover over the plots to see detailed 
    gradient values and measurement locations.</b>
    </div> """, unsafe_allow_html=True)
    
    # Extract coordinates and values
    x = survey_df['Longitude (deg)'].values
    y = survey_df['Latitude (deg)'].values
    z = survey_df['B(nT)'].values
    
    # Extract gradiometer data
    gx = gradiometer_df['Longitud (deg)'].values
    gy = gradiometer_df['Latitud (deg)'].values
    
    # Use the pre-calculated grid
    xi = survey_xi
    yi = survey_yi
    zi = survey_zi
    gzi = grad_zi
    
    # Calculate grid spacing
    dx = Delta_x * abs(x.max()-x.min()) / len(xi[0])
    dy = Delta_y * abs(y.max()-y.min()) / len(yi[:,0])
    
    # Calculate gradients 
    (dBx, dBy) = np.gradient(zi, dy, dx)
    dBx = dBx/Delta_x    # Gradient in X
    dBy = dBy/Delta_y    # Gradient in Y
    
    # Calculate different gradient components
    total_horizontal_gradient = np.sqrt(dBx**2 + dBy**2)
    vertical_gradient = gzi
    
    # Create statistics
    dBh_avg = np.nanmean(total_horizontal_gradient)
    dBh_max = np.nanmax(total_horizontal_gradient)
    dBz_avg = np.nanmean(vertical_gradient)
    dBz_max = np.nanmax(vertical_gradient)
    
    # Display statistics and notes in columns
    col1, col2 = st.columns([1,2])
    
    with col1:
        st.subheader("📈 Gradient Statistics")
        st.markdown(f"""
        <div class="metric-card">            
        <b>Average Horizontal Gradient:</b> {dBh_avg*1e3:.3f} ×10⁻³ nT/m <br>
        <b>Max Horizontal Gradient:</b> {dBh_max*1e3:.3f} ×10⁻³ nT/m <br> 
        <b>Average Vertical Gradient:</b> {dBz_avg:.3f} nT/m <br>
        <b>Max Vertical Gradient:</b> {dBz_max:.3f} nT/m <br>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🏷️ Labels")
        st.markdown("""           
        - **White dots**: Measurement points
        - **Green diamonds**: Gradiometer points  
        - **White square**: Observatory building
        - **Yellow circle**: Sensor hut
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📝 Key Notes")
        st.markdown(r"""
        <div class="metric-card" style="text-align: justify">            
        The maps show the spatial distribution of the total horizontal magnetic gradient
        dBh and the gradiometric profile showing vertical gradient dBz, respectively.<br>   
        
                
        Both graphs show large anomalies close to the west and southeast edges of the
        surveyed area. The main causes of such anomalies are due to buried ferromagnetic 
        objects or the proximity to building remains, fences, etc. <br> 
        
        In general there is a good agreement between the three graphs denoting that 
        the largest magnetic gradients occur in the vertical direction. <br> 
        
        The area of the gradiometric survey is slightly smaller than the magnetic one.
        Despite some large vertical gradients in the area, the planned sensor hut is 
        located over a low gradient area which makes the site suitable to take acceptable 
        magnetic readings.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Horizontal Gradient Plot
    st.subheader("📊 Total Horizontal Magnetic Gradient")
    
    fig_horizontal = go.Figure()
    fig_horizontal.add_trace(
        go.Contour(
            z=1e3*total_horizontal_gradient,
            x=xi[0],
            y=yi[:,0],
            colorscale='Viridis',
            colorbar=dict(title=r"dBh (×10⁻³) (nT/m)",
                          len=0.85,
                          y=0.425,
                          yanchor='middle'
                      ),
            contours=dict(coloring='heatmap', showlabels=True),
            line=dict(color='white', width=1),
            hovertemplate=(
                '<b>Horizontal Gradient</b><br>' +
                'Longitude: %{x:.6f}°<br>Latitude: %{y:.6f}°<br>' +
                'dBh: %{z:.3f} ×10⁻³ nT/m<br><extra></extra>'
            )
        )
    )
    
    # Add measurement points
    fig_horizontal.add_trace(
        go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=8, color='rgba(255, 255, 255, 0.7)', symbol='circle', line=dict(width=1, color='black')),
            name='Survey Stations',
            showlegend=True,
            hovertemplate='<b>Measurement Point</b><br>Longitude: %{x:.6f}°<br>Latitude: %{y:.6f}°<br><extra></extra>'
        )
    )
    
    fig_horizontal = add_observatory_markers(fig_horizontal)
    
    # Update layout with grid lines
    fig_horizontal.update_layout(
        height=1200, 
        title="Total Horizontal Magnetic Gradient",
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
    st.plotly_chart(fig_horizontal, width='stretch')
    
    # Vertical Gradient Plot
    st.subheader("📊 Vertical Magnetic Gradient")
    
    fig_vertical = go.Figure()
    fig_vertical.add_trace(
        go.Contour(
            z=vertical_gradient,
            x=xi[0],
            y=yi[:,0],
            colorscale='Plasma',
            colorbar=dict(title=r"dBz (nT/m)",
                          len=0.85,
                          y=0.425,
                          yanchor='middle'
                      ),
            contours=dict(coloring='heatmap', showlabels=True),
            line=dict(color='white', width=1),
            hovertemplate=(
                '<b>Vertical Gradient</b><br>' +
                'Longitude: %{x:.6f}°<br>Latitude: %{y:.6f}°<br>' +
                'dBz: %{z:.3f} nT/m<br><extra></extra>'
            )
        )
    )
    
    # Add points
    fig_vertical.add_trace(
        go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(size=8, color='rgba(255, 255, 255, 0.7)', symbol='circle', line=dict(width=1, color='black')),
            name='Survey Stations',
            showlegend=True
        )
    )
    
    fig_vertical.add_trace(
        go.Scatter(
            x=gx, y=gy, mode='markers',
            marker=dict(size=4, color='rgba(0, 255, 0, 0.6)', symbol='diamond', line=dict(width=1, color='darkgreen')),
            name='Gradiometer Points'
        )
    )
    
    fig_vertical = add_observatory_markers(fig_vertical)
    
    # Update layout with grid lines
    fig_vertical.update_layout(
        height=1200, 
        title="Vertical Magnetic Gradient",
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
    st.plotly_chart(fig_vertical, width='stretch')
