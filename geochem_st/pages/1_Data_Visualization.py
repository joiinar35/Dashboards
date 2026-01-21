"""
Data Visualization page for interactive geochemical data exploration.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.interpolate import griddata

from shared_data import (
    df,
    column_title_map,
    element_columns
)

st.markdown("""
            <h1> Data Visualization </h1>
            """
            , unsafe_allow_html=True)

# Load the shared CSS file FIRST
#with open("css/style.css", "r") as f:
#    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    

# Page content
st.markdown("""
            <style>
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(115deg, #1d2a3a, #117aca) !important;
    }
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    section[data-testid="stSidebar"] label {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stImage {
        margin-top: auto !important;
        padding-top: 20px !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="explanation-text" style="font-size: 1.3rem; text-align: justify;">
<p>This tab provides interactive visualizations of the geochemical data from a given geographical location and sample set.</p>
<p>Select an element to explore its distribution and correlations.</p>
<ul>
    <li><strong>Distribution of Selected Element:</strong> Violin and box plot showing element distribution</li>
    <li><strong>Interpolated Contour Map:</strong> Spatial distribution with interpolated contours using Viridis palette</li>
    <li><strong>Correlation Matrix:</strong> Heatmap showing correlations between all geochemical elements</li>
</ul>
<p>The white dots mark the location of the samples in the map.</p>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns([2, 8])

with col1:

    
    st.subheader("🔬 Element Selection")
    
    if len(element_columns) > 0:
        selected_element = st.selectbox(
            "Select Element",
            options=element_columns,
            format_func=lambda x: column_title_map.get(x, x),
            index=0
        )
    else:
        selected_element = None
        st.warning("No element columns available in data")
    
    st.markdown("""    
    <div style="margin-top: 20px; background-color: rgba(22, 27, 34, 0.9); border: 1px solid #31688e; border-radius: 8px; padding: 15px;">
    <h6 style="text-align: center; font-weight: bold; margin-bottom: 10px; color: #f5b041; border-bottom: 1px solid #31688e; padding-bottom: 10px;">📊 Data Source</h6>
    <p style="font-size: 12px; margin-bottom: 5px; color: #8b949e;">
    The data used in this dashboard comes from the 'Inventario Minero del Uruguay', 
    which is freely available in the DINAMIGE catalog hosted on the GeoNetwork of MIEM.
    </p>
    <p style="font-size: 12px; color: #8b949e;">
    <a href="https://geonetwork.miem.gub.uy/" target="_blank" style="color: #35b779;">(https://geonetwork.miem.gub.uy/)</a>
    </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Contour Map
    if selected_element and not df.empty:
        # Check for required columns and data
        if ('x_utm' in df.columns and 'y_utm' in df.columns and 
            selected_element in df.columns and not df[selected_element].isnull().all()):
            
            mask = df[['x_utm', 'y_utm', selected_element]].dropna()
            if not mask.empty:
                x = mask['x_utm'].values
                y = mask['y_utm'].values
                values = mask[selected_element].values
                
                # Interpolation grid
                grid_density = 100
                try:
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    
                    xi = np.linspace(x_min, x_max, grid_density)
                    yi = np.linspace(y_min, y_max, grid_density)
                    xi, yi = np.meshgrid(xi, yi)
                    grid_values = griddata((x, y), values, (xi, yi), method='cubic')
                    
                    if grid_values is not None:
                        grid_values = np.where(grid_values < 0, 0, grid_values)
                except Exception as e:
                    grid_values = None
                
                if grid_values is not None:
                    title = column_title_map.get(selected_element, selected_element)
                    fig = go.Figure(data=go.Contour(
                        z=grid_values,
                        x=xi[0, :],
                        y=yi[:, 0],
                        ncontours=25,
                        colorscale='Viridis',  # Using Viridis palette
                        contours=dict(coloring='fill', showlabels=False),
                        hoverinfo='z',
                        hovertemplate='<b>%{z:.1f} ppm</b><extra></extra>',
                        colorbar=dict(
                            title=f'<b>{title}</b>',
                            titleside='right',
                            titlefont=dict(color='black'),
                            tickfont=dict(color='black')
                        )
                    ))
                    
                    # Overlay sample locations (dots)
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='markers',
                        marker=dict(
                            color='rgba(255, 255, 255, 0.8)',
                            size=5,
                            line=dict(width=1, color='rgba(0, 0, 0, 0.5)')
                        ),
                        name='Samples',
                        hoverinfo='skip',
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        title=dict(
                            text=f"<b>Interpolated Contour Map - {title}</b>",
                            x=0.5, xanchor="center", y=0.9, yanchor="top",
                            font=dict(size=16, color="black", family="Arial")),
                        xaxis=dict(
                            showticklabels=False, showgrid=False, zeroline=False, 
                            showline=False, ticks='', color='white'
                        ),
                        yaxis=dict(
                            showticklabels=False, showgrid=False, zeroline=False, 
                            showline=False, ticks='', color='white'
                        ),
                        height=800,
                        margin=dict(l=40, r=20, t=50, b=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                else:
                    st.warning("Interpolation failed for contour map.")
            else:
                st.warning("No valid sample locations for contour map.")
        else:
            st.warning("Not enough data for contour map.")
    
    # Violin & Box plot
    if selected_element and not df.empty and selected_element in df.columns:
        clean_vals = df[selected_element].dropna()
        clean_vals = clean_vals[np.isfinite(clean_vals)]
        
        if not clean_vals.empty:
            title = column_title_map.get(selected_element, selected_element)
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(
                    f'<span style="color:black">Violin Plot of {title}</span>',
                    f'<span style="color:black">Box Plot of {title}</span>'
                )
            )
            
            fig.add_trace(
                go.Violin(
                    y=clean_vals, 
                    name='Violin', 
                    box_visible=True, 
                    meanline_visible=True, 
                    line_color='royalblue',  
                    fillcolor='rgba(48,92,222,0.3)',  # RoyalBlue with transparency
                    points=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Box(
                    y=clean_vals, 
                    name='Boxplot', 
                    marker_color='#31688e',  # Viridis blue
                    line_color='indianred'  # Viridis yellow
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=f"<span style='color:black'>Distribution of {title}</span>",
                showlegend=False,
                height=400,
                title_x=0.5,
                title_y=0.95,
                margin=dict(l=40, r=20, t=50, b=30),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#DEE2E6',
                font=dict(size=16, color="black", family='Arial')
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='black')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255, 1)') # otra forma de escribir negro pero ajustando alpha
            
            st.plotly_chart(fig, use_container_width=True, theme=None)
    
    # Correlation Matrix
    if not df.empty:
        st.markdown("<h3 style='color:black; margin-top: 20px;'>Correlation Matrix</h3>", unsafe_allow_html=True)
        
        elementos = df.select_dtypes(include=[np.number])
        for col in ['x_utm', 'y_utm']:
            if col in elementos.columns:
                elementos = elementos.drop(columns=[col])
        
        elementos = elementos.dropna(axis=1, how='all')
        
        if elementos.shape[0] > 1000:
            elementos = elementos.sample(n=1000, random_state=42)
        
        if elementos.shape[1] > 0:
            corr_matrix = elementos.corr().round(2)
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=[column_title_map.get(c, c) for c in corr_matrix.columns],
                y=[column_title_map.get(c, c) for c in corr_matrix.index],
                colorscale='RdBu',  # Using Viridis palette
                zmin=-1, zmax=1,
                hoverongaps=False,
                hovertemplate='Correlation between %{x} and %{y}: %{z:.2f}<extra></extra>',
                colorbar=dict(
                    title='<b>Correlation Coefficient</b>',
                    titleside='right',
                    titlefont=dict(color='black'),
                    tickfont=dict(color='black')
                )
            ))
            
            annotations = []
            for i, row in enumerate(corr_matrix.values):
                for j, value in enumerate(row):
                    font_color = 'white' if abs(value) > 0.7 else 'black'
                    annotations.append(
                        dict(
                            x=column_title_map.get(corr_matrix.columns[j], corr_matrix.columns[j]),
                            y=column_title_map.get(corr_matrix.index[i], corr_matrix.index[i]),
                            text=f'{value:.2f}',
                            showarrow=False,
                            font=dict(color=font_color, size=10),
                            bgcolor='rgba(255,255,255,0.5)' if abs(value) < 0.3 else 'rgba(0,0,0,0)'
                        )
                    )
            
            fig.update_layout(
                title=dict(
                    text='<b style="color:black">Correlation Matrix of Geochemical Elements</b>',
                    x=0.5, y=0.95, xanchor="center", yanchor="top",
                    font=dict(size=16, color="#f5b041", family="Arial")),
                xaxis=dict(
                    title='Elements',
                    tickangle=-45,
                    tickfont=dict(color='black'),
                    titlefont=dict(color='black')
                ),
                yaxis=dict(
                    title='Elements',
                    tickfont=dict(color='black'),
                    titlefont=dict(color='black')
                ),
                annotations=annotations,
                height=800,
                margin=dict(l=100, r=50, t=80, b=100),
                title_x=0.5,
                title_y=0.95,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
