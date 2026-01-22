"""
Factor Analysis page for identifying underlying factors in geochemical data.
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import geopandas as gpd
from factor_analyzer import FactorAnalyzer
import scipy

# PATCH: Fix for scipy.sum issue
if not hasattr(scipy, 'sum'):
    scipy.sum = np.sum

from shared_data import (
    load_and_preprocess_data, 
    prepare_analysis_data, 
    column_title_map
)

from pathlib import Path

def find_css(filename="css/style.css", max_levels=6):
    base = Path(__file__).resolve()
    for _ in range(max_levels):
        candidate = base.parent / filename if base.is_file() else base / filename
        if candidate.exists():
            return candidate
        base = base.parent
    return None

css_file = find_css()
if css_file:
    try:
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS from {css_file}: {e}")
else:
    st.warning("Could not find css/style.css — verify the file exists in the repository (search will look upward from this page).")

# Load data
df, gdf = load_and_preprocess_data()
data_for_analysis, scaled_data_df, numeric_cols = prepare_analysis_data(df)
st.markdown("""
            <h1> Factor Analysis </h1>
            """
            , unsafe_allow_html=True)


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
<div class="explanation-text" style="font-size: 1.3rem;" "text-align: justify">
<p><strong>Factor Analysis helps identify underlying factors that explain the relationships between variables.</strong></p>
<p>The Factor Analysis Loadings Heatmap shows the relationships between your original geochemical variables and the underlying factors:</p>
<ul>
    <li><strong>Rows:</strong> Original geochemical variables (e.g., Ba, Co).</li>
    <li><strong>Columns:</strong> Factors identified by the analysis (Factor 1, Factor 2, etc.).</li>
    <li><strong>Colors and Values:</strong> Indicate the 'loading' of a variable on a factor.</li>
    <ul>
        <li>High positive loading (warm colors, closer to 1): Variable is strongly and positively associated with the factor.</li>
        <li>High negative loading (cool colors, closer to -1): Variable is strongly and negatively associated with the factor.</li>
        <li>Loading near zero: Variable has little influence on that factor.</li>
    </ul>
</ul>
<p>By interpreting variables with high absolute loadings on each factor, you can understand the geological or geochemical processes represented by each factor.</p>
<p>The Factor Score Maps provide a glimpse the geographical predominance of each factor along the study area.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 8])

with col1:
    st.subheader("Factor Analysis Controls")
    
    if not data_for_analysis.empty:
        n_cols = len(data_for_analysis.columns)
        max_n = max(1, n_cols - 7)
        n_factors = st.selectbox(
            "Number of Factors:",
            options=list(range(1, max_n + 1)),
            index=min(3, max_n - 1) if max_n > 3 else 0
        )
    else:
        n_factors = 1
        st.warning("No data available for Factor Analysis")

with col2:
    if data_for_analysis.empty:
        st.warning("Not enough data for Factor Analysis.")
        st.stop()
    
    try:
        # Perform Factor Analysis
        fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
        fa.fit(data_for_analysis)
        eigenvalues_fa, _ = fa.get_eigenvalues()
        fa_loadings = fa.loadings_
        fa_variance = fa.get_factor_variance()
        fa_scores = fa.transform(data_for_analysis)
        
        # Create factor scores dataframe
        fa_scores_df = pd.DataFrame(
            fa_scores, 
            columns=[f'Factor_{i+1}_Score' for i in range(fa_scores.shape[1])]
        )
        fa_scores_df.index = data_for_analysis.index
        
        # Scree Plot and Variance Plot
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            if eigenvalues_fa is not None and len(eigenvalues_fa) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[f'{i+1}' for i in range(len(eigenvalues_fa))],
                    y=eigenvalues_fa,
                    mode='lines+markers',
                    name='Eigenvalues'
                ))
                fig.add_shape(
                    type="line", 
                    x0=-0.5, y0=1, x1=len(eigenvalues_fa)-0.5, y1=1,
                    line=dict(color="Red", width=2, dash="dash")
                )
                fig.update_layout(
                    title=dict(
                        text='<b>Factor Analysis Scree Plot (Eigenvalues)</b>',
                        x=0.5, y=0.9, xanchor="center", yanchor="top",
                        font=dict(size=16, color="black", family="Arial")
                    ),
                    xaxis_title='Factor Number',
                    yaxis_title='Eigenvalue',
                    showlegend=False,
                    margin={"r": 0, "t": 40, "l": 0, "b": 0}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2_2:
            if (fa_variance is not None and len(fa_variance) > 2 and 
                len(fa_variance[1]) >= n_factors and len(fa_variance[2]) >= n_factors):
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f'Factor {i+1}' for i in range(n_factors)],
                    y=fa_variance[1][:n_factors],  # Proportion of variance explained
                    name='Proportion<br>Explained Variance'
                ))
                fig.add_trace(go.Scatter(
                    x=[f'Factor {i+1}' for i in range(n_factors)],
                    y=fa_variance[2][:n_factors],  # Cumulative proportion
                    mode='lines+markers',
                    name='Cumulative<br>Explained Variance'
                ))
                fig.update_layout(
                    title=dict(
                        text=f'<b>Factor Analysis Explained Variance ({n_factors} Factors)</b>',
                        x=0.5, y=0.9, xanchor="center", yanchor="top",
                        font=dict(size=16, color="black", family="Arial")
                    ),
                    xaxis_title='Factor Number',
                    yaxis_title='Proportion of Variance',
                    legend_title='Variance Type',
                    margin={"r": 0, "t": 40, "l": 0, "b": 0},
                    legend=dict(
                        x=0.95, y=0.75, xanchor='right', yanchor='top',
                        bgcolor='rgba(255, 255, 255, 0.5)'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Loadings Heatmap
        loadings_df_fa = pd.DataFrame(
            fa_loadings,
            index=data_for_analysis.columns,
            columns=[f'Factor {i+1}' for i in range(fa_loadings.shape[1])]
        )
        fig = go.Figure(data=go.Heatmap(
            z=loadings_df_fa.values,
            x=loadings_df_fa.columns,
            y=loadings_df_fa.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='<b>Loading Value</b>', titleside='right')
        ))
        fig.update_layout(
            title=dict(
                text='<b>Factor Analysis Loadings Heatmap</b>',
                x=0.5, y=1, xanchor="center", yanchor="top",
                font=dict(size=16, color="black", family="Arial")
            ),
            xaxis_title='Factor',
            yaxis_title='Variable',
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Factor Score Maps
        st.subheader("Factor Score Maps")
        
        if (df.empty or 'x_utm' not in df.columns or 'y_utm' not in df.columns or
            data_for_analysis.empty or fa_scores_df.empty):
            st.warning("Not enough data or missing coordinates for factor score maps.")
            st.stop()
        
        # Add coordinates to factor scores
        fa_scores_df_with_coords = fa_scores_df.copy()
        fa_scores_df_with_coords['x_utm'] = df.loc[data_for_analysis.index, 'x_utm'].values
        fa_scores_df_with_coords['y_utm'] = df.loc[data_for_analysis.index, 'y_utm'].values
        
        # Create maps for each factor
        for i in range(n_factors):
            factor_score_col = f'Factor_{i+1}_Score'
            if factor_score_col in fa_scores_df_with_coords.columns:
                points = np.array([fa_scores_df_with_coords['x_utm'], fa_scores_df_with_coords['y_utm']]).T
                values = fa_scores_df_with_coords[factor_score_col].values
                
                if len(points) >= 4:
                    try:
                        grid_density = 100
                        x_min, x_max = points[:, 0].min(), points[:, 0].max()
                        y_min, y_max = points[:, 1].min(), points[:, 1].max()
                        
                        xi = np.linspace(x_min, x_max, grid_density)
                        yi = np.linspace(y_min, y_max, grid_density)
                        xi, yi = np.meshgrid(xi, yi)
                        
                        grid_values = griddata(points, values, (xi, yi), method='cubic')
                        
                        fig = go.Figure(data=go.Contour(
                            z=grid_values,
                            x=xi[0, :],
                            y=yi[:, 0],
                            ncontours=25,
                            colorscale='RdBu',
                            contours=dict(coloring='fill', showlabels=False),
                            hoverinfo='z',
                            hovertemplate='<b>%{z:.2f}</b><extra></extra>',
                            colorbar=dict(
                                title=f'<b>Factor {i+1} Score</b>',
                                titleside='right'
                            )
                        ))
                        
                        fig.update_layout(
                            title=dict(
                                text=f'<b>Interpolated Factor {i+1} Score Map</b>',
                                x=0.5, y=0.9, xanchor="center", yanchor="top",
                                font=dict(size=16, color="black", family="Arial")
                            ),
                            xaxis=dict(
                                showticklabels=False, showgrid=False, 
                                zeroline=False, showline=False, ticks=''
                            ),
                            yaxis=dict(
                                showticklabels=False, showgrid=False, 
                                zeroline=False, showline=False, ticks=''
                            ),
                            height=800
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not create map for Factor {i+1}: {str(e)}")
    
    except Exception as e:
        st.error(f"Error during Factor Analysis: {str(e)}")
