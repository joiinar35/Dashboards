"""
PCA Analysis page for principal component analysis and clustering.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from shared_data import (
    load_and_preprocess_data, 
    prepare_analysis_data, 
    column_title_map
)

st.markdown("""
            <h1> PCA Analysis </h1>
            """
            , unsafe_allow_html=True)


# Load data
df, gdf = load_and_preprocess_data()
data_for_analysis, scaled_data_df, numeric_cols = prepare_analysis_data(df)

# Load the shared CSS file FIRST
with open("./css/style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
<p><strong>The PCA loadings heatmap shows how much each original variable contributes to each principal component.</strong></p>
<p><strong>Rows</strong> represent your original geochemical variables (e.g., Ba, Co, Cr).</p>
<p><strong>Columns</strong> represent the principal components (PC1, PC2, etc.).</p>
<p><strong>Colors and Values:</strong> The color and the number in each cell indicate the 'loading' of that variable on that principal component.</p>
<ul>
    <li>A high positive loading (warm colors, closer to 1) means the variable is strongly and positively correlated with that principal component.</li>
    <li>A high negative loading (cool colors, closer to -1) means the variable is strongly and negatively correlated with that principal component.</li>
    <li>A loading close to zero (around the center color) means the variable has little influence on that principal component.</li>
</ul>
<p>By looking at the variables with high absolute loadings on each principal component, you can interpret what each component represents in terms of the original geochemical data.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 8])

with col1:
    st.subheader("PCA Controls")
    
    if not data_for_analysis.empty:
        max_components = min(10, len(data_for_analysis.columns) - 1)
        n_components = st.selectbox(
            "Number of Components:",
            options=list(range(2, max_components + 1)),
            index=0
        )
    else:
        n_components = 2
        st.warning("No data available for PCA")
    
    st.subheader("Clustering Controls")
    n_clusters = st.selectbox(
        "Number of Clusters (k):",
        options=list(range(2, 6)),
        index=0
    )

with col2:
    if data_for_analysis.empty or scaled_data_df.empty:
        st.warning("Not enough data for PCA analysis.")
        st.stop()
    
    try:
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(scaled_data_df)
        pca_explained_variance = pca.explained_variance_ratio_
        pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Create PCA results dataframe
        pca_df = pd.DataFrame(
            pca_components, 
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        pca_df = pca_df.set_index(scaled_data_df.index)
        
        # Scatter Plot
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            x_col = 'PC1'
            y_col = 'PC2' if n_components >= 2 else 'PC1'
            scatter_fig = px.scatter(
                pca_df,
                x=x_col,
                y=y_col,
                title='PCA: PC1 vs PC2' if n_components >= 2 else 'PCA: PC1',
                hover_data=pca_df.columns
            )
            scatter_fig.update_layout(
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                title=dict(
                    text='<b>PCA: PC1 vs PC2</b>' if n_components >= 2 else '<b>PCA: PC1</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial"))
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Scree Plot
        with col2_2:
            explained_variance_subset = pca_explained_variance[:n_components]
            cumulative_variance_subset = np.cumsum(explained_variance_subset)
            
            scree_fig = go.Figure()
            scree_fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_variance_subset,
                name='Individual Explained Variance',
                showlegend=False
            ))
            scree_fig.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=cumulative_variance_subset,
                mode='lines+markers',
                name='Cumulative Explained Variance'
            ))
            scree_fig.update_layout(
                title=dict(
                    text=f'<b>PCA Scree Plot ({n_components} Components)</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")),
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend=dict(
                    x=0.95, y=0.75,
                    xanchor='right', yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.5)'
                )
            )
            st.plotly_chart(scree_fig, use_container_width=True)
        
        # Loadings Heatmap
        loadings_df = pd.DataFrame(
            pca_loadings,
            index=scaled_data_df.columns,
            columns=[f'PC{i+1}' for i in range(pca_loadings.shape[1])]
        )
        
        max_abs_loading = np.abs(loadings_df.values).max()
        zmin, zmax = (-1, 1) if max_abs_loading <= 1 else (-max_abs_loading, max_abs_loading)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=loadings_df.values,
            x=loadings_df.columns,
            y=loadings_df.index,
            colorscale='RdBu',
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title='<b>Loading Value</b>', titleside='right')
        ))
        heatmap_fig.update_layout(
            title=dict(
                text='<b>PCA Loadings Heatmap</b>',
                x=0.5, y=1, xanchor="center", yanchor="top",
                font=dict(size=16, color="black", family="Arial")),
            xaxis_title='Principal Component',
            yaxis_title='Variable',
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            height=500,
            width=500
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Clustering
        st.subheader("Clustering based on PCA Components")
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(pca_components)
        
        # Create cluster dataframe with coordinates
        cluster_df = pd.DataFrame({
            'cluster': clusters,
            'x_utm': df.loc[scaled_data_df.index, 'x_utm'].values,
            'y_utm': df.loc[scaled_data_df.index, 'y_utm'].values
        })
        
        # Create interpolated cluster map
        points = np.array([cluster_df['x_utm'], cluster_df['y_utm']]).T
        values = cluster_df['cluster'].values
        
        try:
            grid_density = 70
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            
            xi = np.linspace(x_min, x_max, grid_density)
            yi = np.linspace(y_min, y_max, grid_density)
            xi, yi = np.meshgrid(xi, yi)
            
            grid_values = griddata(points, values, (xi, yi), method='nearest')
            
            fig = go.Figure(data=go.Contour(
                z=grid_values,
                x=xi[0, :],
                y=yi[:, 0],
                ncontours=n_clusters,
                colorscale='Viridis',
                contours=dict(
                    coloring='heatmap',
                    showlabels=True,
                    labelfont=dict(size=12, color='white')
                ),
                hoverinfo='z',
                hovertemplate='Cluster: %{z:.0f}<extra></extra>',
                colorbar=dict(
                    title='<b>Cluster</b>',
                    titleside='right',
                    tickvals=np.arange(n_clusters),
                    ticktext=[str(i) for i in range(n_clusters)]
                )
            ))
            
            fig.update_layout(
                title=dict(
                    text=f'<b>Interpolated Cluster Map (k={n_clusters})</b>',
                    x=0.5, y=0.9, xanchor="center", yanchor="top",
                    font=dict(size=16, color="black", family="Arial")),
                xaxis=dict(
                    showticklabels=False, showgrid=False, 
                    zeroline=False, showline=False, ticks=''
                ),
                yaxis=dict(
                    showticklabels=False, showgrid=False, 
                    zeroline=False, showline=False, ticks=''
                ),
                height=800,
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating cluster map: {str(e)}")
        
    except Exception as e:
        st.error(f"Error during PCA analysis: {str(e)}")