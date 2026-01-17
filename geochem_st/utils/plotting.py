import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class GeochemistryPlots:
    """Class for creating geochemistry visualizations"""
    
    @staticmethod
    def create_histogram(df, column, title=None, nbins=50):
        """Create histogram with KDE"""
        fig = px.histogram(
            df, 
            x=column,
            nbins=nbins,
            marginal="box",
            opacity=0.7,
            title=title or f"Distribution of {column}",
            color_discrete_sequence=['#1E88E5']
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequency",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_boxplot(df, columns, title="Element Distribution"):
        """Create box plots for multiple elements"""
        # Melt the dataframe for Plotly
        melted_df = df[columns].melt(var_name="Element", value_name="Concentration")
        
        fig = px.box(
            melted_df,
            x="Element",
            y="Concentration",
            title=title,
            color="Element",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(
            xaxis_title="Elements",
            yaxis_title="Concentration (ppm)",
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_scatter_matrix(df, columns, title="Scatter Matrix"):
        """Create scatter matrix plot"""
        fig = px.scatter_matrix(
            df,
            dimensions=columns,
            title=title,
            opacity=0.7
        )
        
        fig.update_layout(
            width=1000,
            height=800,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(corr_matrix, title="Correlation Matrix"):
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Elements",
            yaxis_title="Elements",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def create_pca_plot(pca_results, explained_variance, title="PCA Analysis"):
        """Create PCA visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("PCA Scatter Plot", "Explained Variance Ratio"),
            column_widths=[0.7, 0.3]
        )
        
        # Scatter plot
        fig.add_trace(
            go.Scatter(
                x=pca_results[:, 0],
                y=pca_results[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=pca_results[:, 0],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="PC1")
                )
            ),
            row=1, col=1
        )
        
        # Variance plot
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_variance))],
                y=explained_variance,
                marker_color='#FF6B6B'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=False,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="Principal Components", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance", row=1, col=2)
        
        return fig
    
    @staticmethod
    def create_spatial_map(df, lat_col, lon_col, value_col, title="Spatial Distribution"):
        """Create spatial distribution map"""
        fig = px.scatter_mapbox(
            df,
            lat=lat_col,
            lon=lon_col,
            color=value_col,
            size=value_col,
            hover_name=df.index if df.index.name else None,
            hover_data=[value_col],
            title=title,
            color_continuous_scale=px.colors.sequential.Viridis,
            zoom=5
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":30,"l":0,"b":0},
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_time_series(df, time_col, value_cols, title="Time Series Analysis"):
        """Create time series plot for multiple elements"""
        fig = go.Figure()
        
        for col in value_cols:
            fig.add_trace(go.Scatter(
                x=df[time_col],
                y=df[col],
                name=col,
                mode='lines+markers',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Concentration (ppm)",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        return fig
