"""
Pair Matrix page for comprehensive geochemical element relationships visualization.
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

from shared_data import (
    df,  # Usar df cargado globalmente
    column_title_map,
    element_columns  # Usar element_columns cargado globalmente
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

st.markdown("""
            <h1> Pair Matrix </h1>
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
<p><strong>The Pair Matrix provides a comprehensive view of relationships between multiple geochemical elements:</strong></p>
<ul>
    <li><strong>Upper Triangle:</strong> Scatter plots showing relationships between element pairs</li>
    <li><strong>Diagonal:</strong> Normalized Histograms with KDE showing distribution of each selected element</li>
    <li><strong>Lower Triangle:</strong> 2D Density plots showing density relationships</li>
    <li><strong>Correlation coefficients (r)</strong> are displayed in the upper triangle</li>
</ul>
<p>This interactive visualization helps identify patterns, correlations, and potential outliers in your geochemical data.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 8])

with col1:
    st.subheader("Pair Matrix Controls")
    
    # Sample Size
    sample_size = st.selectbox(
        "Sample Size:",
        options=['Full Dataset', 100, 200, 500],
        format_func=lambda x: f'{x} samples' if x != 'Full Dataset' else x,
        index=1
    )
    
    # Element Selection
    if len(element_columns) > 0:
        selected_elements = st.multiselect(
            "Select Elements for Pair Matrix:",
            options=element_columns,
            format_func=lambda x: column_title_map.get(x, x),
            default=element_columns[:4] if len(element_columns) >= 4 else element_columns,
            help="Select up to 6 elements for Pair Matrix"
        )
    else:
        selected_elements = []
        st.warning("No element columns available in data")
    
    # Limit to 6 elements
    if len(selected_elements) > 6:
        st.warning("Maximum 6 elements allowed. Using first 6 elements.")
        selected_elements = selected_elements[:6]
    
    generate_button = st.button("Generate Pair Matrix", type="primary")
    
    st.markdown("""
    <div class="warning-text">
    <p><strong>Note:</strong> For performance and clarity, maximum 6 elements allowed for Pair Matrix.</p>
    <p>If you select more than 6, only the first 6 will be used.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if not generate_button:
        st.info("Please select up to 6 elements and click 'Generate Pair Matrix'")
        st.stop()
    
    if not selected_elements:
        st.warning("Please select at least one element.")
        st.stop()
    
    if df.empty:
        st.warning("No data available for pair matrix visualization.")
        st.stop()
    
    # Prepare data
    elements_df = df.select_dtypes(include=['float64', 'int64'])
    elements_df = elements_df.drop(
        columns=[col for col in ['x_utm', 'y_utm'] if col in elements_df.columns]
    )
    
    try:
        elements_df = elements_df[selected_elements]
    except KeyError as e:
        st.error(f"Selected elements not found in data: {e}")
        st.stop()
    
    # Remove columns with all NaN values
    elements_df = elements_df.dropna(axis=1, how='all')
    
    if elements_df.empty:
        st.warning("No valid numeric data available for selected elements.")
        st.stop()
    
    # Handle sampling for performance
    if sample_size != 'Full Dataset' and len(elements_df) > int(sample_size):
        try:
            elements_df = elements_df.sample(n=int(sample_size), random_state=42)
        except ValueError:
            pass
    
    # Get readable titles for selected elements
    selected_elements_titles = [
        column_title_map.get(col, col) for col in selected_elements 
        if col in elements_df.columns
    ]
    
    n_elements = len(selected_elements_titles)
    
    # Create pair matrix
    try:
        elements = elements_df.columns.tolist()
        
        if n_elements == 0:
            st.stop()
        
        # Create subplots
        fig = make_subplots(
            rows=n_elements, 
            cols=n_elements,
            shared_xaxes=False,
            shared_yaxes=False,
            horizontal_spacing=0.05,
            vertical_spacing=0.05,
            subplot_titles=[]
        )
        
        # Calculate correlations once
        corr_matrix = elements_df.corr()
        
        for i, y_col in enumerate(elements):
            for j, x_col in enumerate(elements):
                row = i + 1
                col = j + 1
                
                # Get clean data for this pair
                data = elements_df[[x_col, y_col]].dropna()
                
                if len(data) < 2:
                    # Empty subplot if not enough data
                    fig.add_trace(
                        go.Scatter(x=[], y=[], showlegend=False),
                        row=row, col=col
                    )
                    continue
                
                x_data = data[x_col]
                y_data = data[y_col]
                
                if i == j:
                    # Diagonal: Histogram with distplot
                    hist_data = elements_df[x_col].dropna().values
                    
                    if len(hist_data) > 1:
                        try:
                            # Create distplot
                            distplot_fig = ff.create_distplot(
                                [hist_data], 
                                [selected_elements_titles[i]], 
                                bin_size=(hist_data.max() - hist_data.min()) / 20,
                                show_rug=False,
                                colors=['steelblue']
                            )
                            
                            # Add histogram trace
                            fig.add_trace(
                                go.Histogram(
                                    x=hist_data,
                                    nbinsx=20,
                                    name=f'{selected_elements_titles[i]}',
                                    showlegend=False,
                                    marker_color='steelblue',
                                    opacity=0.7,
                                    histnorm='probability density',
                                    hovertemplate=(
                                        f'{selected_elements_titles[i]}<br>'
                                        'Value: %{x}<br>'
                                        'Density: %{y:.3f}<extra></extra>'
                                    )
                                ),
                                row=row, col=col
                            )
                            
                            # Add KDE trace
                            for trace in distplot_fig.data:
                                if 'scatter' in str(type(trace)).lower():
                                    fig.add_trace(
                                        go.Scatter(
                                            x=trace.x,
                                            y=trace.y,
                                            mode='lines',
                                            line=dict(color='red', width=2),
                                            showlegend=False,
                                            hovertemplate=(
                                                f'{selected_elements_titles[i]}<br>'
                                                'Value: %{x}<br>'
                                                'KDE: %{y:.3f}<extra></extra>'
                                            )
                                        ),
                                        row=row, col=col
                                    )
                            
                        except Exception as e:
                            # Fallback to simple histogram
                            fig.add_trace(
                                go.Histogram(
                                    x=hist_data,
                                    nbinsx=20,
                                    name=f'{selected_elements_titles[i]}',
                                    showlegend=False,
                                    marker_color='steelblue',
                                    opacity=0.7,
                                    histnorm='probability density'
                                ),
                                row=row, col=col
                            )
                    
                    else:
                        fig.add_trace(
                            go.Histogram(
                                x=hist_data,
                                nbinsx=10,
                                name=f'{selected_elements_titles[i]}',
                                showlegend=False,
                                marker_color='steelblue',
                                opacity=0.7,
                                histnorm='probability density'
                            ),
                            row=row, col=col
                        )
                
                elif i < j:
                    # Upper triangle: Scatter plot with correlation
                    correlation = corr_matrix.loc[y_col, x_col]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            marker=dict(
                                size=4,
                                color='darkorange',
                                opacity=0.8,
                                line=dict(width=0)
                            ),
                            showlegend=False,
                            hovertemplate=f'{selected_elements_titles[j]}: %{{x}}<br>{selected_elements_titles[i]}: %{{y}}<extra></extra>'
                        ),
                        row=row, col=col
                    )
                    
                    # Add correlation annotation
                    fig.add_annotation(
                        x=max(x_data), y=max(y_data),
                        xanchor='right',
                        yanchor='top',
                        text=f'r = {correlation:.2f}',
                        showarrow=False,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1,
                        borderpad=2,
                        opacity=1,
                        row=row, col=col
                    )
                
                else:
                    # Lower triangle: 2D Density plot
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=x_data,
                            y=y_data,
                            colorscale='Viridis',
                            showscale=False,
                            ncontours=20,
                            line=dict(width=0),
                            hoverinfo='none'
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title_text="",
            title_x=0.5,
            title_font=dict(size=24, color='black'),
            height=200 * n_elements + 100,
            width=200 * n_elements + 100,
            bargap=0.1,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Set individual subplot backgrounds
        for i in range(1, n_elements + 1):
            for j in range(1, n_elements + 1):
                fig.update_xaxes(
                    showline=True,
                    linewidth=1,
                    linecolor='lightgray',
                    mirror=True,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='gray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='gray',
                    row=i, col=j
                )
                fig.update_yaxes(
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True,
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='gray',
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='gray',
                    row=i, col=j
                )
        
        # Set axis labels
        for i, element in enumerate(elements):
            # X-axis labels (bottom row)
            fig.update_xaxes(
                title_text=selected_elements_titles[i],
                row=n_elements, col=i+1,
                title_font=dict(size=10)
            )
            # Y-axis labels (first column)
            fig.update_yaxes(
                title_text=selected_elements_titles[i],
                row=i+1, col=1,
                title_font=dict(size=10)
            )
        
        # Add sample count annotation
        fig.add_annotation(
            x=1, y=1,
            xref="paper", yref="paper",
            xanchor="right", yanchor="bottom",
            text=f"Samples: {len(elements_df)} | Elements: {n_elements}",
            showarrow=False,
            bgcolor="white",
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            opacity=0.1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating pair matrix: {str(e)}")
