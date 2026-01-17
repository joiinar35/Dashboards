import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import GeochemistryDataProcessor
from utils.plotting import GeochemistryPlots
import plotly.express as px
import plotly.graph_objects as go

def show():
    st.title("📊 Element Distribution Analysis")
    
    # Load data
    try:
        ppm_df = pd.read_excel("data/ppm.xlsx")
    except:
        # Create demo data
        np.random.seed(42)
        ppm_df = pd.DataFrame({
            'Sample_ID': [f'Sample_{i}' for i in range(100)],
            'Au_ppm': np.random.lognormal(0, 1, 100),
            'Ag_ppm': np.random.lognormal(2, 0.5, 100),
            'Cu_ppm': np.random.lognormal(3, 0.8, 100),
            'Pb_ppm': np.random.lognormal(1, 0.7, 100),
            'Zn_ppm': np.random.lognormal(2.5, 0.6, 100),
            'Fe_ppm': np.random.lognormal(4, 0.4, 100),
            'As_ppm': np.random.lognormal(1.5, 0.9, 100),
            'Sb_ppm': np.random.lognormal(0.5, 0.8, 100)
        })
    
    # Element selection
    element_cols = [col for col in ppm_df.columns if 'ppm' in col]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_elements = st.multiselect(
            "Select elements to analyze:",
            element_cols,
            default=element_cols[:4] if len(element_cols) >= 4 else element_cols
        )
    
    with col2:
        plot_type = st.selectbox(
            "Plot type:",
            ["Box Plot", "Violin Plot", "Histogram", "ECDF"]
        )
    
    if not selected_elements:
        st.warning("Please select at least one element.")
        return
    
    st.markdown("---")
    
    # Create visualizations
    if plot_type == "Box Plot":
        fig = GeochemistryPlots.create_boxplot(ppm_df, selected_elements)
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot":
        # Melt data for violin plot
        melted_df = ppm_df[selected_elements].melt(var_name="Element", value_name="Concentration")
        
        fig = px.violin(
            melted_df,
            x="Element",
            y="Concentration",
            box=True,
            points="all",
            title="Element Distribution (Violin Plot)",
            color="Element"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Histogram":
        cols = st.columns(len(selected_elements))
        for idx, element in enumerate(selected_elements):
            with cols[idx % len(cols)]:
                fig = GeochemistryPlots.create_histogram(ppm_df, element, nbins=30)
                st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "ECDF":
        fig = go.Figure()
        
        for element in selected_elements:
            sorted_data = np.sort(ppm_df[element].dropna())
            yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            fig.add_trace(go.Scatter(
                x=sorted_data,
                y=yvals,
                mode='lines',
                name=element,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Empirical Cumulative Distribution Function",
            xaxis_title="Concentration (ppm)",
            yaxis_title="Cumulative Probability",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    
    summary_data = []
    for element in selected_elements:
        data = ppm_df[element].dropna()
        summary_data.append({
            'Element': element,
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': data.skew(),
            'Kurtosis': data.kurtosis(),
            'Q1': data.quantile(0.25),
            'Q3': data.quantile(0.75),
            'IQR': data.quantile(0.75) - data.quantile(0.25)
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)
    
    # Outlier detection
    st.subheader("🔍 Outlier Analysis")
    
    outlier_method = st.selectbox(
        "Outlier detection method:",
        ["IQR Method", "Z-score Method", "Percentile Method"]
    )
    
    threshold = st.slider(
        "Outlier threshold:",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.5
    )
    
    outlier_data = []
    for element in selected_elements:
        data = ppm_df[element].dropna()
        
        if outlier_method == "IQR Method":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        elif outlier_method == "Z-score Method":
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > threshold]
        
        else:  # Percentile Method
            lower_bound = data.quantile(threshold / 100)
            upper_bound = data.quantile(1 - threshold / 100)
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        outlier_data.append({
            'Element': element,
            'Total Samples': len(data),
            'Outliers Count': len(outliers),
            'Outlier Percentage': f"{(len(outliers) / len(data) * 100):.2f}%",
            'Min Outlier': outliers.min() if len(outliers) > 0 else np.nan,
            'Max Outlier': outliers.max() if len(outliers) > 0 else np.nan
        })
    
    outlier_df = pd.DataFrame(outlier_data)
    st.dataframe(outlier_df, use_container_width=True)
    
    # Distribution comparison
    st.subheader("📈 Distribution Comparison")
    
    compare_element = st.selectbox(
        "Select element for detailed comparison:",
        selected_elements
    )
    
    if compare_element:
        col1, col2 = st.columns(2)
        
        with col1:
            # QQ Plot
            from scipy import stats
            import matplotlib.pyplot as plt
            
            data = ppm_df[compare_element].dropna()
            fig, ax = plt.subplots(figsize=(8, 6))
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot for {compare_element}")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            # Kernel Density Estimation
            fig = px.histogram(
                ppm_df,
                x=compare_element,
                nbins=50,
                marginal="rug",
                title=f"Distribution of {compare_element} with KDE",
                opacity=0.7
            )
            
            # Add KDE curve
            import scipy.stats as sts
            data = ppm_df[compare_element].dropna()
            x_range = np.linspace(data.min(), data.max(), 1000)
            kde = sts.gaussian_kde(data)
            y_kde = kde(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde * len(data) * (data.max() - data.min()) / 50,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
            
            st.plotly_chart(fig, use_container_width=True)
