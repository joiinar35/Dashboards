import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import GeochemistryDataProcessor
from utils.plotting import GeochemistryPlots
import plotly.express as px

def show():
    st.title("🏠 Data Overview")
    
    # Sample data loading (replace with actual data paths)
    try:
        # Load sample data
        elements_df = pd.read_excel("data/elementos.xlsx")
        ppm_df = pd.read_excel("data/ppm.xlsx")
        samples_df = pd.read_excel("data/samples.xlsx")
        
        st.success("Sample data loaded successfully!")
    except:
        st.warning("Sample data files not found. Using demo data.")
        # Create demo data
        np.random.seed(42)
        elements_df = pd.DataFrame({
            'Element': ['Au', 'Ag', 'Cu', 'Pb', 'Zn', 'Fe', 'As', 'Sb'],
            'Atomic_Number': [79, 47, 29, 82, 30, 26, 33, 51],
            'Atomic_Weight': [197.0, 107.9, 63.5, 207.2, 65.4, 55.8, 74.9, 121.8],
            'Density': [19.3, 10.5, 8.96, 11.3, 7.13, 7.87, 5.73, 6.68]
        })
        
        ppm_df = pd.DataFrame({
            'Sample_ID': [f'Sample_{i}' for i in range(100)],
            'Au_ppm': np.random.lognormal(0, 1, 100),
            'Ag_ppm': np.random.lognormal(2, 0.5, 100),
            'Cu_ppm': np.random.lognormal(3, 0.8, 100),
            'Pb_ppm': np.random.lognormal(1, 0.7, 100),
            'Zn_ppm': np.random.lognormal(2.5, 0.6, 100),
            'Latitude': np.random.uniform(-30, -25, 100),
            'Longitude': np.random.uniform(-70, -65, 100)
        })
        
        samples_df = pd.DataFrame({
            'Sample_ID': [f'Sample_{i}' for i in range(100)],
            'Depth_m': np.random.uniform(0, 200, 100),
            'Rock_Type': np.random.choice(['Granite', 'Basalt', 'Schist', 'Gneiss'], 100),
            'Date': pd.date_range('2020-01-01', periods=100, freq='D')
        })
    
    # Dataset selector
    dataset_choice = st.selectbox(
        "Select dataset to explore:",
        ["Elements Data", "PPM Concentrations", "Samples Metadata"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(ppm_df))
    
    with col2:
        st.metric("Elements Analyzed", len(elements_df))
    
    with col3:
        avg_au = ppm_df['Au_ppm'].mean() if 'Au_ppm' in ppm_df.columns else 0
        st.metric("Avg Au (ppm)", f"{avg_au:.4f}")
    
    st.markdown("---")
    
    # Display selected dataset
    if dataset_choice == "Elements Data":
        st.subheader("📋 Elements Information")
        st.dataframe(elements_df, use_container_width=True)
        
        # Element properties visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                elements_df,
                x='Element',
                y='Atomic_Weight',
                title='Atomic Weight by Element',
                color='Atomic_Weight',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                elements_df,
                x='Atomic_Number',
                y='Density',
                size='Atomic_Weight',
                color='Element',
                title='Element Properties',
                hover_name='Element'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif dataset_choice == "PPM Concentrations":
        st.subheader("📊 Concentration Data")
        
        # Data summary
        numeric_cols = ppm_df.select_dtypes(include=[np.number]).columns
        summary_stats = ppm_df[numeric_cols].describe().T
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.dataframe(ppm_df.head(10), use_container_width=True)
        
        with col2:
            st.dataframe(summary_stats, use_container_width=True)
        
        # Concentration distribution
        element_to_plot = st.selectbox(
            "Select element to visualize:",
            [col for col in ppm_df.columns if 'ppm' in col]
        )
        
        fig = GeochemistryPlots.create_histogram(ppm_df, element_to_plot)
        st.plotly_chart(fig, use_container_width=True)
    
    else:  # Samples Metadata
        st.subheader("🗂️ Samples Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(samples_df.head(10), use_container_width=True)
        
        with col2:
            # Rock type distribution
            if 'Rock_Type' in samples_df.columns:
                rock_counts = samples_df['Rock_Type'].value_counts()
                fig = px.pie(
                    values=rock_counts.values,
                    names=rock_counts.index,
                    title='Rock Type Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Depth distribution
        if 'Depth_m' in samples_df.columns:
            fig = px.histogram(
                samples_df,
                x='Depth_m',
                nbins=20,
                title='Sample Depth Distribution',
                color_discrete_sequence=['#FF9800']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Data quality metrics
    st.markdown("---")
    st.subheader("📈 Data Quality Metrics")
    
    if dataset_choice == "PPM Concentrations":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            missing_values = ppm_df.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        with col2:
            duplicate_rows = ppm_df.duplicated().sum()
            st.metric("Duplicate Rows", duplicate_rows)
        
        with col3:
            element_cols = [col for col in ppm_df.columns if 'ppm' in col]
            zero_values = (ppm_df[element_cols] == 0).sum().sum()
            st.metric("Zero Values", zero_values)
        
        with col4:
            outliers = 0
            for col in element_cols:
                z_scores = np.abs((ppm_df[col] - ppm_df[col].mean()) / ppm_df[col].std())
                outliers += (z_scores > 3).sum()
            st.metric("Potential Outliers", int(outliers))
    
    st.info("💡 Tip: Use the sidebar to upload your own data or select different sample datasets.")
