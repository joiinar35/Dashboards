import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import element_columns, load_data, symbol, unit
from core.plots import CORAL, show, _map_layout

df = load_data()
elements = element_columns(df)
w_km = np.ptp(df.x_utm) / 1000
h_km = np.ptp(df.y_utm) / 1000

st.markdown(
    f"""<div class="geo-hero"><h1>Interactive Geochemical<br>Data Dashboard</h1></div>
    <p style="font-size:1.12rem; max-width:68ch; color:#C9D4E0;">
    {len(df):,} geochemical samples, {len(elements)} analysed elements, collected over a
    {w_km:.0f} × {h_km:.0f} km area in Uruguay (UTM zone 21S). Map single elements, check how they
    relate to each other, and find the multivariate signatures behind them.</p>""",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 4], gap="large")

with left:
    fig = go.Figure(
        go.Scattergl(
            x=df.x_utm, y=df.y_utm, mode="markers",
            marker=dict(size=5, color=CORAL, opacity=0.85),
            hovertemplate="E %{x:,.0f}<br>N %{y:,.0f}<extra></extra>",
        )
    )
    show(_map_layout(fig, "Sample locations", 430))

with right:
    st.markdown("#### Where to start")
    st.page_link("views/data_visualization.py", label="Element maps", icon=":material/map:",
                 help="Interpolated map, distribution and correlations for one element")
    st.caption("Interpolated concentration map, distribution and correlation matrix for any element.")
    st.page_link("views/pair_matrix.py", label="Pair matrix", icon=":material/grid_view:")
    st.caption("Scatter, density and histogram grid for up to six elements at once.")
    st.page_link("views/pca.py", label="PCA & clustering", icon=":material/scatter_plot:")
    st.caption("Principal components, loadings and a k-means cluster map of the samples.")
    st.page_link("views/factor_analysis.py", label="Factor analysis", icon=":material/stacked_line_chart:")
    st.caption("Varimax-rotated factors and where each one dominates spatially.")

st.markdown("#### Element summary")


@st.cache_data(show_spinner=False)
def summary_table() -> pd.DataFrame:
    rows = []
    for c in elements:
        v = df[c].dropna()
        counts, _ = np.histogram(v, bins=24)
        rows.append(
            dict(Element=symbol(c), Unit=unit(c), Median=v.median(), Mean=v.mean(),
                 Min=v.min(), P95=v.quantile(0.95), Max=v.max(),
                 CV=v.std() / v.mean() * 100, Distribution=counts.tolist())
        )
    return pd.DataFrame(rows)


st.dataframe(
    summary_table(),
    hide_index=True,
    width="stretch",
    column_config={
        "Median": st.column_config.NumberColumn(format="%.1f"),
        "Mean": st.column_config.NumberColumn(format="%.1f"),
        "Min": st.column_config.NumberColumn(format="%.1f"),
        "P95": st.column_config.NumberColumn(format="%.1f", help="95th percentile"),
        "Max": st.column_config.NumberColumn(format="%.1f"),
        "CV": st.column_config.NumberColumn("CV (%)", format="%.0f", help="Coefficient of variation"),
        "Distribution": st.column_config.BarChartColumn("Distribution", width="medium"),
    },
)
