import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from core.data import element_columns, load_data, symbol
from core.plots import CORAL, MUTED, show
from core.ui import how_to_read, page_header

df = load_data()
elements = element_columns(df)

page_header(
    "Pair matrix",
    "Every pairwise relationship between the selected elements in one grid.",
)
how_to_read(
    """
- **Upper triangle**: scatter plot of each pair, with the Pearson correlation coefficient *r*.
- **Diagonal**: normalised histogram of each element with a kernel density curve.
- **Lower triangle**: 2D density contours, useful when points overlap.

Use it to spot linear trends, separate populations and outliers before moving on to PCA or factor analysis.
"""
)

c1, c2 = st.columns([3, 1], gap="medium", vertical_alignment="bottom")
selected = c1.multiselect(
    "Elements (up to 6)", elements, default=elements[:4], format_func=symbol,
    max_selections=6, key="pair_elements",
)
n_label = c2.segmented_control("Samples", ["100", "200", "500", "All"], default="200", required=True,
                               key="pair_n")

if len(selected) < 2:
    st.info("Select at least two elements to build the matrix.", icon=":material/info:")
    st.stop()


def _ax(i: int, j: int, n: int, axis: str) -> str:
    idx = i * n + j + 1
    return axis if idx == 1 else f"{axis}{idx}"


@st.cache_data(show_spinner="Building pair matrix…")
def pair_matrix(cols: tuple[str, ...], n_samples: int | None) -> go.Figure:
    data = df[list(cols)].dropna().astype("float64")
    if n_samples and len(data) > n_samples:
        data = data.sample(n=n_samples, random_state=42)
    corr = data.corr()
    n = len(cols)
    names = [symbol(c) for c in cols]

    fig = make_subplots(rows=n, cols=n, shared_xaxes=True, horizontal_spacing=0.025, vertical_spacing=0.025)
    density_scale = [[0, "rgba(63,224,238,0)"], [0.25, "rgba(63,224,238,0.25)"],
                     [0.7, "rgba(63,224,238,0.7)"], [1, "#E9FCFE"]]

    for i, yc in enumerate(cols):
        for j, xc in enumerate(cols):
            r, c = i + 1, j + 1
            if i == j:
                v = data[xc].values
                fig.add_trace(go.Histogram(x=v, nbinsx=24, histnorm="probability density",
                                           marker_color="rgba(63,224,238,0.4)", marker_line_width=0,
                                           hovertemplate=f"{names[i]} %{{x}}<extra></extra>"), r, c)
                if np.ptp(v) > 0:
                    xs = np.linspace(v.min(), v.max(), 120)
                    fig.add_trace(go.Scatter(x=xs, y=gaussian_kde(v)(xs), mode="lines",
                                             line=dict(color=CORAL, width=2), hoverinfo="skip"), r, c)
            elif i < j:
                fig.add_trace(go.Scattergl(
                    x=data[xc], y=data[yc], mode="markers",
                    marker=dict(size=4, color=CORAL, opacity=0.65),
                    hovertemplate=f"{names[j]} %{{x}}<br>{names[i]} %{{y}}<extra></extra>"), r, c)
                rv = corr.loc[yc, xc]
                fig.add_annotation(
                    text=f"r = {rv:.2f}", x=0.97, y=0.95, xanchor="right", yanchor="top",
                    xref=f"{_ax(i, j, n, 'x')} domain", yref=f"{_ax(i, j, n, 'y')} domain",
                    showarrow=False, font=dict(size=12, color="#FFFFFF" if abs(rv) >= 0.5 else MUTED),
                    bgcolor="rgba(22,33,45,0.75)", borderpad=3,
                )
            else:
                fig.add_trace(go.Histogram2dContour(
                    x=data[xc], y=data[yc], colorscale=density_scale, showscale=False,
                    ncontours=14, contours=dict(coloring="fill", showlines=False), hoverinfo="skip"), r, c)

    # One global call instead of n² per-subplot calls: much faster to build.
    fig.update_xaxes(showgrid=False, showticklabels=False, ticks="", linecolor="#2B3D52", mirror=True)
    fig.update_yaxes(showgrid=False, showticklabels=False, ticks="", linecolor="#2B3D52", mirror=True)
    for k in range(n):
        fig.update_xaxes(title_text=names[k], showticklabels=True, tickfont=dict(size=10), row=n, col=k + 1)
        fig.update_yaxes(title_text=names[k], row=k + 1, col=1)
        if k > 0:  # first-column y ticks are meaningful except on the density diagonal
            fig.update_yaxes(showticklabels=True, tickfont=dict(size=10), row=k + 1, col=1)

    side = 170 * n + 80
    fig.update_layout(height=side, showlegend=False, bargap=0.05, margin=dict(l=10, r=10, t=36, b=10),
                      title=f"{len(data):,} samples, {n} elements")
    return fig


n_samples = None if n_label == "All" else int(n_label)
show(pair_matrix(tuple(selected), n_samples))
