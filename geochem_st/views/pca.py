import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.analysis import cluster_grid, pca_clusters, pca_full
from core.data import load_data, symbol
from core.plots import AMBER, CATEGORICAL, CYAN, category_map, matrix_heatmap, show
from core.ui import how_to_read, page_header

df = load_data()
scores, loadings, evr = pca_full()
n_max = min(10, len(evr))

page_header(
    "PCA & clustering",
    "Reduce the elements to a few principal components, then group samples with similar signatures.",
)
how_to_read(
    """
- **Scores plot**: each dot is a sample on the first two components, coloured by k-means cluster.
- **Scree plot**: bars show the variance each component explains, the line the running total.
  Keep components until the curve flattens.
- **Loadings**: how strongly each element drives each component. Warm cells (towards +1) are positive
  associations, cool cells (towards −1) negative, pale cells near zero have little influence.
  Elements with large loadings of the same sign on a component tend to vary together.
- **Cluster map**: k-means is run on the retained components; each area takes the cluster of its
  nearest sample, clipped to the sampled footprint.

Data are standardised (z-scores) before PCA so elements with large ppm values don't dominate.
"""
)

c1, c2 = st.columns(2, gap="large")
n_comp = c1.slider("Components to keep", 2, n_max, 3, key="pca_n")
k = c2.slider("Clusters (k)", 2, 6, 3, key="pca_k")

labels = pca_clusters(n_comp, k)

left, right = st.columns(2, gap="large")
with left:
    fig = go.Figure()
    for c in range(1, k + 1):
        m = labels == c
        fig.add_trace(go.Scattergl(
            x=scores.PC1[m], y=scores.PC2[m], mode="markers", name=f"Cluster {c}",
            marker=dict(size=6, color=CATEGORICAL[c - 1], opacity=0.8, line=dict(width=0)),
            hovertemplate="PC1 %{x:.2f}<br>PC2 %{y:.2f}<extra>Cluster " + str(c) + "</extra>",
        ))
    fig.update_layout(
        title="Scores: PC1 vs PC2", height=420,
        xaxis_title=f"PC1 ({evr[0]:.0%})", yaxis_title=f"PC2 ({evr[1]:.0%})",
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    show(fig)

with right:
    pcs = [f"PC{i + 1}" for i in range(n_max)]
    kept = np.arange(n_max) < n_comp
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pcs, y=evr[:n_max], name="Individual",
        marker_color=[CYAN if keep else "rgba(63,224,238,0.25)" for keep in kept],
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pcs, y=np.cumsum(evr[:n_max]), name="Cumulative", mode="lines+markers",
        line=dict(color=AMBER, width=2), marker=dict(size=7),
        hovertemplate="Up to %{x}: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Explained variance ({np.sum(evr[:n_comp]):.0%} with {n_comp} components)",
        height=420, yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    show(fig)

left, right = st.columns([2, 3], gap="large")
with left:
    L = loadings.iloc[:, :n_comp].rename(index=symbol)
    zabs = max(1.0, float(np.abs(L.values).max()))
    show(matrix_heatmap(L, title="Loadings", cbar="Loading", zabs=zabs, height=560,
                        hover="%{y} on %{x}: %{z:.2f}"))

with right:
    gx, gy, Z = cluster_grid(n_comp, k)
    xy = df.loc[scores.index]
    show(category_map(gx, gy, Z, k, title=f"Cluster map (k = {k})",
                      samples=(xy.x_utm.values, xy.y_utm.values), height=560))

out = scores.iloc[:, :n_comp].copy()
out.insert(0, "cluster", labels)
out = df.loc[out.index, ["x_utm", "y_utm"]].join(out)
st.download_button(
    "Download scores and clusters (CSV)", out.to_csv(index=False).encode(),
    file_name=f"pca_{n_comp}pc_k{k}.csv", mime="text/csv", icon=":material/download:",
)
