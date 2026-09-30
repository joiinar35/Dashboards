import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.analysis import analysis_frames, factor_analysis, factor_grids
from core.data import load_data, symbol
from core.plots import AMBER, CORAL, CYAN, DIVERGING, contour_map, matrix_heatmap, show
from core.ui import how_to_read, page_header

df = load_data()
raw, _ = analysis_frames()
max_n = max(1, raw.shape[1] - 7)

page_header(
    "Factor analysis",
    "Find the few underlying factors (lithology, mineralisation, weathering…) that explain "
    "how the elements co-vary, and see where each one dominates.",
)
how_to_read(
    """
- **Scree plot**: eigenvalues of the correlation matrix. The dashed line marks the Kaiser criterion
  (eigenvalue = 1); factors above it are usually worth keeping.
- **Explained variance**: share of the total variance captured by each retained factor, and the running total.
- **Loadings** (varimax rotation): rows are elements, columns factors. Warm cells (towards +1) mean the element
  is strongly and positively associated with the factor, cool cells (towards −1) negatively, pale cells near
  zero barely at all. Reading which elements load together on a factor is how you interpret the geological
  or geochemical process behind it.
- **Factor score maps**: interpolated factor scores per sample, showing where each factor predominates.
"""
)

n_f = st.slider("Number of factors", 1, max_n, min(4, max_n), key="fa_n")
eig, loadings, prop, cum, scores = factor_analysis(n_f)

left, right = st.columns(2, gap="large")
with left:
    idx = np.arange(1, len(eig) + 1)
    fig = go.Figure()
    fig.add_hline(y=1, line=dict(color=CORAL, width=1.5, dash="dash"),
                  annotation_text="Kaiser criterion", annotation_position="top right",
                  annotation_font=dict(color=CORAL, size=11))
    fig.add_trace(go.Scatter(
        x=idx, y=eig, mode="lines+markers", line=dict(color=CYAN, width=2),
        marker=dict(size=8, color=[CYAN if i <= n_f else "#3A4F66" for i in idx]),
        hovertemplate="Factor %{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(title="Scree plot", height=400, showlegend=False,
                      xaxis=dict(title="Factor", dtick=1), yaxis_title="Eigenvalue")
    show(fig)

with right:
    names = list(loadings.columns)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=prop, name="Individual", marker_color=CYAN,
                         hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=cum, name="Cumulative", mode="lines+markers",
                             line=dict(color=AMBER, width=2), marker=dict(size=7),
                             hovertemplate="Up to %{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(title=f"Explained variance ({cum[-1]:.0%} total)", height=400,
                      yaxis=dict(tickformat=".0%", range=[0, max(1.0, cum[-1]) * 1.05]),
                      legend=dict(orientation="h", y=-0.18, x=0))
    show(fig)

show(matrix_heatmap(loadings.rename(index=symbol), title="Loadings (varimax)", cbar="Loading",
                    height=520, hover="%{y} on %{x}: %{z:.2f}"))

st.markdown("#### Factor score maps")
grids = factor_grids(n_f)
xy = df.loc[scores.index]
samples = (xy.x_utm.values, xy.y_utm.values)
cols = st.columns(2 if n_f > 1 else 1, gap="large")
for i, (name, (gx, gy, Z)) in enumerate(grids.items()):
    top = loadings[name].abs().sort_values(ascending=False).index[:3]
    signs = "driven by " + ", ".join(f"{symbol(e)} ({'+' if loadings.loc[e, name] > 0 else '−'})" for e in top)
    with cols[i % len(cols)]:
        show(contour_map(gx, gy, Z, title=f"{name}  <span style='color:#8FA1B5;font-size:13px'>{signs}</span>",
                         cbar="Score", colorscale=DIVERGING, zmid=0, samples=samples,
                         sample_color="rgba(14,22,32,0.55)", height=380 if n_f > 1 else 600))

out = df.loc[scores.index, ["x_utm", "y_utm"]].join(scores)
st.download_button("Download factor scores (CSV)", out.to_csv(index=False).encode(),
                   file_name=f"factor_scores_{n_f}f.csv", mime="text/csv", icon=":material/download:")
