import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.analysis import correlation, element_grid, kde_curve
from core.data import element_columns, label, load_data, symbol, unit
from core.plots import CORAL, CYAN, contour_map, matrix_heatmap, show
from core.ui import how_to_read, page_header

df = load_data()
elements = element_columns(df)

page_header(
    "Element maps",
    "Pick an element to see where it concentrates, how its values are distributed "
    "and which elements travel with it.",
)
how_to_read(
    """
- **Interpolated map**: cubic interpolation of the sample values; white dots are the sample sites.
  Areas outside the sampled footprint are left blank rather than extrapolated.
- **Distribution**: violin with an inner box plot (median, quartiles, whiskers) and a histogram with a
  kernel density curve. Long right tails are typical of geochemical anomalies.
- **Correlation matrix**: Pearson measures linear association; Spearman uses ranks and is more robust
  to the skewed distributions and outliers common in geochemistry.
"""
)

el = st.pills(
    "Element", elements, default=elements[0], format_func=symbol, required=True,
    key="element", label_visibility="collapsed",
)

v = df[el].dropna()
u = unit(el)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Median", f"{v.median():,.1f} {u}", border=True)
m2.metric("Mean", f"{v.mean():,.1f} {u}", border=True)
m3.metric("95th percentile", f"{v.quantile(.95):,.1f} {u}", border=True)
m4.metric("Maximum", f"{v.max():,.1f} {u}", border=True, help=f"Minimum {v.min():,.1f} {u}")

# ---------------------------------------------------------------- map
gx, gy, Z = element_grid(el)
show(
    contour_map(
        gx, gy, Z,
        title=f"Interpolated concentration of {label(el)}",
        cbar=label(el),
        samples=(df.x_utm.values, df.y_utm.values),
        hover=f"%{{z:.1f}} {u}",
        height=620,
    )
)

# ---------------------------------------------------------------- distribution
fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58], horizontal_spacing=0.08,
                    subplot_titles=("Violin and box", "Histogram and density"))
fig.add_trace(
    go.Violin(
        x=v, name=symbol(el), orientation="h", side="positive", width=1.6,
        line=dict(color=CYAN, width=1.5), fillcolor="rgba(63,224,238,0.18)",
        box=dict(visible=True, fillcolor="rgba(255,94,56,0.35)", line=dict(color=CORAL), width=0.25),
        meanline=dict(visible=True, color="#FFFFFF"), points=False,
        hoveron="kde+points",
    ),
    row=1, col=1,
)
xs, dens = kde_curve(el)
fig.add_trace(
    go.Histogram(x=v, nbinsx=35, histnorm="probability density", marker_color="rgba(63,224,238,0.45)",
                 marker_line=dict(width=0), hovertemplate="%{x}<br>density %{y:.4f}<extra></extra>"),
    row=1, col=2,
)
fig.add_trace(go.Scatter(x=xs, y=dens, mode="lines", line=dict(color=CORAL, width=2.5), hoverinfo="skip"),
              row=1, col=2)
fig.update_layout(height=360, showlegend=False, bargap=0.04, margin=dict(t=60))
fig.update_yaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(title_text=label(el))
fig.update_annotations(font=dict(size=13, color="#8FA1B5"))
show(fig)

# ---------------------------------------------------------------- correlation
st.markdown("#### Correlation between elements")
method = st.segmented_control("Method", ["Pearson", "Spearman"], default="Pearson", required=True,
                              key="corr_method", label_visibility="collapsed")
corr = correlation(method.lower())
corr = corr.rename(index=symbol, columns=symbol)
show(matrix_heatmap(corr, title=f"{method} correlation matrix", cbar="r", lower_only=True, height=560))
