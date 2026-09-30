"""Plotly look & feel shared by every chart, plus reusable figure builders."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Brand palette, taken from the GEO Analytics logo.
NAVY = "#16212D"
PANEL = "#1F2D3D"
LINE = "#2B3D52"
TEXT = "#E4EAF1"
MUTED = "#8FA1B5"
CORAL = "#FF5E38"
CYAN = "#3FE0EE"
AMBER = "#F5B041"
CATEGORICAL = [CORAL, CYAN, AMBER, "#9B8CF2", "#6FD08C", "#F28DB2"]

# Diverging scale for loadings/correlations: cool = negative, warm = positive.
DIVERGING = [
    [0.0, "#0B6F8A"],
    [0.25, "#35BFD6"],
    [0.5, "#E9EEF3"],
    [0.75, "#FF8A66"],
    [1.0, "#C73512"],
]
SEQUENTIAL = "Viridis"  # perceptually uniform, readable on dark backgrounds
FONT = "Lato, 'Helvetica Neue', Arial, sans-serif"

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

_axis = dict(
    gridcolor="rgba(228,234,241,0.07)",
    linecolor=LINE,
    zerolinecolor="rgba(228,234,241,0.15)",
    tickfont=dict(color=MUTED, size=12),
    title=dict(font=dict(color=MUTED, size=13)),
    automargin=True,
)

pio.templates["geo"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, color=TEXT, size=13),
        # Explicit page colour: Streamlit repaints fully transparent backgrounds.
        paper_bgcolor=NAVY,
        plot_bgcolor=NAVY,
        colorway=CATEGORICAL,
        title=dict(font=dict(size=16, color=TEXT), x=0, xanchor="left", pad=dict(l=4)),
        xaxis=_axis,
        yaxis=_axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor="#0E1620", bordercolor=LINE, font=dict(family=FONT, color=TEXT)),
        margin=dict(l=10, r=10, t=48, b=10),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))),
    )
)
pio.templates.default = "geo"


def show(fig: go.Figure, **kwargs):
    """Render with our template (theme=None keeps Streamlit from overriding it)."""
    # Colours set on the layout itself: Streamlit repaints template-level backgrounds.
    fig.update_layout(template=pio.templates["geo"], paper_bgcolor=NAVY, plot_bgcolor=NAVY)
    st.plotly_chart(fig, theme=None, config=CONFIG, **kwargs)


def colorbar(title: str, **extra) -> dict:
    return dict(
        title=dict(text=title, side="right", font=dict(color=MUTED, size=12)),
        outlinewidth=0,
        thickness=12,
        tickfont=dict(color=MUTED, size=11),
        **extra,
    )


def _map_layout(fig: go.Figure, title: str, height: int):
    hidden = dict(visible=False, showgrid=False, zeroline=False)
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(**hidden, constrain="domain"),
        # scaleanchor keeps 1 m east = 1 m north, so the map isn't distorted.
        yaxis=dict(**hidden, scaleanchor="x", scaleratio=1),
        margin=dict(l=0, r=0, t=48, b=0),
    )
    return fig


def contour_map(gx, gy, Z, *, title: str, cbar: str, colorscale=SEQUENTIAL, zmid=None,
                samples: tuple | None = None, hover: str = "%{z:.2f}", height: int = 560,
                sample_color: str = "rgba(255,255,255,0.85)",
                ncontours: int = 22) -> go.Figure:
    fig = go.Figure(
        go.Contour(
            x=gx, y=gy, z=Z,
            colorscale=colorscale, zmid=zmid,
            ncontours=ncontours,
            contours=dict(coloring="fill", showlines=False),
            line=dict(width=0),
            colorbar=colorbar(cbar),
            hovertemplate=f"{hover}<extra></extra>",
            connectgaps=False,
        )
    )
    if samples is not None:
        fig.add_trace(
            go.Scattergl(
                x=samples[0], y=samples[1], mode="markers",
                marker=dict(size=4, color=sample_color, line=dict(width=0.5, color="rgba(0,0,0,0.6)")),
                hoverinfo="skip", showlegend=False,
            )
        )
    return _map_layout(fig, title, height)


def category_map(gx, gy, Z, k: int, *, title: str, samples=None, height: int = 560) -> go.Figure:
    """Discrete map (clusters): one flat colour per class, no fake gradients."""
    colors = CATEGORICAL[:k]
    scale = []
    for i, c in enumerate(colors):
        scale += [[i / k, c], [(i + 1) / k, c]]
    fig = go.Figure(
        go.Heatmap(
            x=gx, y=gy, z=Z, zmin=0.5, zmax=k + 0.5, colorscale=scale,
            colorbar=colorbar("Cluster", tickvals=list(range(1, k + 1)), ticktext=[str(i) for i in range(1, k + 1)]),
            hovertemplate="Cluster %{z:.0f}<extra></extra>", hoverongaps=False,
        )
    )
    if samples is not None:
        fig.add_trace(
            go.Scattergl(
                x=samples[0], y=samples[1], mode="markers",
                marker=dict(size=3.5, color="rgba(14,22,32,0.75)"),
                hoverinfo="skip", showlegend=False,
            )
        )
    return _map_layout(fig, title, height)


def matrix_heatmap(df: pd.DataFrame, *, title: str, cbar: str, zabs: float = 1.0,
                   lower_only: bool = False, height: int | None = None,
                   hover: str = "%{y} vs %{x}: %{z:.2f}") -> go.Figure:
    if lower_only:  # symmetric matrix: drop the redundant upper half and the 1.00 diagonal
        df = df.iloc[1:, :-1]
    z = df.values.astype(float).copy()
    if lower_only:
        z[np.triu_indices_from(z, k=1)] = np.nan
    fig = go.Figure(
        go.Heatmap(
            z=z, x=list(df.columns), y=list(df.index),
            colorscale=DIVERGING, zmin=-zabs, zmax=zabs,
            texttemplate="%{z:.2f}", textfont=dict(size=11),
            hovertemplate=f"{hover}<extra></extra>", hoverongaps=False,
            xgap=2, ygap=2, colorbar=colorbar(cbar),
        )
    )
    fig.update_layout(
        title=title,
        height=height or max(360, 34 * len(df.index) + 120),
        xaxis=dict(showgrid=False, side="bottom"),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    return fig
