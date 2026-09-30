"""Plotly look & feel shared by every chart, plus the survey map builder."""
from __future__ import annotations

import numpy as np
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



def site_map(z, *, title: str, cbar: str, unit: str, colorscale=SEQUENTIAL, zmid=None,
             zmin=None, zmax=None, stations: bool = True, gradiometer: bool = False,
             contour_labels: bool = True, height: int = 620) -> go.Figure:
    """Gridded map in metres from the sensor hut, with stations and landmarks."""
    from core.data import LANDMARKS, load_gradiometer, load_survey
    from core.processing import grid_axes

    ge, gn = grid_axes()
    contour = dict(coloring="heatmap", showlines=True, showlabels=contour_labels,
                   labelfont=dict(size=10, color="rgba(255,255,255,0.85)"), labelformat=",.0f")
    fig = go.Figure(
        go.Contour(
            x=ge, y=gn, z=z, colorscale=colorscale, zmid=zmid,
            zmin=zmin, zmax=zmax, zauto=zmax is None and zmin is None,
            ncontours=16, contours=contour, line=dict(width=0.6, color="rgba(255,255,255,0.35)"),
            colorbar=colorbar(cbar, tickformat=",.0f"), connectgaps=False,
            hovertemplate=f"E %{{x:.0f}} m, N %{{y:.0f}} m<br><b>%{{z:.1f}} {unit}</b><extra></extra>",
        )
    )
    if stations:
        s = load_survey()
        fig.add_trace(go.Scatter(
            x=s.east, y=s.north, mode="markers", name="Magnetometer stations",
            marker=dict(size=6, color="rgba(255,255,255,0.9)", line=dict(width=1, color="rgba(14,22,32,0.8)")),
            customdata=np.column_stack([s.B, s.lat, s.lon]),
            hovertemplate="Station: %{customdata[0]:.0f} nT<br>%{customdata[1]:.5f}°, %{customdata[2]:.5f}°<extra></extra>",
        ))
    if gradiometer:
        g = load_gradiometer()
        fig.add_trace(go.Scatter(
            x=g.east, y=g.north, mode="markers", name="Gradiometer points",
            marker=dict(size=8, symbol="diamond", color=CYAN, line=dict(width=1, color="rgba(14,22,32,0.9)")),
            customdata=g.dBz, hovertemplate="Gradiometer: %{customdata:.1f} nT/m<extra></extra>",
        ))
    obs, hut = LANDMARKS["Observatory"], LANDMARKS["Sensor hut"]
    fig.add_trace(go.Scatter(
        x=[obs[0]], y=[obs[1]], mode="markers", name="Observatory",
        marker=dict(size=14, symbol="square", color="#FFFFFF", line=dict(width=2, color=NAVY)),
        hovertemplate="Observatory building<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[hut[0]], y=[hut[1]], mode="markers", name="Sensor hut",
        marker=dict(size=18, symbol="star", color=CORAL, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="Planned sensor hut<extra></extra>",
    ))
    axis = dict(showgrid=True, zeroline=False, ticksuffix=" m", dtick=20)
    pad = 6  # same extent on every map so they can be compared side by side
    fig.update_layout(
        title=title, height=height,
        xaxis=dict(**axis, title="East of sensor hut", constrain="domain",
                   range=[ge.min() - pad, ge.max() + pad]),
        # scaleanchor keeps 1 m east = 1 m north on screen.
        yaxis=dict(**axis, title="North of sensor hut", scaleanchor="x", scaleratio=1,
                   range=[gn.min() - pad, gn.max() + pad]),
        legend=dict(orientation="h", y=-0.12, x=0, yanchor="top"),
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig
