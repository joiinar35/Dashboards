import numpy as np
import streamlit as st

from core.data import load_gradiometer, load_survey
from core.plots import show, site_map
from core.processing import total_field_grid, value_at_hut
from core.ui import page_header

s = load_survey()
g = load_gradiometer()

page_header(
    "Magnetic survey",
    "Is this site quiet enough for a geomagnetic station? A good site has low magnetic gradients, "
    "no local anomalies and little cultural noise from power lines, traffic, fences or buildings.",
)

B = total_field_grid()
w, h = np.ptp(s.east), np.ptp(s.north)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Magnetometer stations", f"{len(s)}", border=True, help=f"Plus {len(g)} gradiometer points")
c2.metric("Surveyed area", f"{w:.0f} × {h:.0f} m", border=True)
c3.metric("Total field range", f"{s.B.max() - s.B.min():,.0f} nT", border=True,
          help=f"{s.B.min():,.0f} to {s.B.max():,.0f} nT")
c4.metric("Field at sensor hut", f"{value_at_hut(B):,.0f} nT", border=True,
          help="Interpolated from the magnetometer stations")

show(site_map(B, title="Total magnetic field intensity", cbar="B (nT)", unit="nT", height=680))

with st.expander("How the survey was done", icon=":material/menu_book:"):
    st.markdown(
        """
The property covers roughly one hectare. Unevenly spaced stations were measured with a portable
**proton magnetometer**, and a smaller **gradiometric survey** used two Overhauser sensors stacked
vertically 1 m apart, so their difference is the vertical gradient in nT/m.

Coordinates are shown in metres east and north of the planned sensor hut. Values between stations
are interpolated (cubic) on a 1 m grid and left blank outside the surveyed footprint. Urban sites
rarely meet the requirements, which is why a location far from most human activity was chosen.
"""
    )

st.markdown("#### Survey data")
t1, t2 = st.tabs(["Magnetometer", "Gradiometer"])
with t1:
    st.dataframe(
        s[["lat", "lon", "alt", "B", "east", "north"]], width="stretch",
        column_config={
            "lat": st.column_config.NumberColumn("Latitude (°)", format="%.5f"),
            "lon": st.column_config.NumberColumn("Longitude (°)", format="%.5f"),
            "alt": st.column_config.NumberColumn("Altitude (m)", format="%.1f"),
            "B": st.column_config.NumberColumn("B (nT)", format="%.0f"),
            "east": st.column_config.NumberColumn("East (m)", format="%.1f"),
            "north": st.column_config.NumberColumn("North (m)", format="%.1f"),
        },
    )
with t2:
    st.dataframe(
        g, width="stretch", hide_index=True,
        column_config={
            "lat": st.column_config.NumberColumn("Latitude (°)", format="%.6f"),
            "lon": st.column_config.NumberColumn("Longitude (°)", format="%.6f"),
            "sensor1": st.column_config.NumberColumn("Sensor 1 (nT)", format="%.1f"),
            "sensor2": st.column_config.NumberColumn("Sensor 2 (nT)", format="%.1f"),
            "dBz": st.column_config.NumberColumn("dBz (nT/m)", format="%.1f"),
            "east": st.column_config.NumberColumn("East (m)", format="%.1f"),
            "north": st.column_config.NumberColumn("North (m)", format="%.1f"),
        },
    )
