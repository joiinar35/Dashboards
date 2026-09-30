import numpy as np
import streamlit as st

from core.data import DECLINATION, INCLINATION
from core.plots import DIVERGING, show, site_map
from core.processing import reduce_to_pole, total_field_grid
from core.ui import how_to_read, page_header

page_header(
    "Reduction to the pole",
    "Recomputes the field as if it had been measured at the magnetic pole, so each anomaly sits "
    "directly above its source instead of being shifted and split into a positive/negative pair.",
)
how_to_read(
    f"""
At this site the field dips {INCLINATION}° with a declination of {DECLINATION}°, so a buried magnetic
object produces an asymmetric anomaly offset from its true position. Reduction to the pole (RTP)
removes that distortion in the wavenumber domain, assuming the magnetisation is induced by the
present-day field.

- **Processing:** the grid is extended beyond the surveyed footprint (nearest value), the edges are
  tapered with a Tukey window to limit FFT edge effects, the RTP operator is applied and the regional
  mean field is added back. Optional Gaussian smoothing damps high-frequency noise.
- **Stability:** the operator's amplification is bounded by 1/sin²(I) ≈ {1 / np.sin(np.radians(INCLINATION))**2:.1f}
  at this inclination, so no extra stabilisation is needed.
- **Difference map:** observed minus reduced field. It shows how much each anomaly moved or changed
  shape, not a measurement error.
"""
)

c1, c2 = st.columns(2, gap="large")
taper = c1.slider("Edge taper (Tukey α)", 0.0, 0.5, 0.2, 0.05, key="rtp_taper",
                  help="Fraction of the grid edge that is smoothly faded to reduce FFT artefacts")
smooth = c2.slider("Smoothing (grid cells of 1 m)", 0.0, 3.0, 0.5, 0.25, key="rtp_smooth",
                   help="Standard deviation of the Gaussian filter applied after RTP")

B = total_field_grid()
rtp, diff = reduce_to_pole(taper, smooth)
rms = float(np.sqrt(np.nanmean(diff**2)))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Inclination", f"{INCLINATION}°", border=True)
m2.metric("Declination", f"{DECLINATION}°", border=True)
m3.metric("Reduced field span", f"{np.nanmax(rtp) - np.nanmin(rtp):,.0f} nT", border=True,
          help=f"Reduced {np.nanmin(rtp):,.0f} to {np.nanmax(rtp):,.0f} nT; "
               f"observed {np.nanmin(B):,.0f} to {np.nanmax(B):,.0f} nT")
m4.metric("RMS difference", f"{rms:.0f} nT", border=True, help="Root mean square of observed − reduced")

lo, hi = float(np.nanpercentile(B, 1)), float(np.nanpercentile(B, 99))
left, right = st.columns(2, gap="large")
with left:
    show(site_map(B, title="Observed total field", cbar="B (nT)", unit="nT", zmin=lo, zmax=hi,
                  contour_labels=False, height=620))
with right:
    show(site_map(rtp, title="Reduced to the pole", cbar="B (nT)", unit="nT", zmin=lo, zmax=hi,
                  contour_labels=False, height=620))
st.caption("Both maps share the same colour scale (1st–99th percentile of the observed field) so they can be compared directly.")

dmax = float(np.nanpercentile(np.abs(diff), 98))
_, mid, _ = st.columns([1, 3, 1])
with mid:
    show(site_map(diff, title="Difference: observed − reduced", cbar="ΔB (nT)", unit="nT",
                  colorscale=DIVERGING, zmid=0, zmin=-dmax, zmax=dmax, contour_labels=False, height=620))
