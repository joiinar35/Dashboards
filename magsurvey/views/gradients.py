import numpy as np
import streamlit as st

from core.plots import DIVERGING, show, site_map
from core.processing import horizontal_gradient_grid, value_at_hut, vertical_gradient_grid
from core.ui import how_to_read, page_header

page_header(
    "Magnetic gradients",
    "Gradients highlight buried magnetic sources. The sensor hut needs to sit where they are low.",
)
how_to_read(
    """
- **Total horizontal gradient** (left) is computed from the gridded total field:
  √((∂B/∂East)² + (∂B/∂North)²), in nT/m.
- **Vertical gradient** (right) is measured directly: the difference between two Overhauser sensors
  stacked 1 m apart. Positive and negative values are both significant; the scale is centred on zero.
- Colour scales saturate at the 98th percentile so a few extreme values don't hide the rest of the
  site. Hover any point to read the exact value.
- Strong gradients usually come from buried ferromagnetic objects, fences or building remains.
  The gradiometric survey covers a slightly smaller area than the magnetometer survey.
"""
)

H = horizontal_gradient_grid()
V = vertical_gradient_grid()
h_ok, v_ok = H[~np.isnan(H)], V[~np.isnan(V)]
h_hut, v_hut = value_at_hut(H), value_at_hut(V)
h_rank = (h_ok < h_hut).mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Horizontal, site median", f"{np.median(h_ok):.1f} nT/m", border=True,
          help=f"Mean {h_ok.mean():.1f}, maximum {h_ok.max():.1f} nT/m")
c2.metric("Horizontal, at sensor hut", f"{h_hut:.1f} nT/m", border=True,
          help=f"Lower than {100 - h_rank:.0f}% of the surveyed area")
c3.metric("Vertical, site median |dBz|", f"{np.median(np.abs(v_ok)):.1f} nT/m", border=True,
          help=f"Largest magnitude {np.abs(v_ok).max():.1f} nT/m")
c4.metric("Vertical, at sensor hut", "—" if np.isnan(v_hut) else f"{v_hut:.1f} nT/m", border=True)

left, right = st.columns(2, gap="large")
with left:
    show(site_map(H, title="Total horizontal gradient", cbar="dBh (nT/m)", unit="nT/m",
                  zmin=0, zmax=float(np.percentile(h_ok, 98)), contour_labels=False, height=640))
with right:
    vmax = float(np.percentile(np.abs(v_ok), 98))
    show(site_map(V, title="Vertical gradient", cbar="dBz (nT/m)", unit="nT/m", colorscale=DIVERGING,
                  zmid=0, zmin=-vmax, zmax=vmax, gradiometer=True, stations=False,
                  contour_labels=False, height=640))

if not np.isnan(h_hut):
    verdict = "a low-gradient part of the site" if h_rank < 50 else "one of the noisier parts of the site"
    st.info(
        f"The planned sensor hut sits in {verdict}: its horizontal gradient ({h_hut:.1f} nT/m) is lower "
        f"than {100 - h_rank:.0f}% of the surveyed area.",
        icon=":material/my_location:",
    )
