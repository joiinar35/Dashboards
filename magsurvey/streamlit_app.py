"""Magnetic Survey Analysis — entry point.

Run with:  streamlit run streamlit_app.py
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="Magnetic Survey Analysis",
    page_icon=str(ASSETS / "emblem.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import plots  # noqa: E402,F401  (registers the Plotly template once)
from core.ui import inject_css  # noqa: E402

inject_css()
st.logo(str(ASSETS / "banner.png"), size="large", icon_image=str(ASSETS / "emblem.png"))

nav = st.navigation([
    st.Page("views/survey.py", title="Survey data", icon=":material/explore:", default=True),
    st.Page("views/gradients.py", title="Magnetic gradients", icon=":material/gradient:"),
    st.Page("views/rtp.py", title="Reduction to the pole", icon=":material/my_location:"),
])

with st.sidebar:
    st.markdown(
        '<div class="geo-source">Proton magnetometer and Overhauser gradiometer survey of a '
        "~1 ha site, to assess it for a geomagnetic station.</div>",
        unsafe_allow_html=True,
    )

nav.run()
