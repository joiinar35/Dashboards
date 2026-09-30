"""Interactive Geochemical Data Dashboard — entry point.

Run with:  streamlit run streamlit_app.py
"""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="Geochemical Data Dashboard",
    page_icon=str(ASSETS / "emblem.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import plots  # noqa: E402,F401  (registers the Plotly template once)
from core.ui import inject_css  # noqa: E402

inject_css()
st.logo(str(ASSETS / "banner.png"), size="large", icon_image=str(ASSETS / "emblem.png"))

pages = {
    "": [
        st.Page("views/home.py", title="Overview", icon=":material/public:", default=True),
    ],
    "Explore": [
        st.Page("views/data_visualization.py", title="Element maps", icon=":material/map:"),
        st.Page("views/pair_matrix.py", title="Pair matrix", icon=":material/grid_view:"),
    ],
    "Multivariate": [
        st.Page("views/pca.py", title="PCA & clustering", icon=":material/scatter_plot:"),
        st.Page("views/factor_analysis.py", title="Factor analysis", icon=":material/stacked_line_chart:"),
    ],
}

nav = st.navigation(pages)

with st.sidebar:
    st.markdown(
        '<div class="geo-source">Data: <em>Inventario Minero del Uruguay</em>, DINAMIGE catalogue '
        'on the MIEM <a href="https://geonetwork.miem.gub.uy/" target="_blank">GeoNetwork</a>.</div>',
        unsafe_allow_html=True,
    )

nav.run()
