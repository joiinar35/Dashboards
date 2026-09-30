"""Small UI helpers shared by all pages. Colours/fonts live in .streamlit/config.toml."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Tighter, wider canvas for chart-heavy pages */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

/* Page header: the logo's coral→cyan gradient is the one decorative accent */
.geo-header { margin-bottom: 1.4rem; }
.geo-header h1 { font-weight: 900; letter-spacing: -0.01em; margin: 0 0 .35rem 0; padding: 0; line-height: 1.1; }
.geo-header p { color: #8FA1B5; font-size: 1.05rem; margin: 0; max-width: 72ch; }
.geo-header .rule { height: 3px; width: 72px; margin-top: .9rem; border-radius: 2px;
                    background: linear-gradient(90deg, #FF5E38, #3FE0EE); }

.geo-hero h1 { font-size: clamp(2.2rem, 4.5vw, 3.4rem); font-weight: 900; line-height: 1.05;
               background: linear-gradient(90deg, #FF5E38 0%, #F5B041 45%, #3FE0EE 100%);
               -webkit-background-clip: text; background-clip: text; color: transparent;
               margin: 0 0 .6rem 0; padding: 0; }

/* Metric cards */
[data-testid="stMetricValue"] { font-weight: 700; }
[data-testid="stMetricLabel"] p { color: #8FA1B5; }

/* Sidebar footer text */
.geo-source { font-size: .82rem; color: #8FA1B5; line-height: 1.45; }
.geo-source a { color: #3FE0EE; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="geo-header"><h1>{title}</h1>{sub}<div class="rule"></div></div>',
        unsafe_allow_html=True,
    )


def how_to_read(markdown: str) -> None:
    """Keep long explanations out of the way until the user wants them."""
    with st.expander("How to read this page", icon=":material/menu_book:"):
        st.markdown(markdown)
