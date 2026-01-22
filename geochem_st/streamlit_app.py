"""
Main Streamlit application - Entry point
"""
import logging
from pathlib import Path

import streamlit as st
from PIL import Image

# Page configuration (must be set before any UI elements)
st.set_page_config(
    page_title="Interactive Geochemical Data Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
CURRENT_DIR = Path(__file__).resolve().parent

def load_css():
    """Load CSS with proper error handling and an embedded fallback."""
    css_file = CURRENT_DIR / "css" / "style.css"

    if css_file.exists():
        try:
            css_content = css_file.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        except Exception as exc:  # log and fall through to embedded fallback
            logging.exception("Failed to read css/style.css: %s", exc)

    # Try a relative path fallback (useful in some execution contexts)
    rel_css = Path("css") / "style.css"
    if rel_css.exists():
        try:
            css_content = rel_css.read_text(encoding="utf-8")
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        except Exception as exc:
            logging.exception("Failed to read ./css/style.css: %s", exc)

    # Embedded CSS fallback (safe default to ensure readable UI in Streamlit Cloud)
    embedded_fallback = """
    /* Embedded CSS fallback */
    .stApp { background: #000000; }
    .main .block-container { 
        background-color: black; 
        color: white; 
        padding: 20px;
    }
    [data-testid="stSidebar"] { 
        background: linear-gradient(135deg, #1d2a3a 0%, #1d2a3a 100%); 
    }
    .stButton button {
        font-weight: 800 !important;
        font-size: 1.1rem !important;
    }
    .main h1, .main h2, .main h3 { color: #FF5e38; }
    [data-testid="stSidebarNav"] { display: none; }
    """
    st.markdown(f"<style>{embedded_fallback}</style>", unsafe_allow_html=True)


# Run CSS loader at import time so it applies on every page load (Streamlit recommendation)
load_css()


def _find_banner():
    """Return a Path to banner_slim.png if it exists in likely locations, else None."""
    candidates = [
        CURRENT_DIR / "banner_slim.png",
        CURRENT_DIR / "assets" / "banner_slim.png",
        Path.cwd() / "banner_slim.png",
        Path.cwd() / "assets" / "banner_slim.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# Main app
def main():
    # Header
    st.markdown(
        """
        <div class="header">
            <h1>Interactive Geochemical Data Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Introduction section
    st.markdown("## Introduction")
    st.markdown(
        "Welcome to the Interactive Geochemical Data Dashboard. This application provides tools for analyzing and visualizing geochemical data."
    )

    # Sidebar content
    with st.sidebar:
        banner_path = _find_banner()
        if banner_path:
            try:
                img = Image.open(banner_path)
                st.image(img, use_column_width=True)
            except Exception as exc:
                logging.exception("Failed to open banner image: %s", exc)
                st.write("Banner image found but could not be displayed.")
        else:
            st.write("Banner image not found")

        # You can add additional sidebar widgets here
        st.markdown("---")
        st.caption("Geochemical Dashboard")


if __name__ == "__main__":
    main()
