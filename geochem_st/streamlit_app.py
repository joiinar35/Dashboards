"""
Main Streamlit application - Entry point
"""
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Interactive Geochemical Data Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS - MOVE THIS OUTSIDE main() so it runs on all pages
with open("css/style.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>Interactive Geochemical Data Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Introduction section
    st.markdown("## Introduction")
    st.markdown("Welcome to the Interactive Geochemical Data Dashboard. This application provides tools for analyzing and visualizing geochemical data.")
    
    # Sidebar content
    with st.sidebar:
        # Add the banner image at the bottom of sidebar
        try:
            st.image("banner_slim.png", use_container_width=True)
        except:
            st.write("Banner image not found")

if __name__ == "__main__":
    main()
