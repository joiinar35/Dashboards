"""
Main Streamlit application - Entry point
"""
import streamlit as st
from PIL import Image
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Interactive Geochemical Data Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS - MOVE THIS OUTSIDE main() so it runs on all pages

def load_css():
    """Load CSS with proper error handling for Streamlit Cloud"""
    try:
        # Method 1: Try absolute path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "css", "style.css")
        
        if os.path.exists(css_path):
            with open(css_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return
        
        # Method 2: Try relative path (for local development)
        with open("assets/style.css", 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# css_path = 'https://raw.githubusercontent.com/joiinar35/Dashboards/main/geochem_st/css/style.css'
# with open(css_path, "r") as f:
#    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

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
