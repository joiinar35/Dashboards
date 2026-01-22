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
        with open("css/style.css", 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:

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
            
            # Load css sheet
            st.markdown("""
            <style>
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
                font-size: 1.3rem !important;
            }
            .main h1, .main h2, .main h3 { color: #FF5e38; }
            [data-testid="stSidebarNav"] { display: none; }
            </style>
            """, unsafe_allow_html=True)
        
if __name__ == "__main__":
    main()
