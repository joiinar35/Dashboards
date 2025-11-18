# streamlit_app.py
import streamlit as st
import sys
import os

# Add pages directory to path to import page modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'pages'))

# Import page rendering functions
from page_1 import render_survey_data
from page_2 import render_magnetic_gradients
from page_3 import render_reduction_to_pole

# Page configuration
st.set_page_config(
    page_title="Magnetic Survey Analysis",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_url = 'https://raw.githubusercontent.com/joiinar35/Dashboards/main/magsurvey/assets/style.css'
    with open(css_url) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Additional CSS to hide default navigation menu
    st.markdown("""
    <style>
        /* Hide the default Streamlit sidebar navigation */
        [data-testid="stSidebarNav"] {
            display: none;
        }
        
        /* Hide the default hamburger menu if needed */
        /* #MainMenu {visibility: hidden;} */
    </style>
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Survey Data'

# Sidebar navigation
with st.sidebar:
    st.markdown('<h1 style="color: #FF5e38; font-weight: 700; text-align: center;">Magnetic Survey Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation buttons
    if st.button('📊 Survey Data', width='stretch', 
                 type='primary' if st.session_state.current_page == 'Survey Data' else 'secondary'):
        st.session_state.current_page = 'Survey Data'
        st.rerun()
    
    if st.button('🧲 Magnetic Gradients', width='stretch',
                 type='primary' if st.session_state.current_page == 'Magnetic Gradients' else 'secondary'):
        st.session_state.current_page = 'Magnetic Gradients'
        st.rerun()
    
    if st.button('🎯 Reduction to Pole', width='stretch',
                 type='primary' if st.session_state.current_page == 'Reduction to Pole' else 'secondary'):
        st.session_state.current_page = 'Reduction to Pole'
        st.rerun()

# Main content area - render selected page
if st.session_state.current_page == 'Survey Data':
    render_survey_data()
elif st.session_state.current_page == 'Magnetic Gradients':
    render_magnetic_gradients()
elif st.session_state.current_page == 'Reduction to Pole':
    render_reduction_to_pole()
