# Main Python script
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash import Input, Output, callback_context
import os

# Initialize the Dash app with Bootstrap theme
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "assets/style.css"  # Custom CSS
    ],
    suppress_callback_exceptions=True
)

# Define the main layout
app.layout = dbc.Container([
    # Header Row
    dbc.Row(
        dbc.Col(html.H1("Interactive Geochemical Data Dashboard",
                        className="text-center my-4"),
                width=12),
        className="mb-4 header-row"
    ),

    # Navigation Tabs
    dbc.Row([
        dbc.Col(
            dcc.Tabs(id='tabs', value='tab-data-viz', children=[
                dcc.Tab(label='Data Visualization', value='tab-data-viz',
                        className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='PCA Analysis', value='tab-pca',
                        className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Factor Analysis', value='tab-fa',
                        className='custom-tab', selected_className='custom-tab--selected'),
                dcc.Tab(label='Pair Matrix', value='tab-pair-matrix',
                        className='custom-tab', selected_className='custom-tab--selected'),
            ]),
            width=12
        )
    ], className="navigation-row"),

    # Page Content with loading indicator
    dcc.Loading(
        id="page-loading",
        type="circle",
        children=html.Div(id='page-content', className="page-content")
    ),
    
    # Store to track loaded pages
    dcc.Store(id='loaded-pages', data=[])
], fluid=True)

def load_page_module(page_name):
    """Lazy load page modules to reduce initial load time."""
    if page_name == 'tab-data-viz':
        from pages.data_viz import data_viz_layout, register_data_viz_callbacks
        return data_viz_layout, register_data_viz_callbacks
    elif page_name == 'tab-pca':
        from pages.pca_analysis import pca_analysis_layout, register_pca_analysis_callbacks
        return pca_analysis_layout, register_pca_analysis_callbacks
    elif page_name == 'tab-fa':
        from pages.factor_analysis import factor_analysis_layout, register_factor_analysis_callbacks
        return factor_analysis_layout, register_factor_analysis_callbacks
    elif page_name == 'tab-pair-matrix':
        from pages.pair_matrix import pair_matrix_layout, register_pair_matrix_callbacks
        return pair_matrix_layout, register_pair_matrix_callbacks
    return None, None

# Callback to switch between pages with lazy loading
@app.callback(
    [Output('page-content', 'children'),
     Output('loaded-pages', 'data')],
    [Input('tabs', 'value')],
    [dash.dependencies.State('loaded-pages', 'data')]
)
def render_page(tab, loaded_pages):
    # Get current page layout and callback registrar
    page_layout, register_callbacks = load_page_module(tab)
    
    # Register callbacks only once for this page
    if tab not in loaded_pages and register_callbacks:
        try:
            register_callbacks(app)
            loaded_pages.append(tab)
        except Exception as e:
            print(f"Error registering callbacks for {tab}: {e}")
    
    return page_layout, loaded_pages

# Simple home page layout for initial load
@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Input('tabs', 'value'),
    prevent_initial_call=True
)
def initial_load(tab):
    # Return minimal content initially
    return html.Div([
        html.H3("Loading...", className="text-center"),
        dbc.Spinner(color="primary")
    ])

# ⚠️ IMPORTANT for Render
server = app.server

if __name__ == '__main__':
    # Para producción en Render, usar debug=False
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(
        debug=debug_mode,
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8050))
    )