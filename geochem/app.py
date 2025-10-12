# Main Python script
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# Import pages
from pages.data_viz import data_viz_layout, data_viz_callbacks
from pages.pca_analysis import pca_analysis_layout, pca_analysis_callbacks
from pages.factor_analysis import factor_analysis_layout, factor_analysis_callbacks
from pages.pair_matrix import pair_matrix_layout, pair_matrix_callbacks

# Initialize the Dash app with Bootstrap theme
app = Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "/assets/style.css"  # Custom CSS
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

    # Page Content
    html.Div(id='page-content', className="page-content")
], fluid=True)

# Register callbacks from all pages
data_viz_callbacks(app)
pca_analysis_callbacks(app)
factor_analysis_callbacks(app)
pair_matrix_callbacks(app)

# Callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    Input('tabs', 'value')
)
def render_page(tab):
    if tab == 'tab-data-viz':
        return data_viz_layout
    elif tab == 'tab-pca':
        return pca_analysis_layout
    elif tab == 'tab-fa':
        return factor_analysis_layout
    elif tab == 'tab-pair-matrix':
        return pair_matrix_layout
    else:
        return data_viz_layout

#if __name__ == '__main__':
 #   app.run(debug=True, mode='inline')
