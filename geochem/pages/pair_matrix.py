from dash import dcc, html, Input, Output, dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from utils.data_loader import df, element_columns

# Layout for Pair Matrix page
pair_matrix_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("The Pair Matrix provides a comprehensive view of relationships between multiple geochemical elements:"),
            html.Ul([
                html.Li([html.Strong("Upper Triangle: "), "Scatter plots showing relationships between element pairs"]),
                html.Li([html.Strong("Diagonal: "), "Histograms with KDE showing distribution of each element"]),
                html.Li([html.Strong("Lower Triangle: "), "KDE plots showing density relationships"]),
                html.Li([html.Strong("Correlation coefficients (r) "), "are displayed in the upper triangle"])
            ]),
            html.P("This visualization helps identify patterns, correlations, and potential outliers in your geochemical data.")
        ], className="explanation-text"), width=12),
    ]),

    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Pair Matrix Controls", className="mb-3"),
            html.Label("Sample Size (for performance):"),
            dcc.Dropdown(
                id='sample-size-dropdown',
                options=[
                    {'label': 'Full Dataset', 'value': 'full'},
                    {'label': '100 samples', 'value': 100},
                    {'label': '200 samples', 'value': 200}
                ],
                value=100,
                clearable=False
            ),
            html.Hr(),
            html.Label("Select Elements to Include:"),
            dcc.Dropdown(
                id='element-selector',
                options=[{'label': col, 'value': col} for col in element_columns],
                value=element_columns[:6] if len(element_columns) >= 6 else element_columns,
                multi=True,
                placeholder="Select elements to include in pair matrix"
            ),
            dbc.Button('Generate Pair Matrix', id='generate-pair-matrix-btn', n_clicks=0, color="primary", className="mt-3"),
            html.Div(id='pair-matrix-controls-placeholder')
        ], width=3, className="sidebar"),

        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(html.Div(id='pair-matrix-image'), width=12, className="graph-container"),
            ]),
        ], width=9)
    ])
], fluid=True)

# Callbacks for Pair Matrix page
def pair_matrix_callbacks(app):
    @app.callback(
        Output('pair-matrix-image', 'children'),
        [Input('generate-pair-matrix-btn', 'n_clicks')],
        [
            dash.dependencies.State('sample-size-dropdown', 'value'),
            dash.dependencies.State('element-selector', 'value')
        ]
    )
    def update_pair_matrix(n_clicks, sample_size, selected_elements):
        if n_clicks == 0 or not selected_elements:
            return html.Div("Please select elements and click 'Generate Pair Matrix' to visualize the relationships.")

        if df.empty:
            return html.Div("No data available for pair matrix visualization.")

        # Select numeric columns and remove coordinate columns
        elements = df.select_dtypes(include=['float64', 'int64'])
        elements = elements.drop(columns=[col for col in ['x_utm', 'y_utm'] if col in elements.columns])

        # Filter to only selected elements
        try:
            elements = elements[selected_elements]
        except KeyError:
            return html.Div("Selected elements are not present in the data.")

        # Handle sampling for performance
        if sample_size != 'full' and len(elements) > int(sample_size):
            elements = elements.sample(n=int(sample_size), random_state=42)
            # Use logging in production; print is okay for debugging
            print(f"Sampled {sample_size} points for visualization performance.")

        # Create the pair matrix plot
        buf = io.BytesIO()
        try:
            # Set up the matplotlib figure
            plt.figure(figsize=(12, 10))

            # Set Seaborn style
            sns.set(style="whitegrid")

            # Create PairGrid with Seaborn
            g = sns.PairGrid(elements, diag_sharey=False)

            # Function to annotate correlation coefficient
            def corr_annot(x, y, **kws):
                r = np.corrcoef(x, y)[0][1]
                ax = plt.gca()
                # Color of text based on value of correlation
                color = 'darkred' if r > 0.5 else 'darkblue' if r < -0.5 else 'black'
                ax.annotate(f"r = {r:.2f}",
                            xy=(.8, .8), xycoords=ax.transAxes,
                            ha='center', va='center',
                            fontsize=10, weight='bold', color=color,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            # Upper: Scatter plots
            g.map_upper(sns.scatterplot, alpha=0.6, s=20, color='darkorange')

            # Diagonal: Histograms with KDE
            g.map_diag(sns.histplot, kde=True, color='steelblue', alpha=0.7)

            # Lower: KDE plots
            g.map_lower(sns.kdeplot, cmap='viridis', fill=False, alpha=0.7)

            # Add correlation annotations
            g.map_upper(corr_annot)

            # Adjust title and layout
            plt.suptitle("Pair Matrix - Geochemical Elements Correlation",
                         y=1.02, fontsize=16, weight='bold')
            plt.tight_layout()

            # Convert plot to PNG image
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
        except Exception as e:
            plt.close()
            print(f"Error creating pair matrix: {e}")
            return html.Div(f"Error creating pair matrix: {str(e)}")

        # Encode the image
        encoded_image = base64.b64encode(buf.read()).decode('ascii')

        return html.Img(src=f'data:image/png;base64,{encoded_image}',
                        style={'width': '100%', 'height': 'auto'})
