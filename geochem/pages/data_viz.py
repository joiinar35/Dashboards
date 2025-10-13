from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from utils.data_loader import df, gdf, column_title_map, numeric_cols
from scipy.interpolate import griddata
import numpy as np

# Get element columns for dropdown
element_columns = df.columns[df.columns.get_loc('ba_ppm'):df.columns.get_loc('zn_ppm')+1]
dropdown_options = [{'label': col, 'value': col} for col in element_columns]

# Layout for Data Visualization page
data_viz_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div([
            html.P("This tab provides interactive visualizations of the geochemical data from a given geographical location from a given set of samples."),
            html.P("You can select an element from the dropdown menu on the left to explore its distribution and correlations with other elements."),
            html.Ul([
                html.Li(["The", html.Strong(" Distribution of Selected Element "), "plot shows the distribution of the chosen element using a violin and box plot."]),
                html.Li(["The", html.Strong(" Interpolated Contour Map "), "visualizes the spatial distribution of the selected element using interpolated contours."]),
                html.Li(["The", html.Strong(" Correlations of Selected Element "), "heatmap displays the correlation coefficients between the selected element and all other elements."]),
                html.Li(["The", html.Strong(" Correlation Matrix "), "heatmap shows the correlation matrix for all available geochemical elements."])
            ])
        ], className="explanation-text"), width=12),
    ]),
    
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.H4("Menu", className="mb-3"),
            dcc.Dropdown(
                id='column-dropdown',
                options=dropdown_options,
                value=element_columns[0] if element_columns.any() else None,
                clearable=False
            ),
            html.Hr(),
            html.Div(id='controls')
        ], width=3, className="sidebar"),
        
        # Main content
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Graph(id='contour-map'), width=12),
            ], className="graph-container"),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='violin-boxplot-plot'), width=12),
            ], className="graph-container"),
            
            dbc.Row([
                dbc.Col(dcc.Graph(id='correlation-matrix'), width=12),
            ], className="graph-container"),
        ], width=9)
    ])
], fluid=True)

# Callbacks for Data Visualization page
def data_viz_callbacks(app):
    @app.callback(
        Output('contour-map', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_contour_map(selected_column):
        if df.empty or selected_column is None or 'x_utm' not in df.columns or 'y_utm' not in df.columns:
            return go.Figure().update_layout(title=dict(text="<b>Interpolated Contour Map: Not enough data.</b>", x=0.5, xanchor="center", y=0.9, yanchor="top", font=dict(size=16, color="black", family="Arial")))

        # Crear geometría a partir de x_utm y y_utm
        geometry = gpd.points_from_xy(df.x_utm, df.y_utm)
    
        # Crear una grilla regular para interpolación
        x_min, x_max = geometry.x.min(), geometry.x.max()
        y_min, y_max = geometry.y.min(), geometry.y.max()
    
        # Usar una densidad de grilla apropiada para el dashboard
        grid_density = 100  # Reducido para mejor rendimiento en la web
        xi = np.linspace(x_min, x_max, grid_density)
        yi = np.linspace(y_min, y_max, grid_density)
        xi, yi = np.meshgrid(xi, yi)
    
        # Interpolar los valores del elemento seleccionado
        values = df[selected_column].values
        grid_values = griddata((geometry.x, geometry.y), values, (xi, yi), method='cubic')
    
        # Filtrar valores negativos (convertirlos a 0)
        grid_values_pos = np.where(grid_values < 0, 0, grid_values)
    
        # Obtener el título personalizado o usar el nombre de columna por defecto
        title = column_title_map.get(selected_column, f'{selected_column}')
    
        # Crear la figura de contorno
        fig = go.Figure(data=go.Contour(
            z=grid_values_pos,
            x=xi[0, :],
            y=yi[:, 0],
            ncontours=25,
            colorscale='Viridis',
            contours=dict(
                coloring='fill',
                showlabels=False
            ),
            hoverinfo='z',
            hovertemplate='<b>%{z:.1f} ppm</b><extra></extra>',
            colorbar=dict(
                title='<b>' + title + '</b>',
                titleside='right'
            )
        ))
    
        # Actualizar el diseño para quitar los ticks y etiquetas de los ejes
        fig.update_layout(
            title=dict(text=f"<b>Interpolated Contour Map - {title}</b>", x=0.5, xanchor="center", y=0.9, yanchor="top", font=dict(size=16, color="black", family="Arial")),
            xaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks=''
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks=''
            ),
            height=400,
        )
    
        return fig
        pass

    @app.callback(
        Output('violin-boxplot-plot', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_violin_boxplot(selected_column):

        if df.empty or selected_column is None:
            return go.Figure().update_layout(title=dict(text="<b>Distribution Plots: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

        fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Violin Plot of {selected_column}', f'Box Plot of {selected_column}'))
    
        # Add Violin Plot
        fig.add_trace(
            go.Violin(y=df[selected_column], name='Violin', box_visible=True, meanline_visible=True),
            row=1, col=1
        )
    
        # Add Box Plot
        fig.add_trace(
            go.Box(y=df[selected_column], name='Boxplot'),
            row=1, col=2
        )
    
        # Update layout
        fig.update_layout(
            title_text=f"<b>Distribution of {selected_column}</b>",
            showlegend=False,
            height=400,
            title_x=0.5, # Center the title
            title_y=0.9 # Adjust vertical position if needed
        )
    
        return fig
# Callback to generate plots for 'ba_ppm' statistics
    @app.callback(
    Output('ba_ppm_stats_plots', 'figure'),
    Input('controls', 'children') # Dummy input, triggers on initial load
    )
    def update_ba_ppm_stats_plots(_):
        if df.empty or 'ba_ppm' not in df.columns:
            return go.Figure().update_layout(title=dict(text="<b>Barium (Ba) Statistics: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

        # Create subplots: histogram and boxplot
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram of Ba (ppm)', 'Boxplot of Ba (ppm)'))
    
        # Add Histogram
        fig.add_trace(
            go.Histogram(x=df['ba_ppm'], nbinsx=30, name='Histogram'),
            row=1, col=1
        )
    
        # Add Boxplot
        fig.add_trace(
            go.Box(y=df['ba_ppm'], name='Boxplot'),
            row=1, col=2
        )
    
        # Update layout
        fig.update_layout(
            title_text="<b>Statistics of Barium (Ba)</b>",
            showlegend=False,
            height=400,
            title_x=0.5, # Center the title
            title_y=0.9 # Adjust vertical position if needed
        )
    
        return fig
            pass

    @app.callback(
        Output('correlation-matrix', 'figure'),
        Input('column-dropdown', 'value')
    )
    def update_element_correlations(selected_column):
        if df.empty or selected_column is None:
            return go.Figure().update_layout(title=dict(text="<b>Element Correlations: Not enough data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))
    
        # Seleccionar columnas numéricas (elementos)
        elementos = df.select_dtypes(include=['float64', 'int64'])
    
        # Eliminar columnas de coordenadas si existen
        if 'x_utm' in elementos.columns:
            elementos = elementos.drop(columns=['x_utm'])
        if 'y_utm' in elementos.columns:
            elementos = elementos.drop(columns=['y_utm'])
    
        # Si el dataset es muy grande, tomar una muestra para hacer el cálculo más rápido
        if len(elementos) > 1000:
            elementos = elementos.sample(n=1000, random_state=42)
    
        # Calcular matriz de correlación
        corr_matrix = elementos.corr()
    
        # Verificar que el elemento seleccionado existe en la matriz de correlación
        if selected_column not in corr_matrix.columns:
            return go.Figure().update_layout(title=dict(text=f"<b>Element Correlations: Column '{selected_column}' not found in data.</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))


# Callback to update the full correlation matrix heatmap
@app.callback(
    Output('correlation-matrix', 'figure'),
    Input('tabs', 'value')  # Trigger when the Data Visualization tab is selected
)
def update_full_correlation_matrix(tab_value):
    if tab_value != 'tab-data-viz' or df.empty:
        return go.Figure().update_layout(title=dict(text="<b>Correlation Matrix of Geochemical Elements</b>", x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")))

    # Seleccionar columnas numéricas (elementos)
    elementos = df.select_dtypes(include=['float64', 'int64'])

    # Eliminar columnas de coordenadas si existen
    if 'x_utm' in elementos.columns:
        elementos = elementos.drop(columns=['x_utm'])
    if 'y_utm' in elementos.columns:
        elementos = elementos.drop(columns=['y_utm'])

    # Si el dataset es muy grande, tomar una muestra para hacer el gráfico más rápido
    if len(elementos) > 1000:
        elementos = elementos.sample(n=1000, random_state=42)
        print("Se tomó una muestra de 1000 puntos para acelerar la visualización de la matriz de correlación")


    # Calcular matriz de correlación
    corr_matrix = elementos.corr().round(2)
     # Redondear a 2 decimales

    # Crear heatmap de correlación con Plotly
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        hoverongaps=False,
        hovertemplate='Correlación entre %{x} y %{y}: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title='<b>Correlation Coefficient</b>',
            titleside='right'
        )
    ))

    # Añadir anotaciones (valores de correlación en cada celda)
    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            # Color del texto basado en el valor de correlación (blanco para valores extremos)
            font_color = 'white' if abs(value) > 0.7 else 'black'

            annotations.append(
                dict(
                    x=corr_matrix.columns[j],
                    y=corr_matrix.index[i],
                    text=f'{value:.2f}',
                    showarrow=False,
                    font=dict(color=font_color, size=10),
                    bgcolor='rgba(255,255,255,0.5)' if abs(value) < 0.3 else 'rgba(0,0,0,0)'
                )
            )

    fig.update_layout(
        title=dict(text='<b>Correlation Matrix of Geochemical Elements</b>', x=0.5, y=0.9, xanchor="center", yanchor="top", font=dict(size=16, color="black", family="Arial")),
        xaxis=dict(title='Elements', tickangle=-45),
        yaxis=dict(title='Elements'),
        annotations=annotations,
        height=600,
        margin=dict(l=100, r=50, t=80, b=100),
        title_x=0.5, # Center the title
        title_y=0.9 # Adjust vertical position if needed
    )

    return fig
        pass
