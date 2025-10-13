from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

# Simular datos
df = px.data.iris()

app = Dash(__name__)

app.layout = html.Div([
    html.H1("🌺 Dashboard Iris - Interactivo", 
            style={'textAlign': 'center'}),
    
    dcc.Dropdown(
        id='species-selector',
        options=[{'label': sp, 'value': sp} for sp in df['species'].unique()],
        value='setosa',
        style={'width': '50%', 'margin': '20px auto'}
    ),
    
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('species-selector', 'value')
)
def update_plot(selected_species):
    filtered_df = df[df['species'] == selected_species]
    fig = px.scatter(
        filtered_df, 
        x='sepal_width', 
        y='sepal_length',
        color='petal_length',
        title=f'Análisis para: {selected_species}',
        size='petal_width'
    )
    return fig

# ⚠️ IMPORTANTE para Render
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
