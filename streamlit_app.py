# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Mi Dashboard Gratuito")
df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
st.plotly_chart(px.line(df, x='x', y='y'))
