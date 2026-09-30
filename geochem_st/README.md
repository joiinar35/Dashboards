# Interactive Geochemical Data Dashboard

Streamlit dashboard for exploring the geochemistry of 665 samples from the
*Inventario Minero del Uruguay* (DINAMIGE / MIEM GeoNetwork).

## Pages
- **Overview**: sample locations and per-element summary statistics.
- **Element maps**: interpolated concentration map, distribution and correlation matrix.
- **Pair matrix**: scatter / density / histogram grid for up to six elements.
- **PCA & clustering**: scores, scree, loadings and a k-means cluster map.
- **Factor analysis**: varimax loadings and factor score maps.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Structure
```
streamlit_app.py        entry point: page config, logo, navigation
.streamlit/config.toml  theme (colours, fonts)
core/data.py            data loading (cached)
core/analysis.py        interpolation, PCA, k-means, factor analysis (cached)
core/plots.py           Plotly template and figure builders
core/ui.py              shared CSS and page header
views/                  one file per page
assets/                 logo images
data/geochem_clean.csv  dataset
```

## Deploying on Streamlit Community Cloud
Set the main file path to `streamlit_app.py` inside this folder
(e.g. `geochem_st/streamlit_app.py`). If the app lives in a subfolder of the
repository and the theme is not applied, copy the `.streamlit/` folder to the
repository root as well.
