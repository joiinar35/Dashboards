# Magnetic Survey Analysis

Streamlit dashboard assessing a ~1 ha site for a geomagnetic station, from a proton
magnetometer survey and an Overhauser gradiometer survey.

## Pages
- **Survey data**: total-field map, key figures and the raw data tables.
- **Magnetic gradients**: total horizontal gradient and measured vertical gradient, with values at the planned sensor hut.
- **Reduction to the pole**: RTP of the total field (adjustable taper and smoothing) and the observed − reduced difference.

All maps use local coordinates in metres east/north of the planned sensor hut, on a 1 m grid.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Structure
```
streamlit_app.py        entry point: page config, logo, navigation
.streamlit/config.toml  theme (same as the geochem dashboard)
core/data.py            data loading, site constants, lat/lon -> metres
core/processing.py      gridding, gradients, reduction to the pole (cached)
core/plots.py           Plotly template and the site map builder
core/ui.py              shared CSS and page header
views/                  one file per page
assets/                 logo images
data/                   survey CSV files
```

## Deploying on Streamlit Community Cloud
Main file path: `magsurvey/streamlit_app.py`. Community Cloud reads `.streamlit/config.toml`
only from the repository root.
