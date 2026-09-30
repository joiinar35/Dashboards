"""Survey data, site constants and conversion to local metric coordinates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REMOTE = "https://raw.githubusercontent.com/joiinar35/Dashboards/main/magsurvey/data/"

# Geomagnetic field at the site
INCLINATION = -40.3   # degrees
DECLINATION = -11.37  # degrees

# Landmarks (lat, lon)
OBSERVATORY = (-34.33344, -54.71229)
SENSOR_HUT = (-34.33305, -54.71218)

# Local coordinates are metres east/north of the planned sensor hut.
LAT0, LON0 = SENSOR_HUT
_phi = np.radians(LAT0)
M_PER_DEG_LAT = 111132.95 - 559.82 * np.cos(2 * _phi) + 1.175 * np.cos(4 * _phi)
M_PER_DEG_LON = 111412.84 * np.cos(_phi) - 93.5 * np.cos(3 * _phi)


def to_local(lat, lon):
    """(lat, lon) in degrees -> (east, north) in metres from the sensor hut."""
    east = (np.asarray(lon) - LON0) * M_PER_DEG_LON
    north = (np.asarray(lat) - LAT0) * M_PER_DEG_LAT
    return east, north


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    return pd.read_csv(path if path.exists() else REMOTE + name)


@st.cache_data(show_spinner=False)
def load_survey() -> pd.DataFrame:
    df = _read("magnetometria2.csv").rename(
        columns={"Latitude": "lat", "Longitude": "lon", "B(nT)": "B", "Altitude (m)": "alt"}
    )
    df["east"], df["north"] = to_local(df.lat, df.lon)
    return df.set_index("station")


@st.cache_data(show_spinner=False)
def load_gradiometer() -> pd.DataFrame:
    """Two Overhauser sensors stacked 1 m apart: dB in nT over 1 m equals nT/m."""
    df = _read("gradiometria.csv")
    df = df.iloc[:, [0, 1, 3, 4, 5]]
    df.columns = ["lat", "lon", "sensor1", "sensor2", "dBz"]
    df["east"], df["north"] = to_local(df.lat, df.lon)
    return df


LANDMARKS = {
    "Observatory": to_local(*OBSERVATORY),
    "Sensor hut": to_local(*SENSOR_HUT),
}
