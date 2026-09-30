"""Data loading and shared metadata. Everything here is cached once per process."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
LOCAL_CSV = ROOT / "data" / "geochem_clean.csv"
REMOTE_CSV = (
    "https://raw.githubusercontent.com/joiinar35/Dashboards/main/"
    "geochem_st/data/geochem_clean.csv"
)
COORDS = ("x_utm", "y_utm")

ELEMENT_LABELS = {
    "ba_ppm": "Ba (ppm)",
    "co_ppm": "Co (ppm)",
    "cr_ppm": "Cr (ppm)",
    "cu_ppm": "Cu (ppm)",
    "fe2o3_pct": "Fe₂O₃ (%)",
    "ni_ppm": "Ni (ppm)",
    "p_ppm": "P (ppm)",
    "pb_ppm": "Pb (ppm)",
    "v_ppm": "V (ppm)",
    "y_ppm": "Y (ppm)",
    "zn_ppm": "Zn (ppm)",
}


def label(col: str) -> str:
    return ELEMENT_LABELS.get(col, col)


def symbol(col: str) -> str:
    """Short element symbol without the unit, e.g. 'Fe₂O₃'."""
    return label(col).split(" (")[0]


def unit(col: str) -> str:
    lab = label(col)
    return lab[lab.find("(") + 1 : -1] if "(" in lab else ""


@st.cache_data(show_spinner="Loading geochemical data…")
def load_data() -> pd.DataFrame:
    """Read the bundled CSV; fall back to the GitHub copy only if it is missing."""
    source = LOCAL_CSV if LOCAL_CSV.exists() else REMOTE_CSV
    df = pd.read_csv(source)
    # Downcast to float32 where safe: halves memory and speeds up every numeric op.
    num = df.select_dtypes(include=[np.number]).columns.difference(COORDS)
    df[num] = df[num].astype("float32")
    return df


def element_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in COORDS]


@st.cache_data(show_spinner=False)
def analysis_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete-case element table and its standardised version."""
    df = load_data()
    raw = df[element_columns(df)].dropna().astype("float64")
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(raw), columns=raw.columns, index=raw.index
    )
    return raw, scaled
