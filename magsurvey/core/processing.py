"""Gridding, gradients and reduction to the pole — all cached, all in metres."""
from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.interpolate import NearestNDInterpolator, RegularGridInterpolator, griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import windows

from core.data import DECLINATION, INCLINATION, LANDMARKS, load_gradiometer, load_survey

CELL = 1.0  # grid cell size in metres


@st.cache_data(show_spinner=False)
def grid_axes(cell: float = CELL):
    """Regular grid covering the magnetometer survey (rows = north, columns = east)."""
    df = load_survey()
    ge = np.arange(df.east.min(), df.east.max() + cell / 2, cell)
    gn = np.arange(df.north.min(), df.north.max() + cell / 2, cell)
    return ge, gn


def _grid(east, north, values, method="cubic"):
    ge, gn = grid_axes()
    E, N = np.meshgrid(ge, gn)
    return griddata((east, north), values, (E, N), method=method)


@st.cache_data(show_spinner=False)
def total_field_grid():
    df = load_survey()
    return _grid(df.east.values, df.north.values, df.B.values)


@st.cache_data(show_spinner=False)
def vertical_gradient_grid():
    g = load_gradiometer()
    return _grid(g.east.values, g.north.values, g.dBz.values)


@st.cache_data(show_spinner=False)
def horizontal_gradient_grid():
    """Total horizontal gradient sqrt((dB/dE)^2 + (dB/dN)^2) in nT/m.

    Differentiating the raw cubic (triangulated) surface shows triangle-edge artefacts, so the
    field is smoothed lightly (sigma = 2 m) before taking derivatives.
    """
    B = total_field_grid()
    footprint = ~np.isnan(B)
    smooth = gaussian_filter(_fill_nans(B), sigma=2.0 / CELL)
    dB_dN, dB_dE = np.gradient(smooth, CELL, CELL)  # grid spacing is already in metres
    H = np.hypot(dB_dE, dB_dN)
    H[~footprint] = np.nan
    return H


def value_at(grid: np.ndarray, east: float, north: float) -> float:
    ge, gn = grid_axes()
    f = RegularGridInterpolator((gn, ge), grid, bounds_error=False, fill_value=np.nan)
    return float(f([[north, east]])[0])


def value_at_hut(grid: np.ndarray) -> float:
    return value_at(grid, *LANDMARKS["Sensor hut"])


# ---------------------------------------------------------------- reduction to the pole
def _fill_nans(a: np.ndarray) -> np.ndarray:
    """Extend the grid beyond the surveyed footprint by nearest neighbour (FFT needs no gaps)."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    rows, cols = np.indices(a.shape)
    f = NearestNDInterpolator(np.column_stack([rows[~mask], cols[~mask]]), a[~mask])
    out = a.copy()
    out[mask] = f(rows[mask], cols[mask])
    return out


def rtp_filter(ny: int, nx: int, d_north: float, d_east: float, inc: float, dec: float) -> np.ndarray:
    """Wavenumber-domain RTP operator for induced magnetisation (Blakely, 1995).

    Uses the field direction for both the ambient field and the magnetisation:
        RTP(k) = |k|^2 / theta(k)^2,   theta = i(fN kN + fE kE) + fZ |k|
    with (fN, fE, fZ) the unit vector of the field (north, east, down).
    Its amplitude never exceeds 1 / sin^2(I), so at I = -40° it is naturally stable.
    """
    kN = 2 * np.pi * np.fft.fftfreq(ny, d_north)
    kE = 2 * np.pi * np.fft.fftfreq(nx, d_east)
    KE, KN = np.meshgrid(kE, kN)
    K = np.hypot(KE, KN)
    i, d = np.radians(inc), np.radians(dec)
    fN, fE, fZ = np.cos(i) * np.cos(d), np.cos(i) * np.sin(d), np.sin(i)
    theta = 1j * (fN * KN + fE * KE) + fZ * K
    with np.errstate(divide="ignore", invalid="ignore"):
        op = K**2 / theta**2
    op[0, 0] = 1.0  # keep the mean (regional field) unchanged
    return op


@st.cache_data(show_spinner="Reducing to the pole…")
def reduce_to_pole(taper: float = 0.2, smoothing: float = 0.0):
    """Return (rtp_grid, difference_grid) masked to the surveyed footprint."""
    B = total_field_grid()
    footprint = ~np.isnan(B)
    filled = _fill_nans(B)
    regional = filled[footprint].mean()
    anomaly = filled - regional
    if taper > 0:
        ny, nx = anomaly.shape
        anomaly = anomaly * np.outer(windows.tukey(ny, taper), windows.tukey(nx, taper))
    rtp = np.real(np.fft.ifft2(np.fft.fft2(anomaly) * rtp_filter(*anomaly.shape, CELL, CELL,
                                                                      INCLINATION, DECLINATION)))
    if smoothing > 0:
        rtp = gaussian_filter(rtp, smoothing)
    rtp = rtp + regional
    rtp[~footprint] = np.nan
    return rtp, B - rtp
