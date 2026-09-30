"""Heavy computations, each memoised with st.cache_data so reruns are instant."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy
import streamlit as st
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from core.data import analysis_frames, load_data

# factor_analyzer < 0.5.1 calls scipy.sum, which newer SciPy removed.
if not hasattr(scipy, "sum"):
    scipy.sum = np.sum


# ---------------------------------------------------------------- spatial grids
def _grid(x: np.ndarray, y: np.ndarray, density: int):
    """Regular grid whose cell count follows the aspect ratio of the survey area."""
    w, h = np.ptp(x), np.ptp(y)
    nx = density
    ny = max(10, int(round(density * h / w)))
    gx = np.linspace(x.min(), x.max(), nx)
    gy = np.linspace(y.min(), y.max(), ny)
    return gx, gy, *np.meshgrid(gx, gy)


def _interpolate(x, y, v, method: str, density: int):
    gx, gy, XI, YI = _grid(x, y, density)
    Z = griddata((x, y), v, (XI, YI), method=method)
    if method == "nearest":  # nearest fills everything; clip to the sampled area
        inside = Delaunay(np.column_stack([x, y])).find_simplex(np.column_stack([XI.ravel(), YI.ravel()])) >= 0
        Z = np.where(inside.reshape(Z.shape), Z, np.nan)
    return gx, gy, Z


@st.cache_data(show_spinner=False)
def element_grid(element: str, density: int = 140):
    df = load_data()
    d = df[["x_utm", "y_utm", element]].dropna()
    gx, gy, Z = _interpolate(d.x_utm.values, d.y_utm.values, d[element].values, "cubic", density)
    return gx, gy, np.clip(Z, 0, None)  # concentrations can't be negative


# ---------------------------------------------------------------- statistics
@st.cache_data(show_spinner=False)
def correlation(method: str = "pearson") -> pd.DataFrame:
    raw, _ = analysis_frames()
    return raw.corr(method=method)


@st.cache_data(show_spinner=False)
def kde_curve(element: str, n: int = 200):
    v = load_data()[element].dropna().values.astype("float64")
    xs = np.linspace(v.min(), v.max(), n)
    return xs, gaussian_kde(v)(xs)


# ---------------------------------------------------------------- PCA
@st.cache_data(show_spinner=False)
def pca_full():
    """Fit PCA once with every component; pages slice what they need."""
    _, scaled = analysis_frames()
    pca = PCA().fit(scaled)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(pca.transform(scaled), index=scaled.index, columns=names)
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_), index=scaled.columns, columns=names
    )
    return scores, loadings, pca.explained_variance_ratio_


@st.cache_data(show_spinner="Clustering samples…")
def pca_clusters(n_components: int, k: int) -> np.ndarray:
    scores, _, _ = pca_full()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scores.iloc[:, :n_components].values)
    # Relabel so cluster 1 is the largest: stable, readable legend order.
    order = np.argsort(-np.bincount(labels))
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    return remap[labels] + 1


@st.cache_data(show_spinner=False)
def cluster_grid(n_components: int, k: int, density: int = 160):
    scores, _, _ = pca_full()
    labels = pca_clusters(n_components, k)
    xy = load_data().loc[scores.index, ["x_utm", "y_utm"]]
    return _interpolate(xy.x_utm.values, xy.y_utm.values, labels.astype(float), "nearest", density)


# ---------------------------------------------------------------- factor analysis
def _patch_check_array(mod) -> None:
    """scikit-learn 1.8 renamed check_array(force_all_finite=) to ensure_all_finite=,
    which factor_analyzer 0.5.1 still uses. Translate the keyword so both versions work."""
    if getattr(mod.check_array, "_geo_patched", False):
        return
    original = mod.check_array

    def check_array(*args, **kwargs):
        if "force_all_finite" in kwargs:
            try:
                return original(*args, **kwargs)
            except TypeError:
                kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return original(*args, **kwargs)

    check_array._geo_patched = True
    mod.check_array = check_array


@st.cache_data(show_spinner="Running factor analysis…")
def factor_analysis(n_factors: int):
    import factor_analyzer.factor_analyzer as fa_mod  # imported lazily: only this page needs it

    _patch_check_array(fa_mod)

    raw, _ = analysis_frames()
    fa = fa_mod.FactorAnalyzer(rotation="varimax" if n_factors > 1 else None, n_factors=n_factors)
    fa.fit(raw)
    eig, _ = fa.get_eigenvalues()
    names = [f"Factor {i + 1}" for i in range(n_factors)]
    loadings = pd.DataFrame(fa.loadings_, index=raw.columns, columns=names)
    _, prop, cum = fa.get_factor_variance()
    scores = pd.DataFrame(fa.transform(raw), index=raw.index, columns=names)
    return eig, loadings, np.asarray(prop), np.asarray(cum), scores


@st.cache_data(show_spinner=False)
def factor_grids(n_factors: int, density: int = 120):
    *_, scores = factor_analysis(n_factors)
    xy = load_data().loc[scores.index, ["x_utm", "y_utm"]]
    x, y = xy.x_utm.values, xy.y_utm.values
    return {f: _interpolate(x, y, scores[f].values, "cubic", density) for f in scores.columns}
