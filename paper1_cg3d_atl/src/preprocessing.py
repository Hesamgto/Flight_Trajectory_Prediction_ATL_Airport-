"""Data preprocessing for the CG3D 4D flight-trajectory model.

Pipeline (matches the paper): interpolate missing ADS-B samples, split the
trajectory into spatial (lat, lon, heading) and temporal (time, velocity,
vertical rate, hour) feature groups, build sliding-window
(input-window -> next-step) samples, standardize, and reduce dimensionality
with PCA. The unified dataset (all columns together) is produced the same
way for the single-input baselines (CNN, GRU, LSTM, MLP, C3D).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SPATIAL_COLUMNS = ["lat", "lon", "heading"]
TEMPORAL_COLUMNS = ["time", "velocity", "vertrate", "hour"]
ALL_COLUMNS = ["time", "lat", "lon", "heading", "velocity", "vertrate", "hour"]


def load_and_interpolate(csv_path: str, columns: list[str] = ALL_COLUMNS) -> pd.DataFrame:
    """Load an ADS-B CSV and fill gaps with cubic-spline interpolation."""
    df = pd.read_csv(csv_path, usecols=columns, low_memory=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    for col in columns:
        if col == "vertrate":
            df.loc[df.index[:1], col] = df[col].iloc[0] if not pd.isna(df[col].iloc[0]) else 0.0
        df[col] = df[col].interpolate(method="cubicspline", limit_direction="both")
    return df


@njit(cache=True)
def _sliding_window(data: np.ndarray, window_size: int):
    n = len(data) - window_size
    n = max(n, 0)
    x = np.empty((n, window_size, data.shape[1]), dtype=data.dtype)
    y = np.empty((n, data.shape[1]), dtype=data.dtype)
    for i in range(window_size, len(data)):
        x[i - window_size] = data[i - window_size:i]
        y[i - window_size] = data[i]
    return x, y


def sliding_window(data: np.ndarray, window_size: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Turn a (T, F) series into (window -> next-step) samples.

    Returns X flattened to (N, window_size * F) and Y of shape (N, F), matching
    the flatten-then-PCA approach used in the paper.
    """
    x, y = _sliding_window(np.asarray(data, dtype="float64"), window_size)
    return x.reshape(x.shape[0], -1), y


@dataclass
class ScaledSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    x_pca: PCA
    y_pca: PCA

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        """Map PCA/scaled predictions back to physical units."""
        return self.y_scaler.inverse_transform(self.y_pca.inverse_transform(y))


def scale_and_reduce(x: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                      pca_variance: float = 0.95, random_state: int = 1) -> ScaledSplit:
    """Split, standardize, and PCA-reduce a windowed (X, Y) dataset."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    x_scaler = StandardScaler().fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_test = x_scaler.transform(x_test)

    y_scaler = StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_test = y_scaler.transform(y_test)

    x_pca = PCA(n_components=pca_variance).fit(x_train)
    x_train = x_pca.transform(x_train)
    x_test = x_pca.transform(x_test)

    y_pca = PCA(n_components=pca_variance).fit(y_train)
    y_train = y_pca.transform(y_train)
    y_test = y_pca.transform(y_test)

    return ScaledSplit(x_train, x_test, y_train, y_test, x_scaler, y_scaler, x_pca, y_pca)


def prepare_datasets(csv_path: str, window_size: int = 100, test_size: float = 0.2,
                      pca_variance: float = 0.95, random_state: int = 1) -> dict:
    """Build spatial, temporal, and unified train/test tensors ready for Keras.

    All branches share the *same* train/test row split (same random_state) so
    that spatial, temporal, and unified samples line up index-for-index.
    """
    df = load_and_interpolate(csv_path)

    spatial = df[SPATIAL_COLUMNS].to_numpy(dtype="float64")
    temporal = df[TEMPORAL_COLUMNS].to_numpy(dtype="float64")
    unified = df[ALL_COLUMNS].to_numpy(dtype="float64")

    x_sp, _ = sliding_window(spatial, window_size)
    x_tp, _ = sliding_window(temporal, window_size)
    x_all, y_all = sliding_window(unified, window_size)

    sp = scale_and_reduce(x_sp, y_all, test_size, pca_variance, random_state)
    tp = scale_and_reduce(x_tp, y_all, test_size, pca_variance, random_state)
    uni = scale_and_reduce(x_all, y_all, test_size, pca_variance, random_state)

    def as_sequence(a: np.ndarray) -> np.ndarray:
        # Conv1D/GRU expect (samples, timesteps, features); the paper uses a
        # single timestep of PCA-reduced features per window.
        return a.reshape(a.shape[0], 1, a.shape[1])

    return {
        "spatial": sp,
        "temporal": tp,
        "unified": uni,
        "x_sp_train": as_sequence(sp.x_train),
        "x_sp_test": as_sequence(sp.x_test),
        "x_tp_train": as_sequence(tp.x_train),
        "x_tp_test": as_sequence(tp.x_test),
        "x_train": as_sequence(uni.x_train),
        "x_test": as_sequence(uni.x_test),
        "y_train": uni.y_train,
        "y_test": uni.y_test,
        "output_dim": uni.y_train.shape[1],
    }
