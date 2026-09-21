"""Error metrics and small utilities for reporting model performance."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def save_history(history, model_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_history.csv")
    pd.DataFrame(history.history).to_csv(path, index_label="epoch")
    return path


def save_metrics(metrics: dict, model_name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path
