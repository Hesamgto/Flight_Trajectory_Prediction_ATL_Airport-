#!/usr/bin/env python3
"""Render the papers' published comparison tables as bar charts for the READMEs.

Values are the exact numbers reported in Table 3/4 (paper 1) and Table 2
(paper 2) of the published papers -- this only redraws them as a chart.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # baseline
ORANGE = "#eb6834"    # highlight: the paper's proposed / best model
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2dd"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": MUTED,
})


def bar_panel(ax, labels, values, highlight_idx, ylabel, title):
    colors = [ORANGE if i == highlight_idx else BLUE for i in range(len(labels))]
    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                 f"{val:.3f}" if val < 1 else f"{val:.2f}", ha="center", va="bottom",
                 fontsize=9, color=TEXT)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, color=TEXT, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, max(values) * 1.2)


def make_paper1_chart(out_path: str):
    labels = ["CNN-GRU", "3D-CNN", "CG3D", "CG3D +\nMC-Dropout"]
    mae = [0.2164, 0.1785, 0.1776, 0.1406]
    rmse = [0.3728, 0.2646, 0.2626, 0.2231]
    highlight = len(labels) - 1  # CG3D + MC-Dropout is the paper's best result

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    bar_panel(axes[0], labels, mae, highlight, "MAE", "Mean Absolute Error")
    bar_panel(axes[1], labels, rmse, highlight, "RMSE", "Root Mean Squared Error")
    fig.suptitle("CG3D vs. baselines (Transportation Research Part C, 2022)",
                 fontsize=13, color=TEXT, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_paper2_chart(out_path: str):
    labels = ["CNN-GRU\n(unified)", "CNN-GRU\n(separated)"]
    mae = [0.21389, 0.16223]
    rmse = [0.36237, 0.30077]
    highlight = 1  # separated-input architecture is the paper's proposed model

    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    bar_panel(axes[0], labels, mae, highlight, "MAE", "Mean Absolute Error")
    bar_panel(axes[1], labels, rmse, highlight, "RMSE", "Root Mean Squared Error")
    fig.suptitle("Separated vs. unified CNN-GRU (IEEE Aerospace 2022)",
                 fontsize=13, color=TEXT, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    make_paper1_chart(os.path.join(here, "paper1_results.png"))
    make_paper2_chart(os.path.join(here, "paper2_results.png"))
    print("Wrote docs/paper1_results.png and docs/paper2_results.png")
