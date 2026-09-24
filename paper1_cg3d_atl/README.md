# CG3D — 4D Flight Trajectory Prediction (Transportation Research Part C)

Code for:

> H. Shafienya and A. C. Regan, **"4D flight trajectory prediction using a hybrid
> Deep Learning prediction method based on ADS-B technology: A case study of
> Hartsfield–Jackson Atlanta International Airport (ATL),"**
> *Transportation Research Part C: Emerging Technologies*, vol. 144, 2022, 103878.
> [doi.org/10.1016/j.trc.2022.103878](https://doi.org/10.1016/j.trc.2022.103878)

## What this is

4D trajectory prediction (latitude, longitude, altitude/heading, and time)
for aircraft approaching ATL, built from historical ADS-B data. The paper
proposes **CG3D**: a fusion of a hybrid CNN-GRU (spatial features through a
1D-CNN, temporal features through a GRU) with a 3D-CNN (C3D) branch, plus a
Monte-Carlo-Dropout variant for uncertainty estimation. CG3D is compared
against six baselines: CNN, GRU, LSTM, MLP, C3D alone, and CNN-GRU alone.

| Model | Reported MAE | Reported RMSE |
|---|---|---|
| CNN-GRU | 0.2164 | 0.3728 |
| 3D-CNN | 0.1785 | 0.2646 |
| **CG3D** | **0.1776** | **0.2626** |
| CG3D + MC-Dropout | **0.1406** | **0.2231** |

(From Tables 3–4 of the paper; errors are on PCA-reduced, standardized
targets — see "Notes on faithfulness" below.)

![CG3D vs. baselines](../docs/paper1_results.png)

## Repo structure

```
paper1_cg3d_atl/
├── data/                     # sample_flight_data.csv is synthetic (see below)
└── src/
    ├── make_sample_data.py   # generates a demo trajectory
    └── cg3d_model.py         # data prep + all 8 models (CNN, GRU, LSTM, MLP,
                               # C3D, CNN-GRU, CG3D, CG3D+MCDropout), Spyder-
                               # cell-style (#%%), same as the original script
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
```
`plot_model()` also needs the `graphviz` system package (not just the `pydot`
pip package): `brew install graphviz` on macOS, `apt-get install graphviz` on
Ubuntu.

## Quickstart (synthetic demo data)

The real experiments use OpenSky Network ADS-B history for ATL, which is too
large and not ours to redistribute. `make_sample_data.py` fabricates a short,
physically plausible single-flight trajectory with the same columns so the
script can be run immediately:

```bash
python src/make_sample_data.py --rows 5000 --out data/sample_flight_data.csv
python src/cg3d_model.py
```

`cg3d_model.py` is one script, run top to bottom, organized into `#%%` cells
the same way the original research script was (works as-is with `python`, or
open it in Spyder/VS Code and run cell by cell interactively). It builds and
trains all 8 models in sequence -- CNN, GRU, LSTM, MLP, C3D, CNN-GRU, CG3D
(named `CRC3D` in the code), and CG3D+MC-Dropout -- and prints a final MAE/RMSE
comparison. Each model also saves its own architecture diagram (`*_model.png`,
needs `graphviz`), training history (`history_*.csv`), and trained weights
(`*.keras`) into the working directory as it goes, matching the original
script's habit of saving artifacts along the way rather than only at the end.

Training is slow at the paper's own settings (500 epochs, batch size 512); to
try it faster, lower `epochs=500` near the top of each model's `.fit()` call.

## To reproduce the paper's results

Point `DATA_PATH` (near the top of the script) at your own ADS-B extract with
columns `time, lat, lon, heading, velocity, vertrate, hour` (see
[OpenSky Network](https://opensky-network.org/) for historical data access).
`window_size = 100`, `epochs=500`, and `batch_size=512` are already set to
match the paper's training setup.

## Notes on faithfulness to the original code

This is the original research script (`Final_models_Prof_Regan_6.py`),
cleaned up to actually run start-to-finish under current TensorFlow -- same
variable-naming convention, same `#%%` cell layout, same comment style. A
handful of genuine bugs were fixed along the way, since a portfolio piece
should actually run correctly rather than merely look like the original:

- **Interpolation was silently a no-op.** `dataset['lat']` returns an
  independent copy under pandas' copy-on-write, so interpolating that copy
  "inplace" never wrote the filled values back into `dataset`. Missing
  values were reaching the model unfilled. Fixed by assigning the
  interpolated series back explicitly.
- **`CuDNNGRU`/`CuDNNLSTM`** (removed in TF2, so the original no longer even
  imports) replaced with `GRU`/`LSTM`, which use the same fused cuDNN kernel
  automatically under the same conditions.
- **Two spots called `BatchNormalization(x)`** instead of
  `BatchNormalization()(x)` -- constructing the layer with `x` as an argument
  rather than calling the layer on `x` -- fixed.
- **A stale-shape reshape bug in the CRC3D section**: `X_Sp_train` /
  `X_Tp_train` are already reshaped to `(N, 1, F)` earlier in the CNN-GRU
  section; a leftover reshape further down tried to treat them as 2D again,
  which corrupts the array (or crashes, depending on shapes). This only
  didn't surface in the original because Spyder-style cell-by-cell reruns
  can regenerate those arrays fresh out of order -- as a straight-through
  script it needed removing the redundant reshape.
- **Output layer sizes were hard-coded** (e.g. `Dense(6)`) assuming a
  specific PCA-reduced dimensionality; now computed from the data
  (`output_dim = Y_train.shape[1]`) so it adapts to whatever PCA keeps.
- A no-op data-chunking loop that reread the whole CSV without changing
  behavior was removed, and `plt.show()` (which blocks/hangs outside an
  interactive session) was swapped for `plt.savefig()`.
