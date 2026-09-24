# CNN-GRU Separated vs. Unified — 4D Flight Trajectory Prediction (IEEE Aerospace 2022)

Code for:

> H. Shafienya and A. C. Regan, **"4D Flight Trajectory Prediction based on
> ADS-B data: A comparison of CNN-GRU models,"**
> *2022 IEEE Aerospace Conference*, Big Sky, MT, 2022.
> [doi.org/10.1109/AERO53065.2022.9843822](https://doi.org/10.1109/AERO53065.2022.9843822)

## What this is

A hybrid CNN-GRU model for long-term (strategic) 4D trajectory prediction,
comparing two ways of feeding it spatio-temporal ADS-B features:

- **Separated**: spatial features (lat, lon, heading) go through a 1D-CNN,
  temporal features (time, velocity, vertical rate, hour) go through a GRU,
  and the two branches are concatenated before the output head — the
  paper's proposed architecture.
- **Unified**: all features are concatenated up front and passed through a
  single CNN followed by a GRU — the common/baseline approach.

| Architecture | Reported MAE | Reported RMSE |
|---|---|---|
| **CNN-GRU, separated (proposed)** | **0.16223** | **0.30077** |
| CNN-GRU, unified | 0.21389 | 0.36237 |

The separated architecture reduces MAE by 35.85% and RMSE by 20.48% over the
unified one (Table 2 of the paper).

![Separated vs. unified CNN-GRU](../docs/paper2_results.png)

## Repo structure

```
paper2_cnn_gru_comparison/
├── data/                       # sample_flight_data.csv is synthetic (see below)
└── src/
    ├── make_sample_data.py
    └── cnn_gru_comparison.py   # data prep + both architectures, Spyder-cell-
                                 # style (#%%), same as the original script
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

The paper trains on one month of OpenSky Network ADS-B history for ATL,
which isn't included here (too large, not ours to redistribute).
`make_sample_data.py` fabricates a short, plausible flight trajectory with
matching columns so the script runs end-to-end immediately:

```bash
python src/make_sample_data.py --rows 5000 --out data/sample_flight_data.csv
python src/cnn_gru_comparison.py
```

`cnn_gru_comparison.py` is one script, run top to bottom in `#%%` cells the
same way the original was (works as-is with `python`, or open it in
Spyder/VS Code and run cell by cell). It trains the separated model first,
then the unified one, saving each model's architecture diagram, training
history CSV, weights (`*.keras`), and an RMSE-vs-epoch plot as it goes, then
prints the final MAE/RMSE comparison between the two.

Training is slow at the paper's own settings (500 epochs, batch size 512);
lower `epochs=500` in either `.fit()` call to try it faster.

## To reproduce the paper's results

Point `DATA_PATH` (near the top of the script) at your own ADS-B extract with
columns `time, lat, lon, heading, velocity, vertrate, hour`. `window_size = 50`,
`epochs=500`, and `batch_size=512` are already set to match the paper.

## Notes on faithfulness to the original code

This is the original research script (`Third_Paper_November_3_1.py`),
cleaned up to actually run start-to-finish under current TensorFlow -- same
variable names, same `#%%` layout, same comment style. A couple of genuine
bugs were fixed along the way:

- **Interpolation was silently a no-op.** `dataset['col']` returns an
  independent copy under pandas' copy-on-write, so interpolating that copy
  "inplace" never wrote the filled values back into `dataset`. Fixed by
  assigning the interpolated series back explicitly.
- **`CuDNNGRU`** (removed in TF2) replaced with `GRU`, which uses the same
  fused cuDNN kernel automatically.
- **A dead `Dropout(0.35)` in the unified model**: it was computed but the
  final `Dense` layer was wired to the *pre-dropout* tensor instead, so the
  dropout never actually did anything. Now connected correctly.
- Output layer size was hard-coded assuming a specific PCA-reduced
  dimensionality; now computed from the data (`output_dim = Y_train.shape[1]`).
- `plt.show()` (blocks/hangs outside an interactive session) swapped for
  `plt.savefig()`.

The two small auxiliary `Dense` heads inside the CNN and GRU branches
(`model_1`, `model_2`) are built and summarized but never independently
trained, same as the original -- only the final fused model (`model_3`) is
fit against real targets. Left in as-is rather than "cleaned up," since
that's genuinely how the original experiment was structured.
