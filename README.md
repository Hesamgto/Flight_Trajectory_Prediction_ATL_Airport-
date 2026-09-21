# Flight Trajectory Prediction with Deep Learning

Research code for two published papers on 4D flight trajectory prediction
using hybrid deep learning models (CNN, GRU, 3D-CNN) trained on ADS-B
surveillance data, by **Hesam Shafienya** and **Amelia C. Regan**.

| | Paper | Venue | Code |
|---|---|---|---|
| 1 | [4D flight trajectory prediction using a hybrid Deep Learning prediction method based on ADS-B technology: A case study of Hartsfield–Jackson Atlanta International Airport (ATL)](https://doi.org/10.1016/j.trc.2022.103878) | Transportation Research Part C (2022) | [`paper1_cg3d_atl/`](paper1_cg3d_atl) |
| 2 | [4D Flight Trajectory Prediction based on ADS-B data: A comparison of CNN-GRU models](https://doi.org/10.1109/AERO53065.2022.9843822) | IEEE Aerospace Conference (2022) | [`paper2_cnn_gru_comparison/`](paper2_cnn_gru_comparison) |

Both papers tackle the same problem — predicting an aircraft's future
4D position (latitude, longitude, altitude/heading, time) from historical
ADS-B trajectories, evaluated on Hartsfield–Jackson Atlanta International
Airport (ATL) traffic from the OpenSky Network — from two angles:

- **Paper 1 (CG3D)** asks *which model family works best*, comparing a
  CNN-GRU + 3D-CNN fusion (with an optional Monte-Carlo-Dropout uncertainty
  variant) against CNN, GRU, LSTM, MLP, and 3D-CNN baselines.
- **Paper 2** asks *how should spatial and temporal features be fed into a
  CNN-GRU*, comparing a separated-input architecture (spatial → CNN,
  temporal → GRU, fused) against the common unified-input approach.

## Repository layout

```
flight-trajectory-prediction/
├── paper1_cg3d_atl/              # Transportation Research Part C
│   ├── data/                     # synthetic demo trajectory
│   ├── src/                      # preprocessing, models, train, evaluate
│   └── README.md
├── paper2_cnn_gru_comparison/    # IEEE Aerospace 2022
│   ├── data/
│   ├── src/
│   └── README.md
├── requirements.txt
└── LICENSE
```

Each paper's folder is self-contained — see its own README for the model
details, results table, and how to run it.

## Getting started

```bash
git clone https://github.com/<your-username>/flight-trajectory-prediction.git
cd flight-trajectory-prediction
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd paper1_cg3d_atl
python src/make_sample_data.py
python src/train.py --data data/sample_flight_data.csv --model crc3d_mc --epochs 20
```

Real ADS-B trajectory data (historical OpenSky Network extracts for ATL) is
not included here for size and licensing reasons; each `src/make_sample_data.py`
generates a small synthetic trajectory with the right columns so the full
pipeline can be run and inspected without external data. See each paper's
README for how to point the pipeline at your own data.

## Citation

```bibtex
@article{shafienya2022cg3d,
  title   = {4D flight trajectory prediction using a hybrid Deep Learning prediction method based on ADS-B technology: A case study of Hartsfield--Jackson Atlanta International Airport (ATL)},
  author  = {Shafienya, Hesam and Regan, Amelia C.},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume  = {144},
  pages   = {103878},
  year    = {2022},
  doi     = {10.1016/j.trc.2022.103878}
}

@inproceedings{shafienya2022cnngru,
  title     = {4D Flight Trajectory Prediction based on ADS-B data: A comparison of CNN-GRU models},
  author    = {Shafienya, Hesam and Regan, Amelia},
  booktitle = {2022 IEEE Aerospace Conference},
  year      = {2022},
  doi       = {10.1109/AERO53065.2022.9843822}
}
```

## License

[MIT](LICENSE)
