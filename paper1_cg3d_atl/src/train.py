#!/usr/bin/env python3
"""Train and evaluate any of the eight models compared in the CG3D paper.

Example
-------
    python src/train.py --model crc3d_mc --data data/sample_flight_data.csv \\
        --epochs 20 --batch-size 512

Run with `--model all` to train every model back to back (as in the paper's
comparison table).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.dirname(__file__))
import evaluate  # noqa: E402
import models as model_lib  # noqa: E402
import preprocessing  # noqa: E402


def build_and_fit(model_name: str, data: dict, epochs: int, batch_size: int, seed: int):
    tf.random.set_seed(seed)
    output_dim = data["output_dim"]

    if model_name == "cnn_gru":
        model = model_lib.build_cnn_gru(
            data["x_sp_train"].shape[-1], data["x_tp_train"].shape[-1], output_dim
        )
        x_train = [data["x_sp_train"], data["x_tp_train"]]
        x_test = [data["x_sp_test"], data["x_tp_test"]]

    elif model_name == "c3d":
        seq_len = data["x_train"].shape[-1]
        model = model_lib.build_c3d(seq_len, output_dim)
        x_train = model_lib.add_pseudo_channels(data["x_train"].reshape(-1, seq_len))
        x_test = model_lib.add_pseudo_channels(data["x_test"].reshape(-1, seq_len))

    elif model_name in ("crc3d", "crc3d_mc"):
        seq_len = data["x_train"].shape[-1]
        model = model_lib.build_crc3d(
            data["x_sp_train"].shape[-1], data["x_tp_train"].shape[-1], seq_len,
            output_dim, mc_dropout=(model_name == "crc3d_mc"),
        )
        c3d_train = model_lib.add_pseudo_channels(data["x_train"].reshape(-1, seq_len))
        c3d_test = model_lib.add_pseudo_channels(data["x_test"].reshape(-1, seq_len))
        x_train = [data["x_sp_train"], data["x_tp_train"], c3d_train]
        x_test = [data["x_sp_test"], data["x_tp_test"], c3d_test]

    else:
        builders = {"cnn": model_lib.build_cnn, "gru": model_lib.build_gru,
                    "lstm": model_lib.build_lstm, "mlp": model_lib.build_mlp}
        model = builders[model_name](data["x_train"].shape[-1], output_dim)
        x_train, x_test = data["x_train"], data["x_test"]

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse",
                  metrics=["mae", tf.keras.metrics.RootMeanSquaredError()])

    history = model.fit(
        x_train, data["y_train"], validation_data=(x_test, data["y_test"]),
        epochs=epochs, batch_size=batch_size, shuffle=False, verbose=2,
    )
    return model, history, x_test


def run(model_name: str, data: dict, epochs: int, batch_size: int, out_dir: str, seed: int):
    model, history, x_test = build_and_fit(model_name, data, epochs, batch_size, seed)

    y_pred = model.predict(x_test, verbose=0)
    y_true = data["y_test"]

    unified = data["unified"]
    metrics = {
        "mae_pca_space": evaluate.mae(y_true, y_pred),
        "rmse_pca_space": evaluate.rmse(y_true, y_pred),
    }
    try:
        y_pred_phys = unified.inverse_transform_y(y_pred)
        y_true_phys = unified.inverse_transform_y(y_true)
        metrics["mae_physical_units"] = evaluate.mae(y_true_phys, y_pred_phys)
        metrics["rmse_physical_units"] = evaluate.rmse(y_true_phys, y_pred_phys)
    except Exception:
        pass  # PCA/scaler shape mismatch for this model's target; skip physical-unit metrics

    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f"{model_name}.keras"))
    evaluate.save_history(history, model_name, out_dir)
    evaluate.save_metrics(metrics, model_name, out_dir)

    print(f"[{model_name}] " + ", ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to an ADS-B CSV file")
    parser.add_argument("--model", default="crc3d",
                         choices=[*model_lib.MODEL_REGISTRY, "all"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading and preprocessing {args.data} (window={args.window_size}) ...")
    data = preprocessing.prepare_datasets(args.data, window_size=args.window_size,
                                           random_state=args.seed)
    print(f"Train samples: {data['x_train'].shape[0]}, "
          f"test samples: {data['x_test'].shape[0]}, output_dim={data['output_dim']}")

    names = list(model_lib.MODEL_REGISTRY) if args.model == "all" else [args.model]
    for name in names:
        run(name, data, args.epochs, args.batch_size, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
