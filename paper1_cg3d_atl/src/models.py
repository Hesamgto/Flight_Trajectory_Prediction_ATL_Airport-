"""Model builders for the CG3D paper (Transportation Research Part C, 2022).

Eight models are compared in the paper: CNN, GRU, LSTM, MLP, C3D, the CNN-GRU
hybrid, CRC3D (CNN-GRU + C3D, called "CG3D" in the paper text), and CG3D with
Monte-Carlo Dropout for uncertainty estimation. Every builder takes explicit
input/output dimensions inferred from the data instead of the hard-coded
literals in the original script, so the models adapt automatically to
whatever PCA reduces the data down to.
"""
from __future__ import annotations

import numpy as np
from matplotlib.pyplot import cm
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    GlobalAveragePooling3D,
    Input,
    LSTM,
    MaxPool3D,
    MaxPooling1D,
    ReLU,
)
from tensorflow.keras.models import Model


class MCDropout(Dropout):
    """Dropout that stays active at inference time (Monte-Carlo Dropout)."""

    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def add_pseudo_channels(x: np.ndarray, cmap: str = "Oranges") -> np.ndarray:
    """Map a (N, F) feature matrix to a (N, F, 1, 1, 3) 3D-conv volume.

    Reproduces the paper's approach of turning each scalar feature sequence
    into a 3-channel "image" (via a colormap) so a 3D CNN can be applied to
    it alongside the temporal axis.
    """
    scalar_map = cm.ScalarMappable(cmap=cmap)
    out = np.empty((x.shape[0], x.shape[1], 3), dtype="float32")
    for i in range(x.shape[0]):
        out[i] = scalar_map.to_rgba(x[i])[:, :-1]
    return out.reshape(x.shape[0], x.shape[1], 1, 1, 3)


def _cnn_branch(input_dim: int, name: str):
    visible = Input(shape=(1, input_dim), name=f"{name}_input")
    x = Conv1D(32, 3, padding="same", kernel_initializer="he_uniform")(visible)
    x = MaxPooling1D(3, padding="same")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)

    x = Conv1D(64, 3, padding="same", kernel_initializer="he_uniform")(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Conv1D(64, 3, padding="same", kernel_initializer="he_uniform")(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)

    x = Dense(128, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    return visible, Flatten()(x)


def _gru_branch(input_dim: int, name: str):
    visible = Input(shape=(1, input_dim), name=f"{name}_input")
    x = GRU(20, return_sequences=True)(visible)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    return visible, Flatten()(x)


def _c3d_branch(seq_len: int, name: str, dropout_cls=Dropout):
    visible = Input(shape=(seq_len, 1, 1, 3), name=f"{name}_input")
    x = visible
    filters = [32, 64, 128, 128, 256, 256]
    for f in filters:
        x = Conv3D(f, (3, 3, 3), padding="same", kernel_initializer="he_uniform")(x)
        x = MaxPool3D((2, 2, 2), padding="same")(x)
        x = ReLU()(x)
        x = BatchNormalization()(x)
    x = dropout_cls(0.2)(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(x)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = dropout_cls(0.3)(x)
    return visible, x


def build_cnn(input_dim: int, output_dim: int) -> Model:
    visible, flat = _cnn_branch(input_dim, "cnn")
    output = Dense(output_dim, activation="linear")(flat)
    return Model(inputs=visible, outputs=output, name="CNN")


def build_gru(input_dim: int, output_dim: int) -> Model:
    visible = Input(shape=(1, input_dim))
    x = GRU(32, return_sequences=True)(visible)
    x = BatchNormalization()(x)
    output = Dense(output_dim, activation="linear")(x)
    return Model(inputs=visible, outputs=Flatten()(output), name="GRU")


def build_lstm(input_dim: int, output_dim: int) -> Model:
    visible = Input(shape=(1, input_dim))
    x = LSTM(32, return_sequences=True)(visible)
    x = BatchNormalization()(x)
    output = Dense(output_dim, activation="linear")(x)
    return Model(inputs=visible, outputs=Flatten()(output), name="LSTM")


def build_mlp(input_dim: int, output_dim: int) -> Model:
    visible = Input(shape=(1, input_dim))
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(visible)
    x = BatchNormalization()(x)
    output = Dense(output_dim, activation="linear")(x)
    return Model(inputs=visible, outputs=Flatten()(output), name="MLP")


def build_c3d(seq_len: int, output_dim: int) -> Model:
    visible, features = _c3d_branch(seq_len, "c3d")
    output = Dense(output_dim, activation="linear")(features)
    return Model(inputs=visible, outputs=output, name="C3D")


def build_cnn_gru(spatial_dim: int, temporal_dim: int, output_dim: int) -> Model:
    cnn_in, cnn_flat = _cnn_branch(spatial_dim, "cnn_gru_cnn")
    gru_in, gru_flat = _gru_branch(temporal_dim, "cnn_gru_gru")

    merged = Concatenate()([cnn_flat, gru_flat])
    x = Dense(512, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(merged)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(output_dim, activation="linear")(x)
    return Model(inputs=[cnn_in, gru_in], outputs=output, name="CNN-GRU")


def build_crc3d(spatial_dim: int, temporal_dim: int, seq_len: int, output_dim: int,
                 mc_dropout: bool = False) -> Model:
    """CRC3D = CNN-GRU + C3D fused ("CG3D" in the paper).

    When `mc_dropout` is True, every Dropout layer is replaced with
    `MCDropout`, matching the paper's CG3D-with-MC-Dropout uncertainty model.
    """
    dropout_cls = MCDropout if mc_dropout else Dropout

    cnn_in, cnn_flat = _cnn_branch(spatial_dim, "crc3d_cnn")
    gru_in, gru_flat = _gru_branch(temporal_dim, "crc3d_gru")
    c3d_in, c3d_flat = _c3d_branch(seq_len, "crc3d_c3d", dropout_cls=dropout_cls)

    cnn_gru_merge = Concatenate()([cnn_flat, gru_flat])
    x = Dense(512, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(cnn_gru_merge)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    cnn_gru_flat = dropout_cls(0.2)(x)

    merged = Concatenate()([cnn_gru_flat, c3d_flat])
    x = Dense(512, kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(merged)
    x = ReLU()(x)
    x = BatchNormalization()(x)
    x = dropout_cls(0.3)(x)
    output = Dense(output_dim, activation="linear")(x)

    name = "CG3D_MCDropout" if mc_dropout else "CG3D"
    return Model(inputs=[cnn_in, gru_in, c3d_in], outputs=output, name=name)


MODEL_REGISTRY = {
    "cnn": "single-input baseline",
    "gru": "single-input baseline",
    "lstm": "single-input baseline",
    "mlp": "single-input baseline",
    "c3d": "3D-CNN over pseudo-volume input",
    "cnn_gru": "spatial CNN + temporal GRU hybrid",
    "crc3d": "CG3D: CNN-GRU + C3D fusion",
    "crc3d_mc": "CG3D with Monte-Carlo Dropout",
}
