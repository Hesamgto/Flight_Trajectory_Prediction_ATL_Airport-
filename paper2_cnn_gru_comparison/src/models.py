"""Model builders for the IEEE Aerospace 2022 CNN-GRU comparison paper.

Two architectures are compared:
  * "separated": spatial features go through a 1D-CNN, temporal features
    through a GRU, and the two branches are fused before the output head.
  * "unified": all features are concatenated up front and fed through a
    single CNN followed by a GRU.

Output dimensionality is inferred from the (PCA-reduced) target instead of
hard-coded, and every layer's output actually participates in the forward
pass -- the original script computed a Dropout layer for the unified model's
GRU output but then fed the *pre-dropout* tensor to the final Dense layer,
silently discarding that regularization; this version wires it correctly.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    MaxPooling1D,
    ReLU,
)
from tensorflow.keras.models import Model


def build_cnn_gru_separated(spatial_dim: int, temporal_dim: int, output_dim: int) -> Model:
    """Spatial features -> CNN, temporal features -> GRU, fused at the end."""
    spatial_in = Input(shape=(1, spatial_dim), name="spatial_input")
    x = Conv1D(256, 6, padding="same")(spatial_in)
    x = MaxPooling1D(3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(512, 3, padding="same")(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)

    x = Conv1D(256, 3, padding="same")(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    cnn_flat = Flatten()(x)
    cnn_flat = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01),
                      activity_regularizer=regularizers.l2(0.01))(cnn_flat)

    temporal_in = Input(shape=(1, temporal_dim), name="temporal_input")
    gru_out = GRU(150, return_sequences=True)(temporal_in)
    gru_flat = Flatten()(gru_out)

    merged = Concatenate()([cnn_flat, gru_flat])
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(merged)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(x)
    output = Dense(output_dim, activation="linear")(x)

    return Model(inputs=[spatial_in, temporal_in], outputs=output, name="CNN-GRU-Separated")


def build_cnn_gru_unified(input_dim: int, output_dim: int) -> Model:
    """All features concatenated up front, through one CNN then one GRU."""
    visible = Input(shape=(1, input_dim), name="unified_input")
    x = Conv1D(256, 6, padding="same")(visible)
    x = MaxPooling1D(3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(512, 3, padding="same")(x)
    x = MaxPooling1D(3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv1D(256, 3, padding="same")(x)
    x = MaxPooling1D(6, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.1)(x)

    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01),
              activity_regularizer=regularizers.l2(0.01))(x)

    x = GRU(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    output = Dense(output_dim, activation="linear")(x)

    return Model(inputs=visible, outputs=Flatten()(output), name="CNN-GRU-Unified")


MODEL_REGISTRY = {
    "separated": "spatial CNN + temporal GRU, fused (proposed architecture)",
    "unified": "single CNN-GRU over all concatenated features (baseline)",
}
