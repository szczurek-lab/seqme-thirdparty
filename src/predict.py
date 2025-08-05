import tensorflow as tf

tf.keras.utils.disable_interactive_logging()

import logging

logging.basicConfig(level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore")


from pathlib import Path
from typing import Literal

import numpy as np
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Masking
from keras.models import Model

from layers import Attention, MultiHeadAttention

MAX_LEN = 200  # max length for input sequences


def predict(
    sequences: list[str],
    model_type: Literal["balanced", "imbalanced"] = "balanced",
    n_ensembles: int = 5,
    batch_size: int = 128,
) -> np.ndarray:
    repo_path = Path(__file__).resolve().parents[1]
    model_dir = repo_path / "models"

    if n_ensembles > 5:
        raise ValueError("5 ensembles are available!")

    models = [
        model_dir / model_type / f"AMPlify_{model_type}_model_weights_{i + 1}.h5"
        for i in range(n_ensembles)
    ]
    out_model = load_multi_model(models, build_amplify)

    X_seq_valid = one_hot_padding(sequences, MAX_LEN)
    y_score_valid, y_indv_list_valid = ensemble(out_model, X_seq_valid, batch_size)

    return y_score_valid


def one_hot_padding(seq_list, padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences,
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = [
        "A",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "V",
        "W",
        "Y",
    ]
    for i in range(len(aa)):
        one_hot[aa[i]] = [0] * 20
        one_hot[aa[i]][i] = 1
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0] * 20] * (padding - len(seq_list[i]))
        feat_list.append(feat)
    return np.array(feat_list)


def build_amplify():
    """
    Build the complete model architecture
    """
    inputs = Input(shape=(MAX_LEN, 20), name="Input")
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name="Masking")(inputs)
    hidden = Bidirectional(
        LSTM(512, use_bias=True, dropout=0.5, return_sequences=True),
        name="Bidirectional-LSTM",
    )(masking)
    hidden = MultiHeadAttention(
        head_num=32,
        activation="relu",
        use_bias=True,
        return_multi_attention=False,
        name="Multi-Head-Attention",
    )(hidden)
    hidden = Dropout(0.2, name="Dropout_1")(hidden)
    hidden = Attention(name="Attention")(hidden)
    prediction = Dense(1, activation="sigmoid", name="Output")(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    return model


def load_multi_model(model_dir_list, architecture):
    """
    Load multiple models with the same architecture in one function.
    Input: list of saved model weights files.
    Output: list of loaded models.
    """
    model_list = []
    for i in range(len(model_dir_list)):
        model = architecture()
        model.load_weights(model_dir_list[i], by_name=True)
        model_list.append(model)
    return model_list


def ensemble(model_list, X, batch_size: int):
    """
    Ensemble the list of models with processed input X,
    Return results for ensemble and individual models
    """
    indv_pred = []  # list of predictions from each individual model
    for i in range(len(model_list)):
        indv_pred.append(model_list[i].predict(X, batch_size=batch_size).flatten())
    ens_pred = np.mean(np.array(indv_pred), axis=0)
    return ens_pred, np.array(indv_pred)


if __name__ == "__main__":
    scores = predict(["KKKK", "RRRRRR"])
    print(scores)
