from pathlib import Path

import numpy as np


def embed(sequences: list[str], batch_size: int = None) -> np.ndarray:
    repo_path = Path(__file__).resolve().parents[2]

    weights_path = repo_path / "weights.csv"
    weights = np.loadtxt(weights_path, delimiter=",")

    seq_lengths = np.array([len(seq) for seq in sequences])
    embeddings = seq_lengths[:, None] * weights

    return embeddings


if __name__ == "__main__":
    embed(sequences=["ABC", "AAAA"])
