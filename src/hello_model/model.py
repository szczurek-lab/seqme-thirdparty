from pathlib import Path

import numpy as np


def embed(sequences: list[str], batch_size: int = None) -> np.ndarray:
    """Embed sequences into fixed-size vectors using precomputed weights.

    Args:
        sequences: List of string sequences to embed.
        batch_size: Unused. Reserved for future batched processing.

    Returns:
        Array of shape (len(sequences), embedding_dim) where each row is
        the embedding for the corresponding input sequence.
    """
    repo_path = Path(__file__).resolve().parents[2]

    weights_path = repo_path / "weights.csv"
    weights = np.loadtxt(weights_path, delimiter=",")

    seq_lengths = np.array([len(seq) for seq in sequences])
    embeddings = seq_lengths[:, None] * weights

    return embeddings


if __name__ == "__main__":
    vs = embed(sequences=["ABC", "AAAA"])
    print(vs)
