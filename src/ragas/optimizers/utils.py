import numpy as np


def hamming_distance(vectors: np.ndarray) -> np.ndarray:
    """
    Calculate the pairwise Hamming distances between vectors.

    Args:
    vectors: A 2-D array (or list of lists) where each row is a vector.

    Returns:
    A symmetric ``(n, n)`` integer matrix whose ``[i, j]`` entry is the
    Hamming distance between vectors ``i`` and ``j``. The diagonal is 0.
    """

    # Validate that all vectors have the same dimension
    length = len(vectors[0])
    if any(len(v) != length for v in vectors):
        raise ValueError("All vectors must have the same dimensions.")

    # Calculate Hamming distances for all pairs
    distances = np.zeros((len(vectors), len(vectors)), dtype=int)
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distance = np.sum(vectors[i] != vectors[j])
            distances[i][j] = distance
            distances[j][i] = distance

    return distances
