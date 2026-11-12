"""Cosine similarity, with one policy for every caller.

- **Range:** the raw cosine, in ``[-1, 1]``. Results are clipped to that range,
  only to absorb floating-point rounding (e.g. ``1.0000000002`` for identical
  vectors). Metrics defined on ``[0, 1]`` clip negatives themselves.
- **Zero vectors:** a zero vector has no direction, so its similarity to
  anything, itself included, is ``0.0``. It is never NaN, and no epsilon is
  added to the norms, so non-zero vectors get the exact cosine.
"""

from __future__ import annotations

import typing as t

import numpy as np

ArrayLike = t.Union[np.ndarray, t.Sequence[float], t.Sequence[t.Sequence[float]]]


def cosine_matrix(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Cosine similarity of every row of ``a`` with every row of ``b``.

    ``a`` is ``(n, d)`` and ``b`` is ``(m, d)`` (a 1-D input is one row); the
    result is ``(n, m)``.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    # Zero rows stay zero after "normalising", so their dot products are 0.
    unit_a = np.divide(a, norm_a, out=np.zeros_like(a), where=norm_a > 0)
    unit_b = np.divide(b, norm_b, out=np.zeros_like(b), where=norm_b > 0)
    return np.clip(unit_a @ unit_b.T, -1.0, 1.0)


def cosine(a: ArrayLike, b: ArrayLike) -> float:
    """Cosine similarity of two vectors (see the module docstring)."""
    return float(cosine_matrix(np.ravel(a), np.ravel(b))[0, 0])
