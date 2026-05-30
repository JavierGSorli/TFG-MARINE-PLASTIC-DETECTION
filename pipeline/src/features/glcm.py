from __future__ import annotations

import numpy as np


def normalize_gray_uint8(arr, levels=32):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros(arr.shape, dtype="uint8")
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        return np.zeros(arr.shape, dtype="uint8")
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * (levels - 1)).astype("uint8")
