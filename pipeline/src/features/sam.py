from __future__ import annotations

import numpy as np


def spectral_angle(pixels, reference):
    pixels = np.asarray(pixels)
    reference = np.asarray(reference)
    denom = np.linalg.norm(pixels, axis=1) * np.linalg.norm(reference)
    out = np.full(pixels.shape[0], np.nan, dtype="float32")
    valid = denom > 1e-12
    out[valid] = np.arccos(np.clip((pixels[valid] @ reference) / denom[valid], -1.0, 1.0))
    return out
