from __future__ import annotations

import numpy as np


def make_rgb(data, vmax=0.10):
    return np.clip(np.stack([data[3], data[2], data[1]], axis=-1) / vmax, 0, 1)
