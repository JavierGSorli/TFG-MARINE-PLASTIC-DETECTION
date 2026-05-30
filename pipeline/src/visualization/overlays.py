from __future__ import annotations

import numpy as np


def overlay_mask(rgb, mask, color=(1, 0, 0), alpha=0.65):
    out = rgb.copy()
    active = mask.astype(bool)
    for channel, value in enumerate(color):
        out[:, :, channel] = np.where(active, alpha * value + (1 - alpha) * rgb[:, :, channel], rgb[:, :, channel])
    return out
