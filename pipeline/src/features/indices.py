from __future__ import annotations


def compute_fdi(data):
    b06, b08, b11 = data[5], data[7], data[9]
    return b08 - (b06 + (b11 - b06) * ((832.9 - 664.6) / (1613.7 - 664.6)) * 10)
