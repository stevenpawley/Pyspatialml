import numpy as np


def extract_by_chunk(arr, w, idx, pixel_idx):
    d = idx.copy()
    pixel_idx = pixel_idx.copy()

    # subtract chunk offset from row, col positions
    d[:, 0] = d[:, 0] - w.row_off
    d[:, 1] = d[:, 1] - w.col_off

    # remove negative row, col positions
    pos = (d >= 0).all(axis=1)
    d = d[pos, :]
    pixel_idx = pixel_idx[pos]

    # remove row, col > shape
    within_range = (d[:, 0] < arr.shape[1]) & (d[:, 1] < arr.shape[2])
    d = d[within_range, :]
    pixel_idx = pixel_idx[within_range]

    extracted_data = arr[:, d[:, 0], d[:, 1]]

    return (extracted_data, pixel_idx)
