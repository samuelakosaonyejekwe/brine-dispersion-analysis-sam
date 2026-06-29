"""
Shared helpers for the case-study runner: velocity-field interpolation,
artifact registry (figures / tables / CSVs with captions) and CSV writing.
"""

import os, json
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

FIG_DIR = "outputs/figures"
CSV_DIR = "outputs/csv"
MANIFEST = "outputs/manifest.json"

_ARTIFACTS = []


def reset_artifacts():
    _ARTIFACTS.clear()


def register(kind, path, caption, section, **extra):
    """kind in {figure, table, csv}. Records an output artifact for the report."""
    rec = {"kind": kind, "path": path, "caption": caption,
           "section": section, **extra}
    _ARTIFACTS.append(rec)
    return path


def save_manifest():
    with open(MANIFEST, "w") as f:
        json.dump(_ARTIFACTS, f, indent=2)
    return MANIFEST


def write_csv(df: pd.DataFrame, name, caption, section, index=False):
    path = os.path.join(CSV_DIR, name)
    df.to_csv(path, index=index)
    register("csv", path, caption, section)
    return path


def interp_field(field2d, factor):
    """Bilinearly interpolate a (ny, nx) field by an integer/real factor."""
    return zoom(field2d, factor, order=1, mode="nearest")


def interp_series(series3d, factor):
    """Interpolate a (nt, ny, nx) stack in the horizontal only."""
    nt = series3d.shape[0]
    out = None
    for k in range(nt):
        f = interp_field(series3d[k], factor)
        if out is None:
            out = np.empty((nt,) + f.shape, dtype=float)
        out[k] = f
    return out
