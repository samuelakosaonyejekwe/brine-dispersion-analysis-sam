"""
UBDS metrics engine
===================

Post-processing of solver output into the engineering and regulatory metrics
required for an impact assessment:

* dilution and salinity at seabed impact
* mixing-zone radius (distance to a salinity threshold, e.g. 36 psu)
* areas enclosed by salinity-increment contours (0.5 / 1.0 / 1.5 psu)
* diffusion distances parallel and transverse to the dominant current
* compliance check of trace impurities against Environmental Quality
  Standards (EQS) given the achieved dilution
"""

import numpy as np


def cell_area_grid(dx, dy):
    return dx * dy


def area_above(field2d, dx, dy, threshold):
    """Area (m2) where field2d >= threshold."""
    return np.count_nonzero(field2d >= threshold) * dx * dy


def increment_areas(salinity_env, s_bg, dx, dy, increments=(0.5, 1.0, 1.5)):
    """Areas (km2) enclosed by salinity-increment contours above background."""
    out = {}
    for inc in increments:
        out[inc] = area_above(salinity_env, dx, dy, s_bg + inc) / 1e6
    return out


def diffusion_distance(salinity_env, xc, yc, source_xy, s_bg, dx, dy,
                       increment=0.5):
    """
    Maximum reach of the (s_bg+increment) contour measured *parallel* and
    *transverse* to the principal axis of the affected area, relative to the
    source location.  Returns (parallel_m, transverse_m).
    """
    mask = salinity_env >= (s_bg + increment)
    if not mask.any():
        return 0.0, 0.0
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    px = X[mask] - source_xy[0]
    py = Y[mask] - source_xy[1]
    # principal axis via covariance of the affected points
    pts = np.column_stack([px, py])
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]
    minor = evecs[:, np.argmin(evals)]
    proj_major = pts @ major
    proj_minor = pts @ minor
    parallel = proj_major.max() - proj_major.min()
    transverse = proj_minor.max() - proj_minor.min()
    return float(parallel), float(transverse)


def mixing_zone_radius(salinity_env, xc, yc, source_xy, threshold=36.0):
    """Max distance from the source at which salinity >= threshold (m)."""
    mask = salinity_env >= threshold
    if not mask.any():
        return 0.0
    X, Y = np.meshgrid(xc, yc, indexing="xy")
    r = np.hypot(X[mask] - source_xy[0], Y[mask] - source_xy[1])
    return float(r.max())


def eqs_compliance(brine_conc_ugL, dilution, eqs_ugL, background_ugL=0.0):
    """
    Predicted receiving-water concentration after dilution and compliance
    margin against the EQS.

    Returns dict: predicted, eqs, margin (= eqs/predicted), compliant (bool).
    """
    predicted = background_ugL + (brine_conc_ugL - background_ugL) / dilution
    margin = eqs_ugL / predicted if predicted > 0 else np.inf
    return {"predicted_ugL": predicted, "eqs_ugL": eqs_ugL,
            "margin": margin, "compliant": predicted <= eqs_ugL}


def rmse(model, obs):
    model, obs = np.asarray(model), np.asarray(obs)
    return float(np.sqrt(np.nanmean((model - obs) ** 2)))


def skill_willmott(model, obs):
    """Willmott (1981) index of agreement, 0..1 (1 = perfect)."""
    model, obs = np.asarray(model, float), np.asarray(obs, float)
    obar = np.nanmean(obs)
    num = np.nansum((model - obs) ** 2)
    den = np.nansum((np.abs(model - obar) + np.abs(obs - obar)) ** 2)
    return float(1 - num / den) if den > 0 else np.nan


def nash_sutcliffe(model, obs):
    model, obs = np.asarray(model, float), np.asarray(obs, float)
    den = np.nansum((obs - np.nanmean(obs)) ** 2)
    return float(1 - np.nansum((model - obs) ** 2) / den) if den > 0 else np.nan
