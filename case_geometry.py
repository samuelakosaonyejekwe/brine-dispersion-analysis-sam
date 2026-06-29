"""
Synthetic but realistic bathymetry / model geometry for the Bahia Azul site.

The geometry is a recognisable embayment so that the model-output maps read as a
real coastal site assessment:

    * a curved western coastline forming Bahia Azul (the bay),
    * a southern rocky headland - "Punta Faro" - whose lee sets up a tidal eddy,
    * a small offshore island - "Isla Verde" - north of the outfall,
    * the seabed deepening offshore (eastward).

The brine outfall sits ~850 m off the bay shore in ~18 m of water.
"""

import numpy as np
from ubds.hydro import Grid

# Named geographic features (domain-fraction coordinates) used for labels.
# Positions are chosen to sit clear of the feature they name.
FEATURES = dict(
    bay_label=("Bahia Azul", 0.045, 0.50),
    headland_label=("Punta Faro", 0.085, 0.12),
    island_label=("Isla Verde", 0.30, 0.85),
)


def _coastline_x(y, Lx, Ly):
    """Easting of the shoreline as a function of northing (a curved bay)."""
    yr = y / Ly
    base = 0.10 * Lx
    bay = 0.07 * Lx * np.sin(np.pi * np.clip((yr - 0.15) / 0.7, 0, 1))  # bay indent
    return base + bay


def build_bathymetry(Lx, Ly, dx, dy, outfall_xy=None, max_depth=30.0,
                     shore_depth=0.5, headland=True):
    """
    Returns (Grid, X, Y) with a credible depth field (m, positive down) and a
    realistic embayment coastline.
    """
    nx = int(round(Lx / dx)); ny = int(round(Ly / dy))
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="xy")

    xcoast = _coastline_x(Y, Lx, Ly)
    # offshore distance from the local shoreline
    off = (X - xcoast) / (Lx - 0.1 * Lx)
    off = np.clip(off, -0.05, 1.0)
    depth = shore_depth + (max_depth - shore_depth) * np.sqrt(np.clip(off, 0, 1))
    depth += 2.0 * (Y / Ly - 0.5)                      # gentle alongshore trend

    # land wedge behind the coastline
    depth = np.where(X < xcoast, -2.0 - (xcoast - X) / Lx * 20, depth)

    if headland:
        # southern headland (a peninsula protruding offshore)
        hx, hy = 0.20 * Lx, 0.16 * Ly
        r = np.hypot((X - hx) / (0.16 * Lx), (Y - hy) / (0.10 * Ly))
        depth -= (max_depth + 6) * 0.9 * np.exp(-r ** 2)
        # offshore island north of the outfall
        ix, iy = 0.34 * Lx, 0.74 * Ly
        ri = np.hypot((X - ix) / (0.045 * Lx), (Y - iy) / (0.05 * Ly))
        depth -= (max_depth + 8) * np.exp(-ri ** 2)
        # a shallow bank that promotes the eddy
        bx, by = 0.40 * Lx, 0.62 * Ly
        rb = np.hypot((X - bx) / (0.12 * Lx), (Y - by) / (0.07 * Ly))
        depth -= max_depth * 0.45 * np.exp(-rb ** 2)

    depth = np.clip(depth, -12.0, max_depth + 6)
    # modelled depth floored at h_min handled in the hydro model
    grid = Grid(nx=nx, ny=ny, dx=dx, dy=dy, h=np.maximum(depth, 0.3))
    grid.h_raw = depth                      # signed (negative on land) for plots
    return grid, X, Y


def outfall_position(Lx, Ly, offshore_m=850.0, y_frac=0.52):
    """Diffuser position ~offshore_m off the bay shore at the given northing."""
    y = y_frac * Ly
    x = _coastline_x(np.array([y]), Lx, Ly)[0] + offshore_m
    return (float(x), float(y))


def nearest_cell(grid, xy):
    i = int(np.clip(round(xy[0] / grid.dx - 0.5), 0, grid.nx - 1))
    j = int(np.clip(round(xy[1] / grid.dy - 0.5), 0, grid.ny - 1))
    return j, i
