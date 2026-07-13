"""
Synthetic but realistic bathymetry / model geometry for the Bahia Azul site.

The geometry is a recognisable embayment so that the model-output maps read as a
real coastal site assessment:

    * a curved western coastline forming Bahia Azul (the bay),
    * a southern rocky headland - "Punta Faro" - whose lee sets up a tidal eddy,
    * a small offshore island - "Isla Verde" - north of the outfall,
    * the seabed deepening offshore (eastward).

The brine outfall sits ~850 m off the bay shore in ~18 m of water.

Coordinates
-----------
Every feature below is defined in ABSOLUTE metres on a single site grid whose
origin is the south-west corner of the medium-field domain.  A model domain is
then a *window* onto that one site: ``build_bathymetry`` takes the window size
(Lx, Ly) and its origin (x0, y0) in site coordinates, and returns X/Y already in
site coordinates.  So the medium-field and regional domains describe the same
coastline, the same headland and island, and the same diffuser - the regional
one simply sees more of it, more coarsely.

This matters.  An earlier version scaled every feature by a *fraction of the
domain* (coast at 0.10*Lx, island at (0.34*Lx, 0.74*Ly), depth reaching 30 m at
0.9*Lx).  That silently made the domain size the site: enlarging the box moved
the coastline offshore and re-scaled the bathymetry, so the 7.2 km "regional"
domain placed the diffuser 2385 m from shore instead of 850 m.  The two domains
were not two views of one site but two different sites, which made the regional
model useless as a check on the medium-field one and made the nested-domain box
drawn on the regional map a fiction.  Absolute coordinates fix that at the root.
"""

import numpy as np
from ubds.hydro import Grid

# --- the site, in absolute metres -------------------------------------------
COAST_X = 250.0             # shoreline easting on the open coast
BAY_AMP = 175.0             # how far the bay indents seaward of that
BAY_Y0, BAY_Y1 = 375.0, 2125.0      # northing range over which the bay opens

DEPTH_SCALE_M = 2250.0      # offshore distance over which depth -> max_depth
ALONGSHORE_TREND_M = 2500.0  # length scale of the gentle alongshore depth trend
ALONGSHORE_MID_M = 1250.0    # northing about which that trend is centred

HEADLAND = dict(x=500.0, y=400.0, rx=400.0, ry=250.0)     # Punta Faro
ISLAND = dict(x=850.0, y=1850.0, rx=112.5, ry=125.0)      # Isla Verde
BANK = dict(x=1000.0, y=1550.0, rx=300.0, ry=175.0)       # shallow bank (eddy)

# Named geographic features (absolute site coordinates) used for map labels.
# Positions are chosen to sit clear of the feature they name.
FEATURES = dict(
    bay_label=("Bahia Azul", 112.5, 1250.0),
    headland_label=("Punta Faro", 212.5, 300.0),
    island_label=("Isla Verde", 750.0, 2125.0),
)


def coastline_x(y):
    """Easting of the shoreline at northing ``y`` (a curved bay), site coords.

    Use this wherever a distance "from shore" is needed.  Inside the bay the
    shoreline bulges up to BAY_AMP seaward of the open-coast easting, so taking
    COAST_X as the shoreline at any northing understates the offshore distance by
    up to 175 m.
    """
    t = np.clip((np.asarray(y, dtype=float) - BAY_Y0) / (BAY_Y1 - BAY_Y0), 0, 1)
    return COAST_X + BAY_AMP * np.sin(np.pi * t)


_coastline_x = coastline_x       # retained: internal callers use the old name


def build_bathymetry(Lx, Ly, dx, dy, outfall_xy=None, max_depth=30.0,
                     shore_depth=0.5, headland=True, x0=0.0, y0=0.0):
    """
    Returns (Grid, X, Y) with a credible depth field (m, positive down) and a
    realistic embayment coastline.

    (Lx, Ly) is the window size and (x0, y0) its south-west corner, both in site
    coordinates.  X and Y come back in site coordinates too, so results from
    different domains are directly comparable and overlay correctly.
    """
    nx = int(round(Lx / dx)); ny = int(round(Ly / dy))
    xc = x0 + (np.arange(nx) + 0.5) * dx
    yc = y0 + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="xy")

    xcoast = _coastline_x(Y)
    # offshore distance from the local shoreline, as a fraction of the depth scale
    off = (X - xcoast) / DEPTH_SCALE_M
    off = np.clip(off, -0.05, 1.0)
    depth = shore_depth + (max_depth - shore_depth) * np.sqrt(np.clip(off, 0, 1))
    depth += 2.0 * (Y - ALONGSHORE_MID_M) / ALONGSHORE_TREND_M   # alongshore trend

    # land wedge behind the coastline
    depth = np.where(X < xcoast, -2.0 - (xcoast - X) / ALONGSHORE_TREND_M * 20, depth)

    if headland:
        # southern headland (a peninsula protruding offshore)
        r = np.hypot((X - HEADLAND["x"]) / HEADLAND["rx"],
                     (Y - HEADLAND["y"]) / HEADLAND["ry"])
        depth -= (max_depth + 6) * 0.9 * np.exp(-r ** 2)
        # offshore island north of the outfall
        ri = np.hypot((X - ISLAND["x"]) / ISLAND["rx"],
                      (Y - ISLAND["y"]) / ISLAND["ry"])
        depth -= (max_depth + 8) * np.exp(-ri ** 2)
        # a shallow bank that promotes the eddy
        rb = np.hypot((X - BANK["x"]) / BANK["rx"], (Y - BANK["y"]) / BANK["ry"])
        depth -= max_depth * 0.45 * np.exp(-rb ** 2)

    depth = np.clip(depth, -12.0, max_depth + 6)
    # modelled depth floored at h_min handled in the hydro model
    grid = Grid(nx=nx, ny=ny, dx=dx, dy=dy, h=np.maximum(depth, 0.3))
    grid.h_raw = depth                      # signed (negative on land) for plots
    grid.x0 = float(x0); grid.y0 = float(y0)
    return grid, X, Y


def outfall_position(offshore_m=850.0, y=1300.0):
    """Diffuser position ``offshore_m`` off the bay shore at the given northing."""
    return (float(_coastline_x(np.array([y]))[0] + offshore_m), float(y))


def nearest_cell(grid, xy):
    """Cell index of a point given in site coordinates."""
    gx0 = getattr(grid, "x0", 0.0); gy0 = getattr(grid, "y0", 0.0)
    i = int(np.clip(round((xy[0] - gx0) / grid.dx - 0.5), 0, grid.nx - 1))
    j = int(np.clip(round((xy[1] - gy0) / grid.dy - 0.5), 0, grid.ny - 1))
    return j, i
