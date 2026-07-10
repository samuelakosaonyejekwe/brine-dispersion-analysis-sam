"""
UBDS - Universal Brine Dispersion Solver
========================================

A unified, open-source, physics-based solver for the near-, medium- and
far-field dispersion of brine (and any other buoyant or negatively-buoyant
effluent) discharged to coastal and marine waters.

UBDS bridges the long-standing gap between dedicated *near-field* integral
jet/plume models and proprietary *far-field* hydrodynamic transport codes by
combining both regimes in a single, open, dynamically-coupled, mass-conserving
framework.

Modules
-------
eos        : equation of state (ambient seawater + hypersaline brine extension)
nearfield  : sign-agnostic Eulerian flux-integral jet/plume model (UBDS-NF)
diffuser   : multi-port diffuser geometry and plume-merging logic
hydro      : 2-D depth-averaged shallow-water hydrodynamic engine (UBDS-HD)
transport  : quasi-3D sigma-layer advection-dispersion engine with a dense
             seabed gravity-current closure (UBDS-FF)
metrics    : dilution, mixing-zone, impact-area and regulatory-compliance metrics
viz        : publication plotting helpers (colour-blind safe, never pure black)

Author : Akosa Samuel Onyejekwe (Independent Researcher)
"""

from . import eos, nearfield, diffuser, hydro, transport, metrics, viz

__all__ = ["eos", "nearfield", "diffuser", "hydro", "transport", "metrics", "viz"]
__version__ = "1.0.0"
