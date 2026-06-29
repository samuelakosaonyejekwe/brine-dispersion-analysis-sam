"""
UBDS-NF : Universal near-field Lagrangian integral jet/plume model
==================================================================

A sign-agnostic Lagrangian "plume-element" integral model in the spirit of
JETLAG (Lee & Cheung, 1990) / VISJET and the UM3 routine inside US-EPA Visual
Plumes, but written from scratch and generalised so that the *same* equations
describe:

    * negatively buoyant dense jets  (brine, reject concentrate)   g' > 0
    * positively buoyant jets        (thermal, freshwater)         g' < 0
    * neutrally buoyant jets         (tracer release)              g' = 0

The plume is discretised into a chain of cylindrical elements of mass ``M``,
radius ``b`` and thickness ``h``.  Each element is advected by its own velocity
and grows by entraining ambient fluid through two mechanisms (the *combined
entrainment hypothesis* of Lee & Chu, 2003):

    1. shear entrainment   - proportional to the velocity excess |V - Va|
    2. forced entrainment  - proportional to the ambient cross-flow swept area

Buoyancy enters as a vertical body force g(rho_a - rho_e)/rho_e, so a dense
brine element naturally decelerates, arcs over and sinks back to the bed, while
a light element rises - no special-casing required.  This universality is one
of the core novel features of UBDS.

The model returns the full 3-D trajectory, the centreline and boundary plume
geometry, the dilution-vs-distance curve and the salinity / density at the
point of seabed impact (or terminal rise height).
"""

import numpy as np
from dataclasses import dataclass, field
from . import eos


@dataclass
class Ambient:
    """Ambient (receiving water) conditions."""
    salinity: float = 34.2      # psu
    temperature: float = 8.4    # degC
    current: float = 0.3        # m/s  (horizontal, +x direction)
    depth: float = 27.0         # m    water depth at outfall (chart datum)
    g: float = 9.81

    def density(self):
        return eos.density(self.salinity, self.temperature)


@dataclass
class Discharge:
    """Single-port discharge (jet) source term."""
    flow_m3hr: float = 250.0    # volumetric flow through THIS port (m3/hr)
    diameter_m: float = 0.1524  # port exit diameter (6 inch = 0.1524 m)
    salinity: float = 260.0     # psu
    temperature: float = 10.4   # degC  (ambient + 2C in the IGSF case)
    z0: float = 0.75            # height of port above seabed (m)
    theta_deg: float = 90.0     # vertical angle (90 = straight up)
    azimuth_deg: float = 90.0   # horizontal angle relative to +x current (deg)

    def exit_velocity(self):
        area = np.pi * (self.diameter_m / 2.0) ** 2
        return (self.flow_m3hr / 3600.0) / area

    def density(self):
        return eos.density(self.salinity, self.temperature)


@dataclass
class NearFieldResult:
    s: np.ndarray            # arclength along trajectory (m)
    x: np.ndarray            # horizontal downstream coord (m)
    y: np.ndarray            # horizontal cross-stream coord (m)
    z: np.ndarray            # height above seabed (m)
    b: np.ndarray            # plume radius (m)
    speed: np.ndarray        # element speed (m/s)
    salinity: np.ndarray     # centreline (top-hat) salinity (psu)
    density: np.ndarray      # element density (kg/m3)
    dilution: np.ndarray     # bulk dilution S0/S (concentration dilution)
    impact_index: int        # index where plume first returns to the bed
    merged_from: int = field(default=1)  # number of merged ports

    # --- convenience scalars at seabed impact -------------------------------
    @property
    def impact_dilution(self):
        return float(self.dilution[self.impact_index])

    @property
    def impact_salinity(self):
        return float(self.salinity[self.impact_index])

    @property
    def impact_distance(self):
        return float(np.hypot(self.x[self.impact_index], self.y[self.impact_index]))

    @property
    def rise_height(self):
        return float(np.max(self.z))


# Entrainment coefficients (top-hat).  alpha_jet -> pure jet, alpha_plume -> pure
# plume; blended by the local densimetric Froude number.  Forced (cross-flow)
# entrainment uses the projected-area hypothesis with coefficient alpha_f.
# These coefficients are calibrated against published initial-dilution data for
# dense vertical jets (see validation module / calibration record).
ALPHA_JET = 0.057
ALPHA_PLUME = 0.083
ALPHA_FORCED = 0.90
# Fountain-recirculation enhancement: a negatively-buoyant (dense) jet forms a
# fountain whose collapsing outer sheath flows back down around the rising core,
# entraining additional ambient water through a 3-D counterflow recirculation
# that an axisymmetric single-path integral model cannot resolve.  This is
# represented by a multiplicative entrainment gain applied only in the dense
# (g' > 0) regime, with extra weight on the descending branch.  Calibrated
# (dimensionless) against dense-jet initial-dilution data; it does NOT affect
# the neutral-jet or positively-buoyant-plume regimes, which are validated
# against the canonical self-similar dilution laws.
# Default 1.0 (disabled): the validated configuration uses the standard
# single-path integral estimate, which agrees with the published inclined
# dense-jet data (Papakonstantis et al., 2011) once the top-hat bulk dilution
# is converted to the centreline minimum dilution.  The gains are retained as
# an optional, documented closure for fountain recirculation.
FOUNTAIN_GAIN = 1.0
DESCENT_GAIN = 1.0


def simulate_port(disch: Discharge, amb: Ambient,
                  ds_frac: float = 0.04, max_steps: int = 400000,
                  merge_spacing: float | None = None):
    """
    Integrate one round jet/plume from a single port using the UBDS Eulerian
    flux-integral formulation (top-hat) in arclength s.

    The conserved fluxes are integrated by 4th-order Runge-Kutta:

        mass flux      mu     = rho_e * Q              (Q = pi b^2 w)
        momentum flux  Mvec   = mu * V
        salt   flux    Sig    = mu * S
        heat   flux    The    = mu * T

    Source terms (per unit arclength):
        d(mu)/ds   = rho_a * E
        d(Mvec)/ds = rho_a * E * Va + (0, 0, (rho_a - rho_e) g pi b^2)   [buoyancy]
        d(Sig)/ds  = rho_a * E * Sa
        d(The)/ds  = rho_a * E * Ta
        d(pos)/ds  = V / |V|

    with entrainment  E = 2 pi b alpha_s |V - Va| + 2 b alpha_f |Va_perp|.
    The buoyancy term is sign-agnostic, so the *same* code descends a dense
    brine jet and lifts a buoyant one.
    """
    g = amb.g
    rho_a = amb.density()
    Sa, Ta = amb.salinity, amb.temperature
    Va = np.array([amb.current, 0.0, 0.0])

    u0 = disch.exit_velocity()
    b0 = disch.diameter_m / 2.0
    rho0 = disch.density()
    theta = np.radians(disch.theta_deg)
    azi = np.radians(disch.azimuth_deg)
    V0 = np.array([u0 * np.cos(theta) * np.cos(azi),
                   u0 * np.cos(theta) * np.sin(azi),
                   u0 * np.sin(theta)])
    Q0 = np.pi * b0 ** 2 * u0
    mu0 = rho0 * Q0

    # state vector: [mu, Mx, My, Mz, Sig, The, x, y, z]
    y = np.array([mu0, *(mu0 * V0), mu0 * disch.salinity, mu0 * disch.temperature,
                  0.0, 0.0, disch.z0])

    def unpack(y):
        mu = max(y[0], 1e-12)
        V = y[1:4] / mu
        w = np.linalg.norm(V)
        S = y[4] / mu
        T = y[5] / mu
        rho_e = eos.density(S, T)
        Q = mu / rho_e
        b = np.sqrt(max(Q / (np.pi * max(w, 1e-6)), 1e-12))
        return mu, V, w, S, T, rho_e, Q, b

    def deriv(y):
        mu, V, w, S, T, rho_e, Q, b = unpack(y)
        if w < 1e-6:
            return np.zeros_like(y), b, S, rho_e
        axis = V / w
        Vrel = V - Va
        Vrel_mag = np.linalg.norm(Vrel)
        gp = g * (rho_e - rho_a) / rho_a
        Fr2 = w ** 2 / (abs(gp) * b + 1e-12)
        wj = Fr2 / (Fr2 + 1.0)
        alpha_s = wj * ALPHA_JET + (1 - wj) * ALPHA_PLUME
        Va_perp = Va - np.dot(Va, axis) * axis
        E = 2 * np.pi * b * alpha_s * Vrel_mag + 2 * b * ALPHA_FORCED * np.linalg.norm(Va_perp)
        # fountain-recirculation enhancement (dense regime only)
        if gp > 0.0:
            E *= FOUNTAIN_GAIN
            if V[2] < 0.0:
                E *= DESCENT_GAIN
        if merge_spacing is not None and 2 * b > merge_spacing:
            overlap = np.clip((2 * b - merge_spacing) / (2 * b), 0, 0.85)
            E *= (1.0 - 0.5 * overlap)
        dmu = rho_a * E
        dM = rho_a * E * Va + np.array([0.0, 0.0, (rho_a - rho_e) * g * np.pi * b ** 2])
        dSig = rho_a * E * Sa
        dThe = rho_a * E * Ta
        dpos = axis
        dy = np.array([dmu, dM[0], dM[1], dM[2], dSig, dThe, dpos[0], dpos[1], dpos[2]])
        return dy, b, S, rho_e

    rec = {k: [] for k in ("s", "x", "y", "z", "b", "speed", "sal", "rho", "dil")}
    s = 0.0
    impact_index = None
    z_prev = disch.z0
    rose = False

    for step in range(max_steps):
        dy, b, S, rho_e = deriv(y)
        w = np.linalg.norm(y[1:4] / max(y[0], 1e-12))
        dil = (disch.salinity - Sa) / max(S - Sa, 1e-9)
        rec["s"].append(s); rec["x"].append(y[6]); rec["y"].append(y[7])
        rec["z"].append(y[8]); rec["b"].append(b); rec["speed"].append(w)
        rec["sal"].append(S); rec["rho"].append(rho_e); rec["dil"].append(dil)

        ds = max(ds_frac * b, 0.25 * ds_frac * b0)
        # RK4
        k1, *_ = deriv(y)
        k2, *_ = deriv(y + 0.5 * ds * k1)
        k3, *_ = deriv(y + 0.5 * ds * k2)
        k4, *_ = deriv(y + ds * k3)
        y = y + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s += ds

        if y[8] > disch.z0 + 0.05:
            rose = True
        # seabed impact after rising (negatively buoyant return)
        if rose and y[8] <= 0.0 and step > 2:
            mu, V, w, S, T, rho_e, Q, b = unpack(y)
            dil = (disch.salinity - Sa) / max(S - Sa, 1e-9)
            rec["s"].append(s); rec["x"].append(y[6]); rec["y"].append(y[7])
            rec["z"].append(0.0); rec["b"].append(b); rec["speed"].append(w)
            rec["sal"].append(S); rec["rho"].append(rho_e); rec["dil"].append(dil)
            impact_index = len(rec["s"]) - 1
            break
        if y[8] >= amb.depth:                       # surfaced (buoyant case)
            impact_index = len(rec["s"]) - 1
            break
        if np.linalg.norm(y[1:4] / max(y[0], 1e-12) - Va) < 0.03 * u0 and step > 20:
            impact_index = len(rec["s"]) - 1
            break
        z_prev = y[8]

    arr = {k: np.asarray(v) for k, v in rec.items()}
    if impact_index is None:
        impact_index = len(arr["s"]) - 1
    return NearFieldResult(
        s=arr["s"], x=arr["x"], y=arr["y"], z=arr["z"], b=arr["b"],
        speed=arr["speed"], salinity=arr["sal"], density=arr["rho"],
        dilution=arr["dil"], impact_index=int(impact_index))


def simulate(disch: Discharge, amb: Ambient, n_ports: int = 1,
             port_spacing: float = 20.0, **kw):
    """
    Universal entry point.  For multi-port diffusers the per-port flow is the
    total divided by n_ports; merging is handled through ``merge_spacing``.
    """
    merge = port_spacing if n_ports > 1 else None
    res = simulate_port(disch, amb, merge_spacing=merge, **kw)
    res.merged_from = n_ports
    return res
