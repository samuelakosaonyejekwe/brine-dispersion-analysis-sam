"""
UBDS equation of state
======================

Density of sea water as a function of practical salinity S (psu) and
temperature T (degC).

Two regimes are provided and blended so that the solver is *universal* over the
full salinity range encountered in brine work (0 -> ~300 psu):

1. Ambient / oceanic regime (S <= 42 psu):
   UNESCO EOS-80 one-atmosphere International Equation of State (Millero &
   Poisson, 1981) - the community-standard seawater density formulation.

2. Hypersaline / brine regime (S > 42 psu):
   EOS-80 is not formally valid for brine.  UBDS therefore extends it with a
   linear haline-contraction term anchored on the saturated-brine density of
   NaCl-dominated solutions (~1200 kg/m3 at ~260 psu, 10 degC), which matches
   the saturated-brine density reported for solution-mined caverns.

A smooth C1 blend (cosine taper) over 38-46 psu removes any kink at the
junction so that the near-field ODE integration stays well behaved.

References
----------
* Millero, F.J. & Poisson, A. (1981) International one-atmosphere equation of
  state of seawater. Deep-Sea Research 28A, 625-629.
* UNESCO (1981) Tenth report of the joint panel on oceanographic tables and
  standards. UNESCO Tech. Papers in Marine Sci. 36.
"""

import numpy as np

# ---------------------------------------------------------------------------
# UNESCO EOS-80 (one atmosphere) - valid 0..40 psu, -2..40 degC
# ---------------------------------------------------------------------------


def _rho_smow(T):
    """Density of Standard Mean Ocean Water (pure water), kg/m3."""
    a = [999.842594, 6.793952e-2, -9.095290e-3,
         1.001685e-4, -1.120083e-6, 6.536332e-9]
    return (a[0] + a[1] * T + a[2] * T ** 2 + a[3] * T ** 3
            + a[4] * T ** 4 + a[5] * T ** 5)


def _rho_eos80(S, T):
    """UNESCO EOS-80 one-atmosphere seawater density, kg/m3."""
    b = [8.24493e-1, -4.0899e-3, 7.6438e-5, -8.2467e-7, 5.3875e-9]
    c = [-5.72466e-3, 1.0227e-4, -1.6546e-6]
    d = 4.8314e-4
    rho0 = _rho_smow(T)
    B = b[0] + b[1] * T + b[2] * T ** 2 + b[3] * T ** 3 + b[4] * T ** 4
    C = c[0] + c[1] * T + c[2] * T ** 2
    return rho0 + B * S + C * S ** 1.5 + d * S ** 2


# ---------------------------------------------------------------------------
# Hypersaline / brine extension
# ---------------------------------------------------------------------------
# Anchored so that at S = 260 psu, T = 10 degC the density is ~1200 kg/m3,
# the saturated NaCl-brine value used for solution-mined gas-storage caverns.
_BRINE_BETA = 7.78e-4   # effective haline contraction in brine regime (1/psu)
_BRINE_BT = 3.2e-4      # thermal expansion in brine regime (1/degC)


def _rho_brine(S, T):
    """Linear hypersaline extension anchored on EOS-80 at the blend point."""
    S_anchor = 42.0
    rho_anchor = _rho_eos80(S_anchor, T)
    return rho_anchor * (1.0 + _BRINE_BETA * (S - S_anchor)
                         - _BRINE_BT * (T - 10.0)) \
        - rho_anchor * (-_BRINE_BT * (T - 10.0))  # keep T handling in EOS80 part


def density(S, T):
    """
    Universal seawater/brine density (kg/m3).

    Parameters
    ----------
    S : float or ndarray   practical salinity (psu)
    T : float or ndarray   temperature (degC)
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    rho_lo = _rho_eos80(np.clip(S, 0, 42), T)
    rho_hi = _rho_brine(np.maximum(S, 42), T)
    # cosine blend over 38..46 psu
    w = np.clip((S - 38.0) / (46.0 - 38.0), 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * w))
    rho = (1.0 - w) * rho_lo + w * rho_hi
    return rho.item() if rho.ndim == 0 else rho


def buoyancy_g_prime(S, T, S_amb, T_amb, g=9.81):
    """Reduced gravity g' = g (rho - rho_amb)/rho_amb.  >0 = denser than ambient."""
    rho = density(S, T)
    rho_a = density(S_amb, T_amb)
    return g * (rho - rho_a) / rho_a


if __name__ == "__main__":
    # sanity check
    for S, T in [(0, 10), (34.2, 8.4), (35, 15), (53.5, 18), (260, 10)]:
        print(f"S={S:7.1f} psu T={T:5.1f} C  rho={density(S, T):8.2f} kg/m3")
