"""
Calibrate the UBDS near-field entrainment coefficients against published
initial-dilution data for dense vertical jets discharged through 6-inch ports.

Reference dilution targets (independent dense-brine outfall study, vertical
6-inch ports, 27 m water depth, brine 260 psu into 34.2 psu ambient) - used here
purely as a calibration/validation dataset for the open UBDS kernel:

    flow (m3/hr, single port) | tidal vel (m/s) | dilution at seabed contact
    ------------------------- | --------------- | --------------------------
    250                       | 0.10 (slack)    | 13.8   (50.5 psu)
    250                       | 0.30 (neap)     | 22.4   (44.3 psu)
    250                       | 0.60 (spring)   | 37.0   (40.3 psu)
    500                       | 0.10 (slack)    | 25.1   (43.2 psu)
    500                       | 0.30 (neap)     | 41.0   (39.7 psu)
    500                       | 0.60 (spring)   | 66.0   (37.6 psu)
"""
import numpy as np
from scipy.optimize import minimize
from ubds import nearfield as nf

TARGETS = [
    (250, 0.10, 13.8), (250, 0.30, 22.4), (250, 0.60, 37.0),
    (500, 0.10, 25.1), (500, 0.30, 41.0), (500, 0.60, 66.0),
]


def run(flow, vel):
    amb = nf.Ambient(salinity=34.2, temperature=8.4, depth=27.0, current=vel)
    d = nf.Discharge(flow_m3hr=flow, diameter_m=0.1524, salinity=260,
                     temperature=10.4, z0=0.75, theta_deg=90)
    return nf.simulate(d, amb, n_ports=1).impact_dilution


def cost(p):
    nf.ALPHA_JET, nf.ALPHA_FORCED, nf.DESCENT_GAIN = p
    nf.ALPHA_PLUME = nf.ALPHA_JET * 1.46
    err = 0.0
    for flow, vel, tgt in TARGETS:
        m = run(flow, vel)
        err += ((m - tgt) / tgt) ** 2          # relative least squares
    return err


if __name__ == "__main__":
    x0 = np.array([0.057, 0.90, 2.6])
    res = minimize(cost, x0, method="Nelder-Mead",
                   bounds=[(0.04, 0.16), (0.3, 3.0), (1.0, 6.0)],
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
    nf.ALPHA_JET, nf.ALPHA_FORCED, nf.DESCENT_GAIN = res.x
    nf.ALPHA_PLUME = nf.ALPHA_JET * 1.46
    print("Calibrated coefficients:")
    print(f"  ALPHA_JET    = {nf.ALPHA_JET:.4f}")
    print(f"  ALPHA_PLUME  = {nf.ALPHA_PLUME:.4f}")
    print(f"  ALPHA_FORCED = {nf.ALPHA_FORCED:.4f}")
    print(f"  DESCENT_GAIN = {nf.DESCENT_GAIN:.4f}")
    print(f"  final cost   = {res.fun:.5f}")
    print("\n flow  vel   model   target   err%")
    for flow, vel, tgt in TARGETS:
        m = run(flow, vel)
        print(f"{flow:5d} {vel:4.2f}  x{m:6.2f}  x{tgt:6.2f}  {100*(m-tgt)/tgt:+6.1f}")
