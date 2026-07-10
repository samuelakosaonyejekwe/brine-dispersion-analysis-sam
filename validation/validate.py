"""
UBDS VALIDATION SUITE
=====================

Independent validation of the UBDS solver kernels against credible, published
benchmark data and analytical self-similar laws.  Each test records its data
source (see validation/SOURCES.md).  Outputs:

    outputs/figures/V_*.png
    outputs/csv/V_*.csv
    validation/validation_summary.json

Author: Akosa Samuel Onyejekwe (Independent Researcher)
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ubds import nearfield as nf, eos, viz
import run_helpers as H

os.makedirs(H.FIG_DIR, exist_ok=True)
os.makedirs(H.CSV_DIR, exist_ok=True)
SUMMARY = {}

# UBDS integrates top-hat (flux-averaged) fluxes, so ``impact_dilution`` is a BULK
# dilution.  Published dense-jet dilutions (Papakonstantis et al. 2011; Roberts et
# al. 1997) are centreline MINIMUM dilutions, measured on the plume axis.  For a
# Gaussian concentration profile of spread lambda*b relative to the velocity
# half-width b, the flux-averaged concentration is lambda^2/(1+lambda^2) of the
# centreline value, so S_min = S_bulk * lambda^2/(1+lambda^2).  With the standard
# lambda = 1.2 this is S_min = S_bulk / 1.694.  Comparing the model's bulk value
# directly against a centreline band would overstate the model's dilution by this
# factor; the conversion is applied wherever the published band is quoted.
LAMBDA_C = 1.2
BULK_TO_MIN = (1.0 + LAMBDA_C ** 2) / LAMBDA_C ** 2   # 1.694


# ---------------------------------------------------------------------------
# 1. Equation of state vs reference seawater/brine densities
# ---------------------------------------------------------------------------
def validate_eos():
    # Reference one-atmosphere seawater densities (UNESCO EOS-80, Millero &
    # Poisson 1981) and saturated NaCl brine (CRC Handbook).
    refs = [  # S, T, reference rho (kg/m3), source
        (0.0, 25.0, 997.05, "Pure water (CRC)"),
        (35.0, 15.0, 1025.97, "EOS-80 (Millero & Poisson 1981)"),
        (35.0, 25.0, 1023.34, "EOS-80"),
        (38.0, 20.0, 1026.95, "EOS-80"),
        (260.0, 20.0, 1200.0, "Saturated NaCl brine (CRC Handbook)"),
    ]
    rows = []
    for S, T, ref, src in refs:
        model = eos.density(S, T)
        rows.append(dict(salinity_psu=S, temp_C=T, ref_density=ref,
                         model_density=round(model, 2),
                         error_pct=round(100 * (model - ref) / ref, 3), source=src))
    d = pd.DataFrame(rows)
    H.write_csv(d, "V_eos_validation.csv",
                "Equation-of-state validation against reference seawater/brine densities.",
                "Validation")
    SUMMARY["eos_max_error_pct"] = float(d.error_pct.abs().max())
    return d


# ---------------------------------------------------------------------------
# 2. Pure momentum jet - canonical self-similar dilution law
#    (Fischer, List, Koh, Imberger & Brooks 1979)
# ---------------------------------------------------------------------------
def validate_jet():
    amb = nf.Ambient(salinity=35.0, temperature=15.0, depth=400.0, current=0.0)
    d = nf.Discharge(flow_m3hr=200, diameter_m=0.15, salinity=35.0001,
                     temperature=15.0, z0=0.0, theta_deg=90)
    r = nf.simulate(d, amb, n_ports=1)
    D = 0.15
    Q = np.pi * r.b ** 2 * r.speed
    S = Q / Q[0]
    zod = r.z / D
    m = (zod > 8) & (zod < 120)
    slope = np.polyfit(zod[m], S[m], 1)[0]
    spread = np.polyfit(r.z[m], r.b[m], 1)[0]
    SUMMARY["jet_dilution_slope_perZD"] = round(float(slope), 3)
    SUMMARY["jet_spread_dbdz"] = round(float(spread), 3)
    # canonical: S ~ 0.25-0.32 (z/D); db/dz ~ 0.10-0.11
    fig, ax = viz.new_ax((7, 5), "Round-jet validation: dilution vs distance",
                         "z/D", "Bulk dilution S/S0")
    ax.plot(zod[m], S[m], color="#2a6f97", lw=2.4, label="UBDS")
    ax.plot(zod[m], 0.28 * zod[m], "--", color="#d1495b", lw=2,
            label="Fischer et al. (1979): 0.28 z/D")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/V_jet.png"),
               "Momentum-jet dilution: UBDS vs the canonical self-similar law (Fischer et al. 1979).",
               "Validation")
    pd.DataFrame({"z_over_D": zod[m], "dilution": S[m]}).to_csv(
        f"{H.CSV_DIR}/V_jet.csv", index=False)
    H.register("csv", f"{H.CSV_DIR}/V_jet.csv", "Round-jet dilution validation data.", "Validation")


# ---------------------------------------------------------------------------
# 3. Pure plume - 5/3 volume-flux power law (Morton, Taylor & Turner 1956)
# ---------------------------------------------------------------------------
def validate_plume():
    amb = nf.Ambient(salinity=35.0, temperature=10.0, depth=500.0, current=0.0)
    d = nf.Discharge(flow_m3hr=25, diameter_m=0.2, salinity=8.0, temperature=10.0,
                     z0=0.0, theta_deg=90)
    r = nf.simulate(d, amb, n_ports=1)
    Q = np.pi * r.b ** 2 * r.speed
    S = Q / Q[0]
    z = r.z
    m = (z > 3) & (z < 0.85 * z.max())
    exp = np.polyfit(np.log(z[m]), np.log(S[m]), 1)[0]
    SUMMARY["plume_dilution_exponent"] = round(float(exp), 3)  # canonical 5/3
    fig, ax = viz.new_ax((7, 5), "Plume validation: volume-flux power law",
                         "Height z (m)", "Dilution S/S0")
    ax.loglog(z[m], S[m], color="#3a9278", lw=2.4, label=f"UBDS (slope={exp:.2f})")
    ax.loglog(z[m], S[m][0] * (z[m] / z[m][0]) ** (5 / 3), "--", color="#d1495b",
              lw=2, label="MTT (1956): slope 5/3")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/V_plume.png"),
               "Pure-plume dilution: UBDS vs the 5/3 power law (Morton, Taylor & Turner 1956).",
               "Validation")


# ---------------------------------------------------------------------------
# 4. Inclined (60 deg) dense jet - geometry & dilution vs Froude number
#    (Papakonstantis, Christodoulou & Papanicolaou 2011, J. Hydraul. Res.)
#    Reported dimensionless constants for a 60 deg negatively buoyant jet:
#       terminal rise of centreline peak  z_t/(D Fr)  ~ 1.6-2.2
#       horizontal distance to return     x_r/(D Fr)  ~ 2.2-3.3
#       dilution at the return point       S_r/Fr      ~ 0.4-0.6
# ---------------------------------------------------------------------------
def validate_dense_jet():
    Sa, Ta = 35.0, 18.0
    S0, T0 = 68.0, 18.3
    g = 9.81
    rho_a = eos.density(Sa, Ta); rho_0 = eos.density(S0, T0)
    gp = g * (rho_0 - rho_a) / rho_a
    D = 0.1524
    rows = []
    # The sweep must span the Froude numbers the CASE STUDY actually runs at, not
    # a convenient band above them.  The saturated-cavern worst case discharges at
    # Fr 9.2-10.7 and the RO concentrate at Fr 24.0-27.9; u0 = 1.75 and 2.15 m/s
    # extend the benchmark down to Fr ~9 so the worst case is covered rather than
    # extrapolated to.
    for u0 in [1.75, 2.15, 2.5, 3.5, 4.5, 5.5, 6.5]:
        flow = u0 * (np.pi * (D / 2) ** 2) * 3600
        Fr = u0 / np.sqrt(gp * D)
        amb = nf.Ambient(salinity=Sa, temperature=Ta, depth=60.0, current=0.0)
        d = nf.Discharge(flow_m3hr=flow, diameter_m=D, salinity=S0, temperature=T0,
                         z0=0.0, theta_deg=60)
        r = nf.simulate(d, amb, n_ports=1)
        z_t = r.rise_height
        x_r = r.impact_distance
        S_r = r.impact_dilution
        rows.append(dict(Fr=round(Fr, 1),
                         zt_over_DFr=round(z_t / (D * Fr), 3),
                         xr_over_DFr=round(x_r / (D * Fr), 3),
                         Sr_bulk_over_Fr=round(S_r / Fr, 3),
                         Sr_min_over_Fr=round(S_r / (BULK_TO_MIN * Fr), 3),
                         termination=r.termination))
    d = pd.DataFrame(rows)
    if (d.termination != "seabed").any():
        raise RuntimeError("dense-jet benchmark: a case failed to reach the seabed: "
                           f"{d[d.termination != 'seabed'].to_dict('records')}")
    H.write_csv(d, "V_dense_jet.csv",
                "Inclined (60 deg) dense-jet dimensionless geometry/dilution vs published ranges. "
                "S_r is reported both as the model's bulk (top-hat) dilution and converted to the "
                "centreline minimum dilution, which is what the published band refers to.",
                "Validation")
    SUMMARY["dense_jet_zt_over_DFr_mean"] = round(float(d.zt_over_DFr.mean()), 3)
    SUMMARY["dense_jet_xr_over_DFr_mean"] = round(float(d.xr_over_DFr.mean()), 3)
    SUMMARY["dense_jet_Sr_bulk_over_Fr_mean"] = round(float(d.Sr_bulk_over_Fr.mean()), 3)
    SUMMARY["dense_jet_Sr_min_over_Fr_mean"] = round(float(d.Sr_min_over_Fr.mean()), 3)
    # Froude span actually benchmarked, and how many cases land inside each band.
    SUMMARY["dense_jet_Fr_min"] = round(float(d.Fr.min()), 1)
    SUMMARY["dense_jet_Fr_max"] = round(float(d.Fr.max()), 1)
    SUMMARY["dense_jet_n_cases"] = int(len(d))
    SUMMARY["dense_jet_zt_in_band"] = int(((d.zt_over_DFr >= 1.6) & (d.zt_over_DFr <= 2.2)).sum())
    SUMMARY["dense_jet_xr_in_band"] = int(((d.xr_over_DFr >= 2.2) & (d.xr_over_DFr <= 3.3)).sum())
    SUMMARY["dense_jet_Sr_min_in_band"] = int(((d.Sr_min_over_Fr >= 0.4) & (d.Sr_min_over_Fr <= 0.6)).sum())
    # plot vs published bands
    fig, ax = viz.new_ax((7.5, 5), "Inclined dense-jet validation (60 deg) vs published data",
                         "Jet densimetric Froude number Fr", "Dimensionless value")
    ax.plot(d.Fr, d.zt_over_DFr, "-o", color="#2a6f97", label="UBDS z_t/(D·Fr)")
    ax.axhspan(1.6, 2.2, color="#2a6f97", alpha=0.15, label="Papakonstantis (2011) z_t band")
    ax.plot(d.Fr, d.xr_over_DFr, "-^", color="#3a9278", label="UBDS x_r/(D·Fr)")
    ax.axhspan(2.2, 3.3, color="#3a9278", alpha=0.10, label="Papakonstantis (2011) x_r band")
    ax.plot(d.Fr, d.Sr_min_over_Fr, "-s", color="#d1495b",
            label=f"UBDS S_r(min)/Fr  [bulk ÷ {BULK_TO_MIN:.2f}]")
    ax.plot(d.Fr, d.Sr_bulk_over_Fr, ":s", color="#d1495b", alpha=0.55,
            label="UBDS S_r(bulk)/Fr  [not directly comparable]")
    ax.axhspan(0.4, 0.6, color="#d1495b", alpha=0.15, label="Papakonstantis (2011) S_r(min) band")
    ax.legend(fontsize=7)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/V_dense_jet.png"),
               "Inclined 60 deg dense-jet dimensionless rise and dilution vs published experimental bands.",
               "Validation")
    return d


if __name__ == "__main__":
    H._ARTIFACTS.clear() if False else None
    # keep existing manifest; append validation artifacts
    if os.path.exists(H.MANIFEST):
        H._ARTIFACTS.extend(json.load(open(H.MANIFEST)))
    print("EOS ...");      validate_eos()
    print("jet ...");      validate_jet()
    print("plume ...");    validate_plume()
    print("dense jet ..."); validate_dense_jet()
    H.save_manifest()
    json.dump(SUMMARY, open("validation/validation_summary.json", "w"), indent=2)
    print(json.dumps(SUMMARY, indent=2))
