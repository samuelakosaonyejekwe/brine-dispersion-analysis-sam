# UBDS — Universal Brine Dispersion Solver

**A unified, open, physics-based solver for the near-, medium- and far-field dispersion of brine and other buoyant or negatively-buoyant effluents in coastal and marine waters — with a complete worked case study.**

**Author:** Akosa Samuel Onyejekwe — Independent Researcher
**Version:** UBDS v1.0.0 · Language: Python

---

## Overview

Brine discharges from desalination plants and solution-mined storage caverns are denser than seawater: they sink toward the seabed, where elevated salinity can stress benthic ecosystems. Assessing their impact requires resolving two very different regimes — the small-scale turbulent jet/plume mixing in the immediate vicinity of the diffuser (the **near field**), and the tidally-driven transport and dilution across the wider water body (the **medium and far field**). Conventionally these are handled by two separate, often proprietary, codes with a manual hand-off between them.

**UBDS unifies both regimes in a single, open, dynamically-coupled, mass-conserving framework.** The same solver describes negatively buoyant (brine), positively buoyant (thermal/freshwater) and neutral (tracer) discharges without special-casing, and it carries the near-field result directly into the far-field transport engine, removing the manual transfer and the mass-conservation errors it introduces.

This repository contains the solver package (`ubds/`), the scripts that run the case study end to end, the validation suite, a complete worked case study as a report (`case.pdf`), and the figures, tables and animations that the study produced (`outputs/`).

---

## Why UBDS

- **Unified near-to-far field coupling** in one open framework — no manual hand-off, no inter-code mass-conservation error.
- **Sign-agnostic buoyancy** — one set of governing equations for dense, buoyant and neutral discharges. A dense element naturally decelerates, arcs over and sinks; a light one rises — no special-casing.
- **Dense-water gravity-current closure** in the sigma-layer transport engine, reproducing the strong seabed trapping of brine that 2-D depth-averaged models miss and that full 3-D models capture only at large computational cost.
- **Built-in regulatory metrics** — initial dilution, mixing-zone radius, salinity-increment impact areas, diffusion distances and Environmental Quality Standard (EQS) compliance, computed directly from the solution.
- **Open and reproducible** — documented, physically-standard entrainment and turbulence closures, with all inputs and outputs exposed as plain data.

---

## Scientific approach

**Near field — `UBDS-NF`.** A sign-agnostic Lagrangian "plume-element" integral model in the spirit of JETLAG/VISJET and the US-EPA UM3 routine, written from scratch and generalised across buoyancy sign. The plume is discretised into a chain of cylindrical elements that grow by the *combined entrainment hypothesis* (shear entrainment ∝ velocity excess, plus forced entrainment ∝ ambient cross-flow). Buoyancy enters as a vertical body force `g(ρ_a − ρ_e)/ρ_e`. The model returns the full 3-D trajectory, the dilution-versus-distance curve, and the salinity/density at seabed impact (or terminal rise height).

**Hydrodynamics — `UBDS-HD`.** A 2-D depth-averaged shallow-water engine (mass and momentum with Manning friction, Coriolis and turbulent mixing) driven by tidal forcing, producing the time-varying current field.

**Far field — `UBDS-FF`.** A quasi-3D sigma-layer advection–dispersion engine driven by the calibrated currents, augmented with a buoyancy-scaled inter-layer **gravity-current flux** that reproduces bottom-trapping of dense brine.

Governing equations (top-hat near field; depth-averaged far field):

```
Near field (steady flux-integral form, along jet arclength s):
  d(ρ_e Q)/ds   = ρ_a E
  d(ρ_e Q V)/ds = ρ_a E V_a + (0, 0, (ρ_a − ρ_e) g π b²)
  d(ρ_e Q S)/ds = ρ_a E S_a
  E = 2π b α_s |V − V_a| + 2 b α_f |V_a,⊥|

Far field (shallow-water hydrodynamics + sigma-layer transport):
  ∂η/∂t + ∂(Hu)/∂x + ∂(Hv)/∂y = 0
  ∂(hs)/∂t + ∂(uhs)/∂x + ∂(vhs)/∂y = ∂/∂x(hD_x ∂s/∂x) + ∂/∂y(hD_y ∂s/∂y) + sources
```

---

## Repository structure

```
README.md                  This file
case.pdf                   The full Bahía Azul case-study report (collates everything)

ubds/                      The solver package
  __init__.py              Package API and version
  eos.py                   Equation of state — seawater + hypersaline-brine extension
  nearfield.py             Sign-agnostic Lagrangian integral jet/plume model (UBDS-NF)
  diffuser.py              Multi-port diffuser geometry and plume-merging logic
  hydro.py                 2-D depth-averaged shallow-water engine (UBDS-HD)
  transport.py             Sigma-layer advection–dispersion + gravity-current closure (UBDS-FF)
  metrics.py               Dilution, mixing-zone, impact-area and EQS-compliance metrics
  viz.py                   Publication plotting helpers (colour-blind safe; never pure black)

case_inputs.py             All site-specific inputs (synthetic, documented, physically credible)
case_geometry.py           Bathymetry and nested-domain construction
run_case_study.py          Runs the whole study and writes outputs/ + the artifact manifest
run_helpers.py             Artifact registry and CSV helpers
make_animations.py         Tidal-cycle GIFs and filmstrip
build_docx.py              Assembles case.docx (-> case.pdf) from the manifest
render_pdf_refs.py         Reference-figure rendering

validation/
  validate.py              Benchmarks vs analytical laws and published experimental data
  SOURCES.md               Provenance of every benchmark and reference value
  validation_summary.json  Machine-readable benchmark results

outputs/
  figures/                 Charts, maps, contours, dilution curves, current vectors, profiles
  csv/                     The tabular data behind every figure
  animations/              GIF animations + filmstrip of the dispersion over a tidal cycle
```

Reproduce the study with `python run_case_study.py`, then `python validation/validate.py`,
`python make_animations.py` and `python build_docx.py`.

---

## The solver (`ubds`)

The package can be imported and used on its own:

```python
from ubds import eos, nearfield, hydro, transport, diffuser, metrics, viz
```

**Equation of state** — seawater density with a hypersaline-brine extension, and the reduced gravity (buoyancy) that drives the plume:

```python
from ubds import eos

rho_ambient = eos.density(34.2, 17.0)   # ambient seawater density (kg/m3)
rho_brine   = eos.density(68.0, 19.0)   # RO concentrate density   (kg/m3)

# Reduced gravity of the discharge relative to ambient (g' > 0 => sinks)
g_prime = eos.buoyancy_g_prime(68.0, 19.0, 34.2, 17.0)
```

**Near-field initial dilution** — define the ambient and the port discharge, then simulate the jet/plume:

```python
from ubds import nearfield

amb = nearfield.Ambient(salinity=34.2, temperature=17.0, current=0.15, depth=27.0)
disch = nearfield.Discharge(
    flow_m3hr=250.0, diameter_m=0.1524, salinity=68.0, temperature=19.0,
    z0=0.75, theta_deg=60.0,            # 60-deg inclined dense jet
)

result = nearfield.simulate(disch, amb)   # full 3-D trajectory + dilution curve
```

The `result` carries the plume trajectory, the dilution-versus-distance curve, and the salinity/density at seabed impact. The `hydro` and `transport` modules then provide the depth-averaged current field and the sigma-layer far-field dispersion, and `metrics` turns any salinity field into regulatory quantities (mixing-zone radius, salinity-increment areas, diffusion distances, EQS compliance, and model-skill scores).

---

## The case study (`case.pdf`)

`case.pdf` is the complete **Bahía Azul** worked example produced end-to-end with UBDS — a mid-size seawater reverse-osmosis desalination plant discharging hypersaline concentrate through an offshore inclined multiport diffuser, with provision for later receipt of saturated brine from an adjacent salt-cavern energy-storage scheme.

The report covers:

- bathymetry and nested mesh construction for the embayment;
- ADCP current and tidal-level calibration of the hydrodynamic engine;
- near-field initial-dilution trajectories and dilution curves for both brine types and all flow scenarios;
- sigma-layer resolution sensitivity;
- medium- and far-field salinity-increment envelopes;
- tidal-phase snapshots with current vectors and vertical profiles through the diffuser;
- diffuser design optimisation (port angle and layout);
- the outfall-siting / intake-recirculation study;
- trace-impurity EQS compliance; and
- the validation suite.

All of the figures and animations behind the report are provided in `outputs/`.

---

## Validation

UBDS was checked against canonical analytical laws and published experimental data. Representative results:

| Benchmark | UBDS | Reference | Source |
|---|---|---|---|
| Seawater/brine density | max error **0.40 %** | EOS-80 / CRC densities | UNESCO (1981); CRC Handbook |
| Momentum jet — dilution slope | **0.228** per z/D | 0.25–0.32 — *below the band* | Fischer et al. (1979) |
| Momentum jet — spread rate | **0.114** db/dz | ≈ 0.11 | Fischer et al. (1979) |
| Pure plume — volume-flux exponent | **1.60** | 5/3 ≈ 1.67 — *4 % below* | Morton, Taylor & Turner (1956) |
| Inclined 60° dense jet — return distance | **x_r/(D·Fr) = 2.57** | 2.2–3.3 | Papakonstantis et al. (2011) |
| Inclined 60° dense jet — return dilution | **S_r(min)/Fr = 0.59** | 0.4–0.6 | Papakonstantis et al. (2011) |
| Inclined 60° dense jet — terminal rise | **z_t/(D·Fr) = 1.53** | 1.6–2.2 — *outside the band* | Papakonstantis et al. (2011) |

Three quantities sit just outside their reference ranges: the round-jet dilution slope, the pure-plume
exponent and the dense-jet terminal rise. Each is stated above rather than rounded into agreement. Refitting the entrainment coefficients
closes it and lowers the calibration RMS from 7.8 % to 2.6 %, but pushes the independent 60° benchmark
out of band — so the shipped coefficients stand and the deviation is reported rather than tuned away.

**On the hydrodynamic "calibration".** Bahía Azul is a fictitious site, so there are no measured
currents for it. The ADCP and tidal-level series the model is scored against are the model solution
itself, perturbed by a 3 % bias and random noise. The resulting Willmott and Nash–Sutcliffe scores
(d = 0.992–0.999, NSE = 0.967–0.996) therefore quantify that imposed perturbation and exercise the
skill-metric pipeline. **They are not evidence of agreement with field data** and are not counted as
validation. The table above is the independent evidence.

**Selected references / data sources**

- Fischer, H.B., List, E.J., Koh, R.C.Y., Imberger, J. & Brooks, N.H. (1979). *Mixing in Inland and Coastal Waters.* Academic Press.
- Morton, B.R., Taylor, G.I. & Turner, J.S. (1956). Turbulent gravitational convection from maintained and instantaneous sources. *Proc. R. Soc. Lond. A* 234, 1–23.
- Papakonstantis, I.G., Christodoulou, G.C. & Papanicolaou, P.N. (2011). Inclined negatively buoyant jets. *J. Hydraulic Research* 49(1), 3–22.
- Roberts, P.J.W., Ferrier, A. & Daviero, G. (1997). Mixing in inclined dense jets. *J. Hydraulic Engineering* 123(8), 693–699.
- Lee, J.H.W. & Chu, V.H. (2003). *Turbulent Jets and Plumes — A Lagrangian Approach.* Kluwer.
- Jirka, G.H. (2004). Integral model for turbulent buoyant jets in unbounded stratified flows. *Environmental Fluid Mechanics* 4, 1–56.
- Millero, F.J. & Poisson, A. (1981). International one-atmosphere equation of state of seawater. *Deep-Sea Research* 28A, 625–629; UNESCO (1981) Tech. Papers Mar. Sci. 36.

> All site-specific inputs (bathymetry, tides, ADCP currents, CTD profiles, brine chemistry) were synthesised by the author for this case study; they are engineering-credible and are not drawn from any real consent.

---

## Citation

> Onyejekwe, A.S. (2026). *UBDS — Universal Brine Dispersion Solver, v1.0.0, and the Bahía Azul brine-outfall case study.*

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

---

## License

Released under the [MIT License](LICENSE) — use, modify and redistribute it freely,
keeping the copyright notice. The software is provided without warranty; it is a
research code, and the Bahía Azul case study runs on synthetic inputs, so it is not a
substitute for a site-specific assessment against real survey data.

---

## Status & contact

UBDS is an independent research project by **Akosa Samuel Onyejekwe**. Questions, suggestions and collaboration are welcome via the repository's issues.
