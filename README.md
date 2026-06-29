# UBDS — Universal Brine Dispersion Solver & Bahía Azul Case Study

A novel, open, physics-based solver for the near-, medium- and far-field
dispersion of brine (and any buoyant/neutral effluent) in coastal waters, with
a complete worked case study and validation suite.

**Author:** Akosa Samuel Onyejekwe (Independent Researcher)

## What it does
UBDS unifies, in a single dynamically-coupled, mass-conserving framework:
- a **near-field** Eulerian integral jet/plume model (sign-agnostic buoyancy —
  dense brine, buoyant thermal, or neutral),
- a **2-D depth-averaged shallow-water** hydrodynamic engine, and
- a **quasi-3D sigma-layer advection–dispersion** transport engine with a novel
  **dense gravity-current closure** that reproduces bottom-trapping of brine.

## Project layout
```
README.md                 This file
case.docx                 The case-study report (collates everything)

case_inputs.py            All case-study input data (site-specific)
case_geometry.py          Bathymetry / mesh builder (Bahia Azul embayment)
run_case_study.py         Master runner (Parts A-H) -> figures, CSVs
run_helpers.py            Artifact registry + field interpolation
render_pdf_refs.py        Renders reference figures from the 3 source PDFs
make_animations.py        Animated GIF + filmstrip outputs
build_docx.py             Assembles case.docx

ubds/                     The solver package
  eos.py                  Seawater + hypersaline-brine equation of state
  nearfield.py            Near-field integral jet/plume model
  hydro.py                2-D shallow-water hydrodynamic engine
  transport.py            Sigma-layer transport + gravity-current closure
  diffuser.py             Multi-port diffuser geometry
  metrics.py              Dilution / mixing-zone / EQS-compliance metrics
  viz.py                  Plotting style (no pure black)

validation/
  validate.py             Validation against canonical laws & published data
  calibrate_nearfield.py  Near-field calibration record
  SOURCES.md              All external data sources
  validation_summary.json Validation results

reference_pdfs/           The three guidance studies (PDF + extracted text)

outputs/
  figures/                All charts, maps, contours, curves, vectors, profiles
  csv/                    Every dataset behind the figures
  animations/             GIF animations + filmstrip
  pdf_reference_figures/  Reference figures rendered from the 3 source studies
  cache/                  Cached hydrodynamic fields (.npz)
  logs/                   Run logs
  _backup_v1/             Safety backup of an earlier output set
  manifest.json           Registry of all artifacts
  results.pkl             Numeric results for the report builder
```

## How to run (in order)
```bash
python3 run_case_study.py        # hydrodynamics + near/far-field + all outputs
python3 render_pdf_refs.py       # reference figures from the source studies
python3 -m validation.validate   # validation suite
python3 make_animations.py       # animations
python3 build_docx.py            # assemble case.docx
```

## Outputs
Bathymetry/mesh, ADCP current & tidal-level calibration, near-field trajectories
and dilution curves, sigma-layer sensitivity, medium- and far-field salinity
envelopes, tidal-phase snapshots with current vectors, vertical profiles,
diffuser design optimisation, intake-recirculation siting study, trace-impurity
EQS compliance, validation plots, animations — all as figures + CSVs, collated
into `case.docx`.

## Validation (see validation/SOURCES.md)
Canonical round-jet (Fischer et al. 1979) and plume (Morton–Taylor–Turner 1956)
laws; inclined dense-jet data (Papakonstantis et al. 2011); EOS-80 / CRC
densities; synthetic ADCP current dataset.
