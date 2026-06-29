"""
Render representative output figures from the three reference studies so they
can be placed alongside the UBDS case-study outputs for comparison.

These are the author's reference materials (the three guidance PDFs); they are
reproduced here at page level purely for methodological comparison and are
clearly attributed.

    python3 render_pdf_refs.py

Outputs:
    outputs/pdf_reference_figures/*.png  (registered under a report section)
"""
import os, json
import fitz
import run_helpers as H

OUT = "outputs/pdf_reference_figures"
os.makedirs(OUT, exist_ok=True)
if os.path.exists(H.MANIFEST):
    H._ARTIFACTS.extend(json.load(open(H.MANIFEST)))

SEC = "Reference output figures (source studies)"

# (pdf path, fitz page index, short caption)
PAGES = [
    ("reference_pdfs/appendix_b.pdf", 8,
     "Reference Study 1 (salt-cavern brine outfall): model mesh, bathymetry and current-meter comparison."),
    ("reference_pdfs/appendix_b.pdf", 25,
     "Reference Study 1: near-field brine-plume trajectory and initial-dilution prediction."),
    ("reference_pdfs/appendix_b.pdf", 33,
     "Reference Study 1: sigma-layer scheme and layer-resolution sensitivity."),
    ("reference_pdfs/appendix_b.pdf", 39,
     "Reference Study 1: maximum seabed-salinity envelope and salinity/current snapshots."),
    ("reference_pdfs/appendix_b.pdf", 55,
     "Reference Study 1: far-field maximum seabed-salinity envelopes."),
    ("reference_pdfs/water_15_02402.pdf", 4,
     "Reference Study 2 (Laoshan Bay desalination): computational mesh and bathymetry of the bay."),
    ("reference_pdfs/water_15_02402.pdf", 6,
     "Reference Study 2: tidal flow-field simulation results."),
    ("reference_pdfs/water_15_02402.pdf", 7,
     "Reference Study 2: salinity-field results and salinity profile for the layout plans."),
    ("reference_pdfs/ijciet.pdf", 4,
     "Reference Study 3 (Chennai desalination): bathymetry and flexible computational mesh."),
    ("reference_pdfs/ijciet.pdf", 5,
     "Reference Study 3: hydrodynamic calibration (observed vs computed tidal level and current)."),
    ("reference_pdfs/ijciet.pdf", 7,
     "Reference Study 3: contours of salinity rise and time history at intake/outfall."),
]

for path, idx, cap in PAGES:
    d = fitz.open(path)
    if idx >= d.page_count:
        continue
    pix = d[idx].get_pixmap(dpi=120)
    name = f"{OUT}/{os.path.basename(path).split('.')[0]}_fig_p{idx}.png"
    pix.save(name)
    H.register("figure", name, cap, SEC)
    print("rendered", name)

H.save_manifest()
print(f"{len([p for p in PAGES])} reference figures rendered & registered.")
