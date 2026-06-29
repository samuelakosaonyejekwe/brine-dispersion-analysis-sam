"""
CASE STUDY INPUT DECK
=====================

Holistic, credible (synthetic but realistic) site-specific input data for the
UBDS universal brine-dispersion solver case study.

Site (fictional, credible):
    BAHIA AZUL SEAWATER DESALINATION & BRINE OUTFALL
    A mid-size SWRO desalination plant on a semi-open, headland-sheltered
    embayment, with an offshore multiport diffuser discharging hypersaline
    reject brine (RO concentrate).  The same outfall is engineered to also
    receive, in a later phase, saturated leaching brine from an adjacent
    solution-mined salt-cavern energy-storage scheme - so the assessment spans
    the full salinity range from RO concentrate (~68 psu) to saturated cavern
    brine (~250 psu), exercising the universality of the solver.

All quantities below are documented synthetic values chosen to be physically
and engineering-credible; none are taken from any real consent.

Author: Akosa Samuel Onyejekwe (Independent Researcher)
"""

import numpy as np

# ===========================================================================
# 0.  PROJECT / SITE METADATA
# ===========================================================================
SITE = dict(
    name="Bahia Azul Seawater Desalination & Brine Outfall",
    locality="Bahia Azul (fictional), open Mediterranean-type coast",
    latitude_deg=36.50,          # N  (Coriolis + tidal regime)
    longitude_deg=-4.30,         # E/W (informational)
    chart_datum="Lowest Astronomical Tide (LAT)",
    project_phase="FEED-stage marine dispersion assessment",
    assessor="Akosa Samuel Onyejekwe (Independent Researcher)",
    solver="UBDS - Universal Brine Dispersion Solver v1.0.0",
)

# ===========================================================================
# 1.  PLANT & DISCHARGE (source term)
# ===========================================================================
PLANT = dict(
    technology="Seawater Reverse Osmosis (SWRO)",
    product_capacity_m3d=60000,      # freshwater produced (m3/day)
    recovery_ratio=0.45,             # permeate / feed
    feed_m3d=lambda: 60000 / 0.45,   # ~133,333 m3/d intake
    brine_m3d=lambda: 60000 / 0.45 - 60000,  # ~73,333 m3/d concentrate
)

# Brine (effluent) characterisation
BRINE = dict(
    ro_salinity_psu=68.0,        # RO concentrate (feed 36.5 at 46% recovery)
    ro_excess_temp_C=0.3,        # RO is near-isothermal
    cavern_salinity_psu=250.0,   # saturated solution-mining brine (phase 2)
    cavern_excess_temp_C=2.0,    # leaching brine excess temperature
    # trace impurities in the cavern brine (ug/L) - synthetic, credible, with
    # marine Environmental Quality Standards (EQS) for comparison
    impurities_ugL={
        # parameter : (brine_conc, seawater_conc, EQS_seawater)
        "Arsenic":  (3.2,  0.9,  25.0),
        "Boron":    (3450, 4300, 7000.0),
        "Cadmium":  (0.30, 0.04, 2.5),
        "Chromium": (1.1,  0.8,  15.0),
        "Copper":   (1.8,  1.1,  5.0),
        "Lead":     (2.0,  0.10, 25.0),
        "Mercury":  (0.02, 0.015, 0.30),
        "Nickel":   (2.9,  0.30, 30.0),
        "Zinc":     (4.0,  1.2,  40.0),
    },
)

# Discharge flow scenarios (total volumetric brine flow through the diffuser).
# Per-port flow is deliberately distinct across scenarios (300 / 700 / 600 per
# port) so the near-field behaviour genuinely differs: a single port handles the
# commissioning and ramp-up flows; the second port is opened for full capacity.
# Tideflex-type duckbill ports, 6 inch nominal.
SCENARIOS = [
    dict(id="S1", label="Commissioning (1 port)", flow_m3hr=300,  n_ports=1),
    dict(id="S2", label="Ramp-up (1 port)",       flow_m3hr=700,  n_ports=1),
    dict(id="S3", label="Full capacity (2 ports)", flow_m3hr=1200, n_ports=2),
]
PORT_SWITCH_FLOW = 700.0          # >this flow activates the second port

# Tidal current states sampled for the near-field initial-dilution study
TIDAL_STATES = [
    dict(label="slack",  velocity=0.08),
    dict(label="neap",   velocity=0.28),
    dict(label="spring", velocity=0.55),
]

# ===========================================================================
# 2.  DIFFUSER GEOMETRY
# ===========================================================================
DIFFUSER = dict(
    n_ports_installed=2,
    port_spacing_m=20.0,
    port_diameter_m=0.1524,       # 6 inch
    port_height_m=0.80,           # above seabed
    port_angle_deg=60.0,          # inclined (best practice for dense jets)
    diffuser_bearing_deg=25.0,    # diffuser line orientation (deg from East)
    valve_type="6-inch duckbill (Tideflex-type) check valve",
)
# design-optimisation variants compared in the study
PORT_ANGLE_VARIANTS = [90.0, 60.0, 45.0, 30.0]

# ===========================================================================
# 3.  RECEIVING-WATER / AMBIENT CONDITIONS
# ===========================================================================
AMBIENT = dict(
    background_salinity_psu=36.5,    # ambient (Mediterranean-type)
    background_temperature_C=18.0,
    salinity_seasonal_range=(36.1, 37.2),
    temperature_seasonal_range=(14.5, 23.0),
    water_density_note="EOS-80 with UBDS hypersaline extension",
    outfall_depth_m=18.0,            # water depth at diffuser (chart datum)
    offshore_distance_m=850.0,       # diffuser distance from shore
)

# Regulatory thresholds
THRESHOLDS = dict(
    mixing_zone_radius_m=100.0,      # regulatory mixing-zone boundary
    salinity_threshold_psu=38.5,     # 2 psu above background -> ecological limit
    increment_limit_psu=2.0,         # max permitted salinity increment at MZ edge
    background_psu=36.5,
)

# ===========================================================================
# 4.  TIDAL FORCING (harmonic constituents at the open boundary)
# ===========================================================================
# Amplitudes (m) / phases (deg) - synthetic, semidiurnal meso-tidal regime
TIDE = dict(
    constituents={
        #          amp(m)  phase(deg)
        "M2": (0.95, 0.0),
        "S2": (0.32, 42.0),
        "K1": (0.11, 58.0),
        "O1": (0.08, 50.0),
    },
    spring_neap_factor_spring=1.0,   # M2+S2 in phase  -> spring
    spring_neap_factor_neap=0.42,    # M2-S2           -> neap
    sn_phase_lag_deg=26.0,           # S->N progressive-wave lag (drives current)
    principal_current_bearing_deg=205.0,  # NNW-SSE flood/ebb axis (deg from E)
)

# ===========================================================================
# 5.  COMPUTATIONAL DOMAIN, MESH & MODEL SETUP
# ===========================================================================
# Medium-field (high-resolution) domain centred on the outfall
MEDIUM_FIELD = dict(
    Lx_m=2500.0, Ly_m=2500.0,        # domain size
    dx_m=25.0, dy_m=25.0,            # base resolution (refined near outfall)
    refine_near_outfall_m=5.0,       # nominal near-diffuser resolution (doc)
    outfall_xy=(1250.0, 1300.0),     # diffuser midpoint in domain coords
    n_sigma_layers=9,
)
# Far-field (regional) domain
FAR_FIELD = dict(
    Lx_m=7200.0, Ly_m=7200.0,
    dx_m=60.0, dy_m=60.0,
    outfall_xy=(3600.0, 3900.0),
    n_sigma_layers=4,
)

# Sigma-layer schemes (percentage of water-column depth per layer,
# ordered BOTTOM -> SURFACE). Thin near the bed for the dense plume.
SIGMA_LAYERS = {
    14: [0.1, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.5, 6.5, 8.5, 11, 20, 40],
    9:  [1.0, 1.2, 1.5, 2.5, 4.5, 9.3, 15, 25, 40],
    4:  [15, 15, 35, 35],
}

# Numerical / physical model parameters
MODEL = dict(
    manning_n=0.025,                 # bed roughness (Manning, s/m^(1/3))
    manning_range=(0.022, 0.040),
    horizontal_eddy_viscosity_m2s=2.0,
    smagorinsky_const=0.2,
    horizontal_dispersion_m2s=1.0,   # scalar horizontal dispersion
    vertical_diffusivity_m2s=5e-4,
    gravity_current_coef=12.0,       # UBDS dense-gravity-current closure
    cfl_number=0.45,
    min_timestep_s=0.4,
    spin_up_cycles=1.0,
    sim_cycles=1.5,
    tidal_period_s=12.4206 * 3600,
)

# Tidal snapshot phases within a cycle (for vector/contour plots)
SNAPSHOT_PHASES = dict(high_water=0.00, mid_ebb=0.25, low_water=0.50,
                       mid_flood=0.75)

# Heights above bed at which max-salinity envelopes are extracted
SALINITY_HEIGHTS_M = [0.0, 1.0, 3.0, 6.0]

# ===========================================================================
# 6.  CURRENT-METER (ADCP) VALIDATION STATIONS (synthetic field survey)
# ===========================================================================
# Positions relative to the outfall (m), and the model-vs-observed comparison
# is generated in the runner.
ADCP_STATIONS = [
    dict(name="A0", dx=-150, dy=120,  bins_m=1, note="near-outfall, 1 m bins"),
    dict(name="A1", dx=0,    dy=700,  bins_m=2, note="north of outfall, 2 m bins"),
    dict(name="A2", dx=1200, dy=-300, bins_m=2, note="offshore south, 2 m bins"),
    dict(name="A3", dx=-400, dy=-650, bins_m=1, note="lee of headland eddy"),
    dict(name="A4", dx=300,  dy=-900, bins_m=1, note="headland tip"),
]

# Field-survey window (synthetic) for documentation
SURVEY = dict(
    provider="Independent metocean survey (synthetic)",
    adcp="Bottom-mounted upward-looking Acoustic Doppler Current Profilers",
    neap_window="2024-03-04 to 2024-03-11",
    spring_window="2024-03-12 to 2024-03-19",
    salinity_temp_monitoring="Monthly CTD casts, 12-month baseline (synthetic)",
)


def summary():
    print(f"CASE: {SITE['name']}")
    print(f"  brine flow (full)   : {SCENARIOS[-1]['flow_m3hr']} m3/hr "
          f"({SCENARIOS[-1]['flow_m3hr']*24} m3/d)")
    print(f"  RO brine salinity   : {BRINE['ro_salinity_psu']} psu")
    print(f"  ambient salinity    : {AMBIENT['background_salinity_psu']} psu")
    print(f"  outfall depth       : {AMBIENT['outfall_depth_m']} m")
    print(f"  diffuser            : {DIFFUSER['n_ports_installed']} x 6\" @ "
          f"{DIFFUSER['port_spacing_m']} m, {DIFFUSER['port_angle_deg']} deg")


if __name__ == "__main__":
    summary()
