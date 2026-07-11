"""
BAHIA AZUL BRINE-OUTFALL CASE STUDY  -  master simulation runner
================================================================

Runs the full UBDS workflow for the case study and generates the complete set
of engineering outputs (figures, tables, CSVs) collating every output type
present in the two guidance studies, plus additional UBDS-specific output.

    python3 run_case_study.py

Outputs:
    outputs/figures/*.png   - all charts, contours, curves, vectors, profiles
    outputs/csv/*.csv       - every dataset behind the figures
    outputs/results.pkl     - numeric results for the report builder
    outputs/manifest.json   - registry of all artifacts with captions

Author: Akosa Samuel Onyejekwe (Independent Researcher)
"""
import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import case_inputs as C
import case_geometry as CG
from case_geometry import build_bathymetry, nearest_cell
from ubds import hydro, nearfield as nf, transport as tr, diffuser as df, metrics, viz, eos
import run_helpers as H

os.makedirs(H.FIG_DIR, exist_ok=True)
os.makedirs(H.CSV_DIR, exist_ok=True)
H.reset_artifacts()
RESULTS = {}

# place the outfall a realistic distance off the bay shore in each domain
C.MEDIUM_FIELD["outfall_xy"] = CG.outfall_position(
    C.MEDIUM_FIELD["Lx_m"], C.MEDIUM_FIELD["Ly_m"], C.AMBIENT["offshore_distance_m"], 0.52)
C.FAR_FIELD["outfall_xy"] = CG.outfall_position(
    C.FAR_FIELD["Lx_m"], C.FAR_FIELD["Ly_m"], C.AMBIENT["offshore_distance_m"], 0.54)

# tide states -> phase lag (calibrated to peak current at outfall)
TIDE_LAG = {"neap": 3.0, "spring": 6.0}
FORCE = dict(amp={k: v[0] for k, v in C.TIDE["constituents"].items()},
             pha={k: v[1] for k, v in C.TIDE["constituents"].items()})
PRINC_BEARING = 100.0   # along-shore principal current axis (deg from East)


# ===========================================================================
#  HYDRODYNAMICS
# ===========================================================================
def run_hydro_state(field, label):
    """Run (or load cached) coarse hydrodynamics for a tide state."""
    fld = C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD
    dxc = 50.0 if field == "medium" else 90.0
    os.makedirs("outputs/cache", exist_ok=True)
    cache = f"outputs/cache/hydro_{field}_{label}.npz"
    grid_c, Xc, Yc = build_bathymetry(fld["Lx_m"], fld["Ly_m"], dxc, dxc,
                                      fld["outfall_xy"])
    if os.path.exists(cache):
        d = np.load(cache)
        return dict(grid=grid_c, X=Xc, Y=Yc, u=d["u"], v=d["v"], eta=d["eta"],
                    t=d["t"], dxc=dxc)
    forcing = hydro.TidalForcing(amp=FORCE["amp"], pha=FORCE["pha"],
                                 sn_lag_deg=TIDE_LAG[label])
    m = hydro.HydroModel(grid_c, forcing, manning=C.MODEL["manning_n"],
                         Ah=C.MODEL["horizontal_eddy_viscosity_m2s"],
                         latitude=C.SITE["latitude_deg"],
                         east_bc="flather")
    dt = max(m.cfl_dt(C.MODEL["cfl_number"]), C.MODEL["min_timestep_s"])
    T = C.MODEL["tidal_period_s"]
    res = hydro.run_hydro(m, t_end=2.0 * T + 3600, dt=dt,
                          spin_up=C.MODEL["spin_up_cycles"] * T, sample_dt=T / 36)
    np.savez_compressed(cache, u=res["u"], v=res["v"], eta=res["eta"],
                        t=res["t"])
    return dict(grid=grid_c, X=Xc, Y=Yc, u=res["u"], v=res["v"],
                eta=res["eta"], t=res["t"], dxc=dxc)


# ===========================================================================
#  PLOT PRIMITIVES (no black)
# ===========================================================================
def _outfall_marker(ax, xy, mz=None):
    ax.plot(xy[0], xy[1], marker="*", ms=15, mfc="#e6c84f", mec="#9b2226",
            mew=1.4, zorder=8, label="Diffuser")
    if mz:
        e = Ellipse(xy, 2 * mz, 2 * mz, fill=False, ec="#9b2226", lw=1.6,
                    ls="--", zorder=8)
        ax.add_patch(e)


# current-domain basemap (set per part so contour plots can draw the coastline)
_CURGRID = {}


def set_basemap(grid, X, Y):
    _CURGRID.update(grid=grid, X=X, Y=Y)


def _north_arrow(ax):
    ax.annotate("N", xy=(0.95, 0.97), xytext=(0.95, 0.86),
                xycoords="axes fraction", textcoords="axes fraction",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#1b2a4a",
                arrowprops=dict(arrowstyle="-|>", color="#1b2a4a", lw=2.2))


def _scale_bar(ax, X, Y, frac=0.25):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    span = x1 - x0
    # pick a round length close to frac of the view
    raw = span * frac
    nice = min([50, 100, 200, 250, 500, 1000, 2000], key=lambda v: abs(v - raw))
    xa = x0 + 0.06 * span; ya = y0 + 0.06 * (y1 - y0)
    ax.plot([xa, xa + nice], [ya, ya], color="#1b2a4a", lw=3, zorder=9)
    ax.text(xa + nice / 2, ya + 0.018 * (y1 - y0),
            f"{nice} m" if nice < 1000 else f"{nice/1000:.0f} km",
            ha="center", va="bottom", fontsize=8, color="#1b2a4a", zorder=9)


def _draw_basemap(ax, grid, X, Y, labels=False):
    """Fill land, draw the coastline, and add a north arrow + scale bar."""
    hraw = getattr(grid, "h_raw", grid.h)
    land = hraw <= 0.0
    if land.any():
        ax.contourf(X, Y, land.astype(float), levels=[0.5, 1.5],
                    colors=["#d9c7a3"], zorder=4)
        ax.contour(X, Y, hraw, levels=[0.0], colors=["#8a6d3b"],
                   linewidths=1.4, zorder=5)
    if labels:
        Lx = grid.nx * grid.dx; Ly = grid.ny * grid.dy
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        for key, (name, fx, fy) in CG.FEATURES.items():
            px, py = fx * Lx, fy * Ly
            if x0 <= px <= x1 and y0 <= py <= y1:      # only label what is in view
                ax.text(px, py, name, fontsize=8.5, style="italic",
                        color="#5b4a2f", ha="center", va="center", zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=0.4, pad=1))
    _north_arrow(ax)
    _scale_bar(ax, X, Y)


def plot_bathy(grid, X, Y, title, fname, section, outfall_xy=None,
               stations=None, zones=False):
    fig, ax = viz.new_ax((7.4, 6.6), title, "Easting (m)", "Northing (m)")
    hraw = getattr(grid, "h_raw", grid.h)
    land = hraw <= 0.0
    hh = np.ma.masked_where(land, grid.h)
    pc = ax.pcolormesh(X, Y, hh, cmap=viz.BATHY_CMAP, shading="auto")
    ax.contour(X, Y, grid.h, levels=[5, 10, 15, 20, 25], colors=["#33526e"],
               linewidths=0.6, alpha=0.7)
    cb = fig.colorbar(pc, ax=ax, shrink=0.85); cb.set_label("Water depth (m)")
    _draw_basemap(ax, grid, X, Y, labels=True)
    if outfall_xy is not None:
        _outfall_marker(ax, outfall_xy)
    if stations:
        for s in stations:
            sx, sy = outfall_xy[0] + s["dx"], outfall_xy[1] + s["dy"]
            ax.plot(sx, sy, "o", mfc="#d1495b", mec="#3a1f2b", ms=8, zorder=7)
            ax.annotate(s["name"], (sx, sy), textcoords="offset points",
                        xytext=(6, 5), fontsize=9, color="#1b2a4a", weight="bold")
    if zones:
        # medium & far field nested boxes
        mf = C.MEDIUM_FIELD
        ax.add_patch(plt.Rectangle((outfall_xy[0]-mf["Lx_m"]/2, outfall_xy[1]-mf["Ly_m"]/2),
                     mf["Lx_m"], mf["Ly_m"], fill=False, ec="#d1495b", lw=2, ls="-"))
        ax.text(outfall_xy[0]-mf["Lx_m"]/2+60, outfall_xy[1]+mf["Ly_m"]/2-120,
                "Medium-field zone", color="#d1495b", weight="bold", fontsize=9)
    ax.set_aspect("equal"); ax.legend(loc="upper left", framealpha=0.92, fontsize=8)
    p = H.register("figure", viz.save(fig, f"{H.FIG_DIR}/{fname}"), title, section)
    return p


def _tidal_clock(ax, phase, mean_uv=None):
    """Small inset 'tidal clock' showing the phase + mean-current arrow."""
    iax = ax.inset_axes([0.015, 0.70, 0.20, 0.27])
    th = np.linspace(0, 2 * np.pi, 100)
    iax.plot(np.cos(th), np.sin(th), color="#7b8fa3", lw=1)
    ang = 2 * np.pi * phase
    iax.plot([0, np.sin(ang)], [0, np.cos(ang)], color="#d1495b", lw=2.2)
    if mean_uv is not None:
        mu, mv = mean_uv
        nrm = (mu**2 + mv**2) ** 0.5 or 1
        iax.annotate("", xy=(0.8*mu/nrm, 0.8*mv/nrm), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#2a6f97", lw=2))
    iax.set_xlim(-1.2, 1.2); iax.set_ylim(-1.2, 1.2); iax.set_aspect("equal")
    iax.axis("off")
    iax.text(0, -1.15, "tidal phase", ha="center", fontsize=6, color="#1b2a4a")


def plot_salinity_field(X, Y, S2d, title, fname, section, outfall_xy,
                        mz=None, s_bg=36.5, vectors=None, levels=None,
                        threshold=38.5, zoom_box=None, phase=None,
                        cmap=None, current_speed_bg=False):
    fig, ax = viz.new_ax((7.6, 6.6), title, "Easting (m)", "Northing (m)")
    if levels is None:
        levels = viz.nonlinear_salinity_levels(s_bg)
    cf = ax.contourf(X, Y, S2d, levels=levels, cmap=cmap or viz.SAL_CMAP, extend="max")
    if S2d.max() >= threshold:
        ax.contour(X, Y, S2d, levels=[threshold], colors=["#3a9278"], linewidths=2.2)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, ticks=levels[::3])
    cb.set_label("Salinity (psu)")
    mean_uv = None
    if vectors is not None:
        u, v = vectors
        sk = max(1, X.shape[0] // 24)
        spd = np.hypot(u, v)
        ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], u[::sk, ::sk], v[::sk, ::sk],
                  spd[::sk, ::sk], cmap="cool", scale=11, width=0.0032, alpha=0.95)
        mean_uv = (float(np.mean(u)), float(np.mean(v)))
    # apply the zoom FIRST so the scale bar / annotations use the final view
    if zoom_box:
        ax.set_xlim(zoom_box[0]); ax.set_ylim(zoom_box[1])
    ax.set_aspect("equal")
    # draw the coastline/land basemap if the current-domain grid matches
    g = _CURGRID.get("grid")
    if g is not None and getattr(g, "h", np.zeros((1,))).shape == S2d.shape:
        _draw_basemap(ax, g, _CURGRID["X"], _CURGRID["Y"],
                      labels=(zoom_box is None))
    _outfall_marker(ax, outfall_xy, mz)
    if phase is not None:
        _tidal_clock(ax, phase, mean_uv)
    # quantitative info-box (gives every contour figure distinct numbers)
    smax = float(np.nanmax(S2d))
    dxg = float(abs(X[0, 1] - X[0, 0])); dyg = float(abs(Y[1, 0] - Y[0, 0]))
    area_thr = float(np.count_nonzero(S2d >= threshold)) * dxg * dyg / 1e6
    area_05 = float(np.count_nonzero(S2d >= s_bg + 0.5)) * dxg * dyg / 1e6
    info = (f"max = {smax:.1f} psu\n"
            f"area >{threshold:.0f} psu = {area_thr:.3f} km²\n"
            f"area >+0.5 psu = {area_05:.3f} km²")
    if mean_uv is not None:
        info += f"\nmean current = {np.hypot(*mean_uv):.2f} m/s"
    ax.text(0.985, 0.015, info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#1b2a4a",
            bbox=dict(fc="white", ec="#7b8fa3", alpha=0.88, boxstyle="round,pad=0.3"),
            zorder=10)
    p = H.register("figure", viz.save(fig, f"{H.FIG_DIR}/{fname}"), title, section)
    return p


def plot_salinity_panels(panels, title, fname, section, outfall_xy, levels=None,
                         mz=None, s_bg=36.5, threshold=38.5, zoom_box=None,
                         ncols=None, cmap=None, cb_label="Salinity (psu)"):
    """
    Multi-panel salinity comparison (shared colour scale) - used where several
    cases would otherwise be near-identical standalone maps (e.g. flow scenarios
    or heights above bed).  Each panel carries its own label and max/area box.
    """
    n = len(panels)
    ncols = ncols or (n if n <= 3 else 2)
    nrows = int(np.ceil(n / ncols))
    if levels is None:
        levels = viz.nonlinear_salinity_levels(s_bg)
    g = _CURGRID.get("grid"); X = _CURGRID.get("X"); Y = _CURGRID.get("Y")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.4 * nrows),
                             squeeze=False, layout="constrained")
    fig.suptitle(title, color=viz.INK, fontweight="bold", fontsize=12)
    cf = None
    for idx, pan in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        S2d = pan["S"]
        cf = ax.contourf(pan["X"], pan["Y"], S2d, levels=levels,
                         cmap=cmap or viz.SAL_CMAP, extend="max")
        if S2d.max() >= threshold:
            ax.contour(pan["X"], pan["Y"], S2d, levels=[threshold],
                       colors=["#3a9278"], linewidths=2.0)
        if zoom_box:
            ax.set_xlim(zoom_box[0]); ax.set_ylim(zoom_box[1])
        ax.set_aspect("equal")
        gg = pan.get("grid", g)
        if gg is not None and gg.h.shape == S2d.shape:
            _draw_basemap(ax, gg, pan["X"], pan["Y"], labels=False)
        _outfall_marker(ax, outfall_xy, mz)
        dxg = abs(pan["X"][0, 1] - pan["X"][0, 0]); dyg = abs(pan["Y"][1, 0] - pan["Y"][0, 0])
        area_thr = float(np.count_nonzero(S2d >= threshold)) * dxg * dyg / 1e6
        area_05 = float(np.count_nonzero(S2d >= s_bg + 0.5)) * dxg * dyg / 1e6
        ax.set_title(pan["label"], fontsize=10, color=viz.INK)
        ax.text(0.97, 0.03, f"max {S2d.max():.1f} psu\n>+0.5: {area_05:.3f} km²",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color="#1b2a4a", zorder=10,
                bbox=dict(fc="white", ec="#7b8fa3", alpha=0.88, boxstyle="round,pad=0.25"))
        ax.set_xlabel("Easting (m)", fontsize=9); ax.set_ylabel("Northing (m)", fontsize=9)
        ax.tick_params(labelsize=8)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    cb = fig.colorbar(cf, ax=axes, shrink=0.8, location="right")
    cb.set_label(cb_label)
    return H.register("figure", viz.save(fig, f"{H.FIG_DIR}/{fname}"), title, section)


# ===========================================================================
#  PART A - NEAR-FIELD INITIAL DILUTION
# ===========================================================================
def part_A_nearfield():
    print("[A] near-field initial dilution ...")
    sec = "Near-field initial dilution"
    rows = []
    traj_store = {}
    brine_types = {"RO concentrate": (C.BRINE["ro_salinity_psu"],
                                      C.AMBIENT["background_temperature_C"] + C.BRINE["ro_excess_temp_C"]),
                   "Saturated cavern brine": (C.BRINE["cavern_salinity_psu"],
                                      C.AMBIENT["background_temperature_C"] + C.BRINE["cavern_excess_temp_C"])}
    Sa = C.AMBIENT["background_salinity_psu"]; Ta = C.AMBIENT["background_temperature_C"]
    diff = df.Diffuser(n_ports_installed=C.DIFFUSER["n_ports_installed"],
                       port_spacing=C.DIFFUSER["port_spacing_m"],
                       port_diameter=C.DIFFUSER["port_diameter_m"],
                       port_height=C.DIFFUSER["port_height_m"],
                       theta_deg=C.DIFFUSER["port_angle_deg"])
    for bname, (S0, T0) in brine_types.items():
        for sc in C.SCENARIOS:
            for ts in C.TIDAL_STATES:
                amb = nf.Ambient(salinity=Sa, temperature=Ta,
                                 depth=C.AMBIENT["outfall_depth_m"], current=ts["velocity"])
                tmpl = nf.Discharge(salinity=S0, temperature=T0)
                out = df.run_diffuser(sc["flow_m3hr"], amb, tmpl, diff,
                                      switch_flow=C.PORT_SWITCH_FLOW,
                                      n_active=sc["n_ports"])
                r = out["result"]
                rows.append(dict(brine=bname, scenario=sc["id"], flow_m3hr=sc["flow_m3hr"],
                                 n_ports=out["n_active"], tide=ts["label"],
                                 velocity_ms=ts["velocity"], exit_velocity_ms=round(out["exit_velocity"], 2),
                                 dilution=round(r.impact_dilution, 1),
                                 seabed_salinity_psu=round(r.impact_salinity, 2),
                                 rise_height_m=round(r.rise_height, 2),
                                 impact_distance_m=round(r.impact_distance, 2),
                                 termination=r.termination))
                traj_store[(bname, sc["id"], ts["label"])] = r
    dfres = pd.DataFrame(rows)
    # Every dense-brine run must actually reach the bed, otherwise the columns
    # named seabed_* / impact_* describe some other point on the trajectory.
    _bad = dfres[dfres.termination != "seabed"]
    if len(_bad):
        raise RuntimeError("near-field runs that never reached the seabed - the "
                           f"'seabed_salinity_psu' column would be mislabelled:\n{_bad}")
    H.write_csv(dfres, "A_nearfield_initial_dilution.csv",
                "Near-field initial-dilution results (bulk top-hat dilution and salinity at first "
                "seabed contact) for all brine types, flow scenarios and tidal states.", sec)
    RESULTS["nearfield"] = dfres

    # trajectory + dilution figures (RO concentrate, per scenario)
    tide_colors = viz.TIDAL_VEL_COLORS
    for bn in ["RO concentrate", "Saturated cavern brine"]:
        for sc in C.SCENARIOS:
            per_port = sc["flow_m3hr"] / sc["n_ports"]
            # ---- rich 2-panel near-field figure: trajectory (coloured by
            #      salinity) + dilution/salinity vs distance ------------------
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.6), layout="constrained")
            # The near-field solution depends on the flow through ONE port, so S1 and S3
            # (both 300 m3/hr per port) are the same jet.  Say so on the figure rather
            # than let two identical panels read as two independent results.
            _twin = [o["id"] for o in C.SCENARIOS
                     if o["id"] != sc["id"] and o["flow_m3hr"] / o["n_ports"] == per_port]
            _same = (f"  -  identical to {', '.join(_twin)} (same flow per port)"
                     if _twin else "")
            _pl = "port" if sc["n_ports"] == 1 else "ports"
            fig.suptitle(f"Near-field behaviour - {bn}  |  {sc['flow_m3hr']} m3/hr, "
                         f"{sc['n_ports']} {_pl} = {per_port:.0f} m3/hr/port{_same}",
                         color=viz.INK, fontweight="bold", fontsize=16)
            # LEFT: trajectory in the water column, line coloured by salinity
            from matplotlib.collections import LineCollection
            smax = max(traj_store[(bn, sc["id"], ts["label"])].salinity.max()
                       for ts in C.TIDAL_STATES)
            for _ti, ts in enumerate(C.TIDAL_STATES):
                r = traj_store[(bn, sc["id"], ts["label"])]
                xx = np.hypot(r.x, r.y)
                pts = np.array([xx, r.z]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc = LineCollection(segs, cmap=viz.SAL_CMAP, lw=3.4,
                                    norm=plt.Normalize(C.AMBIENT["background_salinity_psu"], smax))
                lc.set_array(r.salinity)
                axL.add_collection(lc)
                axL.fill_between(xx, r.z - r.b, r.z + r.b,
                                 color=tide_colors[ts["label"]], alpha=0.10)
                axL.plot(xx[r.impact_index], r.z[r.impact_index], "v",
                         color=tide_colors[ts["label"]], ms=9, mec="#1b2a4a")
                # The cavern jet lands within ~3-6 m of the port for all three tides, so
                # labels pinned at the impact point collide.  Stack them vertically.
                axL.annotate(f"{ts['label']} ({ts['velocity']} m/s)",
                             (xx[-1], r.z[-1]), xytext=(6, 6 + 12 * _ti),
                             textcoords="offset points", fontsize=11, fontweight="bold",
                             color=tide_colors[ts["label"]])
            axL.axhline(0, color="#8a6d3b", lw=2.0)
            axL.axhline(C.DIFFUSER["port_height_m"], color="#7b8fa3", ls=":", lw=1)
            axL.set_xlabel("Horizontal distance from port (m)", fontsize=13)
            axL.set_ylabel("Height above seabed (m)", fontsize=13)
            axL.set_title("Plume trajectory (line colour = salinity)", fontsize=14)
            axL.tick_params(labelsize=12)
            axL.set_ylim(bottom=-0.4); axL.autoscale_view()
            cb = fig.colorbar(lc, ax=axL, shrink=0.8)
            cb.set_label("Salinity (psu)", fontsize=12); cb.ax.tick_params(labelsize=11)
            axL.grid(True, color=viz.GRIDC, alpha=0.5)
            # RIGHT: dilution (solid) and salinity (dashed) vs distance
            axR2 = axR.twinx()
            for ts in C.TIDAL_STATES:
                r = traj_store[(bn, sc["id"], ts["label"])]
                axR.plot(r.s, r.dilution, color=tide_colors[ts["label"]], lw=2.4,
                         label=f"{ts['label']}: x{r.impact_dilution:.0f}, {r.impact_salinity:.1f} psu")
                axR2.plot(r.s, r.salinity, color=tide_colors[ts["label"]], lw=1.4, ls="--", alpha=0.7)
            axR.set_xlabel("Distance along plume centreline (m)", fontsize=13)
            axR.set_ylabel("Dilution S/S0  (solid)", fontsize=13)
            axR2.set_ylabel("Centreline salinity, psu  (dashed)", color="#5b4a2f", fontsize=13)
            axR.set_title("Dilution & salinity vs distance", fontsize=14)
            axR.tick_params(labelsize=12); axR2.tick_params(labelsize=12)
            leg = axR.legend(title="Tidal velocity (seabed values)", fontsize=11)
            leg.get_title().set_fontsize(11)
            axR.grid(True, color=viz.GRIDC, alpha=0.5)
            H.register("figure", viz.save(fig, f"{H.FIG_DIR}/A_nearfield_{bn[:2]}_{sc['id']}.png"),
                       f"Near-field trajectory, dilution and salinity - {bn}, "
                       f"{sc['flow_m3hr']} m3/hr ({per_port:.0f} m3/hr per port).", sec)

    # ---- master comparison across scenarios (one figure) -------------------
    fig, ax = viz.new_ax((8, 5.2), "Seabed dilution by scenario and tidal state (RO concentrate)",
                         "Tidal velocity (m/s)", "Seabed dilution (S/S0)")
    # Near-field dilution is set by the flow through ONE port, so S3 (1200 m3/hr through
    # four ports) reproduces S1 (300 through one) exactly.  Plotting both stacks two
    # identical curves and makes neither readable; draw the distinct cases and state the
    # equivalence instead.
    cols = {"S1": "#12355b", "S2": "#1b9aaa"}
    for sid in ["S1", "S2"]:
        sc = next(o for o in C.SCENARIOS if o["id"] == sid)
        sub = dfres[(dfres.brine == "RO concentrate") & (dfres.scenario == sid)].sort_values("velocity_ms")
        ax.plot(sub.velocity_ms, sub.dilution, "-o", color=cols[sid], lw=2.4, ms=8,
                label=f"{sc['id']} - {sc['flow_m3hr']} m3/hr, {sc['n_ports']} port"
                      f"{'' if sc['n_ports'] == 1 else 's'}")
    ax.legend(title="Scenario", loc="upper left")
    ax.annotate("S3 (1200 m3/hr, 4 ports) reproduces S1 exactly: both discharge 300 m3/hr per port",
                xy=(0.98, 0.02), xycoords="axes fraction", fontsize=8,
                ha="right", va="bottom", color=viz.INK)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/A_scenario_comparison.png"),
               "Seabed initial dilution by flow scenario and tidal state.", sec)
    return dfres, traj_store, diff


# ===========================================================================
#  PART A2 - PORT-ANGLE DESIGN OPTIMISATION
# ===========================================================================
def part_A2_angle_opt(diff):
    print("[A2] port-angle optimisation ...")
    sec = "Diffuser design optimisation"
    Sa = C.AMBIENT["background_salinity_psu"]; Ta = C.AMBIENT["background_temperature_C"]
    rows = []
    amb = nf.Ambient(salinity=Sa, temperature=Ta, depth=C.AMBIENT["outfall_depth_m"], current=0.28)
    for ang in C.PORT_ANGLE_VARIANTS:
        d = nf.Discharge(flow_m3hr=300, diameter_m=C.DIFFUSER["port_diameter_m"],
                         salinity=C.BRINE["ro_salinity_psu"],
                         temperature=Ta + C.BRINE["ro_excess_temp_C"],
                         z0=C.DIFFUSER["port_height_m"], theta_deg=ang)
        r = nf.simulate(d, amb, n_ports=1)
        rows.append(dict(port_angle_deg=ang, dilution=round(r.impact_dilution, 1),
                         seabed_salinity_psu=round(r.impact_salinity, 2),
                         rise_height_m=round(r.rise_height, 2),
                         impact_distance_m=round(r.impact_distance, 2)))
    d2 = pd.DataFrame(rows)
    H.write_csv(d2, "A2_port_angle_optimisation.csv",
                "Effect of diffuser port discharge angle on near-field dilution (300 m3/hr, neap).", sec)
    fig, ax = viz.new_ax((7, 5), "Diffuser port-angle optimisation",
                         "Port discharge angle (deg from horizontal)", "Seabed dilution (S/S0)")
    ax.plot(d2.port_angle_deg, d2.dilution, "-o", color="#2a6f97", lw=2.4, ms=8)
    ax2 = ax.twinx()
    ax2.plot(d2.port_angle_deg, d2.seabed_salinity_psu, "--s", color="#d1495b", lw=2)
    ax2.set_ylabel("Seabed salinity (psu)", color="#d1495b")
    ax2.tick_params(axis="y", colors="#d1495b")
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/A2_angle_opt.png"),
               "Diffuser port-angle optimisation: dilution and seabed salinity vs discharge angle.", sec)
    RESULTS["angle_opt"] = d2
    return d2


def fine_grid(field="medium"):
    fld = C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD
    g, X, Y = build_bathymetry(fld["Lx_m"], fld["Ly_m"], fld["dx_m"], fld["dy_m"],
                               fld["outfall_xy"])
    return g, X, Y, fld


def drive_fields(hyd, field="medium"):
    """Interpolate coarse hydro (u,v) onto the fine transport grid."""
    fld = C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD
    factor = hyd["dxc"] / fld["dx_m"]
    u = H.interp_series(hyd["u"], factor)
    v = H.interp_series(hyd["v"], factor)
    g, X, Y, _ = fine_grid(field)
    # crop/pad to match grid shape
    u = u[:, :g.ny, :g.nx]; v = v[:, :g.ny, :g.nx]
    return {"u": u, "v": v, "t": hyd["t"]}, g, X, Y


# ===========================================================================
#  PART B - HYDRODYNAMICS, MESH, ADCP VALIDATION, TIDAL FLOW FIELD
# ===========================================================================
def part_B_hydro():
    print("[B] hydrodynamics + validation ...")
    sec = "Hydrodynamics, mesh & calibration"
    med_n = run_hydro_state("medium", "neap")
    med_s = run_hydro_state("medium", "spring")
    far_n = run_hydro_state("far", "neap")
    far_s = run_hydro_state("far", "spring")
    RESULTS["hydro_peak"] = {}

    # --- mesh & bathymetry figures
    gm, Xm, Ym, _ = fine_grid("medium")
    gf, Xf, Yf, _ = fine_grid("far")
    plot_bathy(gm, Xm, Ym, "Medium-field mesh & bathymetry (Bahia Azul)",
               "B_mesh_medium.png", sec, C.MEDIUM_FIELD["outfall_xy"])
    plot_bathy(gf, Xf, Yf, "Far-field (regional) domain & bathymetry",
               "B_mesh_far.png", sec, C.FAR_FIELD["outfall_xy"], zones=True)
    plot_bathy(gm, Xm, Ym, "Diffuser location & ADCP current-meter stations",
               "B_stations.png", sec, C.MEDIUM_FIELD["outfall_xy"],
               stations=C.ADCP_STATIONS)

    # --- ADCP current comparison (SYNTHETIC reference series) ---------------
    # The "ADCP" series below is NOT an independent measurement: it is the model
    # solution itself, perturbed by a fixed proportional bias and random noise.
    # The resulting skill scores therefore quantify that imposed perturbation and
    # exercise the skill-metric pipeline; they are not evidence of agreement with
    # field data, and nothing in the report may present them as such.  Bahia Azul
    # is a fictitious site and no measured currents exist for it.
    rng = np.random.default_rng(7)
    skill_rows = []
    for st in C.ADCP_STATIONS:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), layout="constrained")
        fig.suptitle(f"Modelled vs synthetic ADCP reference - station {st['name']}",
                     color=viz.INK, fontweight="bold")
        for col, (hyd, lab) in enumerate([(med_n, "neap"), (med_s, "spring")]):
            f, g, X, Y = drive_fields(hyd, "medium")
            j, i = nearest_cell(g, (C.MEDIUM_FIELD["outfall_xy"][0] + st["dx"],
                                    C.MEDIUM_FIELD["outfall_xy"][1] + st["dy"]))
            u = f["u"][:, j, i]; v = f["v"][:, j, i]
            t = (f["t"] - f["t"][0]) / 3600.0
            spd = np.hypot(u, v)
            drc = (np.degrees(np.arctan2(u, v)) + 360) % 360   # from North
            # Synthetic reference: the model signal with a small proportional
            # bias plus noise scaled to the local amplitude, so that low-current
            # stations are not swamped by absolute noise.  Good agreement here is
            # imposed, not demonstrated.
            amp = max(float(spd.std()), 0.03)
            obs_spd = np.clip(spd * (1 + 0.03) + rng.normal(0, 0.06 * amp, spd.shape), 0, None)
            obs_dir = (drc + rng.normal(2.5, 4.0, drc.shape)) % 360
            axes[0, col].plot(t, spd, color="#2a6f97", lw=2, label="UBDS model")
            axes[0, col].plot(t, obs_spd, "o", color="#d1495b", ms=3, label="Synthetic ADCP reference")
            axes[0, col].set_title(f"{lab}: speed"); axes[0, col].set_ylabel("Speed (m/s)")
            axes[1, col].plot(t, drc, color="#2a6f97", lw=2)
            axes[1, col].plot(t, obs_dir, "o", color="#d1495b", ms=3)
            axes[1, col].set_title(f"{lab}: direction"); axes[1, col].set_ylabel("Dir (deg from N)")
            axes[1, col].set_xlabel("Time (h)")
            for ax in axes[:, col]:
                ax.grid(True, color=viz.GRIDC, alpha=0.5)
            axes[0, col].legend(fontsize=8)
            skill_rows.append(dict(station=st["name"], tide=lab,
                                   peak_speed_ms=round(spd.max(), 3),
                                   rmse_speed_ms=round(metrics.rmse(spd, obs_spd), 3),
                                   willmott_d=round(metrics.skill_willmott(spd, obs_spd), 3),
                                   nse=round(metrics.nash_sutcliffe(spd, obs_spd), 3)))
        H.register("figure", viz.save(fig, f"{H.FIG_DIR}/B_adcp_{st['name']}.png"),
                   f"Modelled current speed & direction at station {st['name']} (neap & spring), against a "
                   f"synthetic ADCP reference generated from the model solution with a 3 percent bias "
                   f"and random noise. Not an independent measurement.", sec)
    sk = pd.DataFrame(skill_rows)
    H.write_csv(sk, "B_current_validation_skill.csv",
                "Skill of the model against the synthetic ADCP reference series (RMSE, Willmott index, "
                "Nash-Sutcliffe). The reference is the model solution plus an imposed bias and noise, "
                "so these scores measure that perturbation, not agreement with field data.", sec)
    RESULTS["skill"] = sk

    # --- tidal-level validation (PDF2 style) --------------------------------
    t = (med_s["t"] - med_s["t"][0]) / 3600.0
    eta_mod = med_s["eta"][:, med_s["grid"].ny // 2, med_s["grid"].nx // 2]
    eta_obs = eta_mod * 1.02 + rng.normal(0, 0.04, eta_mod.shape)
    fig, ax = viz.new_ax((8, 4.4), "Modelled vs synthetic tidal-level reference (spring) - station T1",
                         "Time (h)", "Water level (m, CD)")
    ax.plot(t, eta_mod, color="#2a6f97", lw=2, label="UBDS model")
    ax.plot(t, eta_obs, "o", color="#d1495b", ms=3, label="Synthetic reference")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/B_tidal_level.png"),
               "Modelled tidal level (spring tide) against a synthetic reference series derived from the "
               "model solution with an imposed bias and noise. Not an independent measurement.", sec)
    td = pd.DataFrame({"time_h": t, "model_m": eta_mod, "synthetic_ref_m": eta_obs})
    H.write_csv(td, "B_tidal_level.csv", "Tidal-level validation time series.", sec)

    # --- tidal flow field: flood & ebb snapshots (speed + vectors) ----------
    for hyd, lab in [(med_s, "spring")]:
        f, g, X, Y = drive_fields(hyd, "medium")
        spd_all = np.hypot(f["u"], f["v"])
        vmean = f["v"].mean(axis=(1, 2))
        i_flood = int(np.argmax(vmean)); i_ebb = int(np.argmin(vmean))
        for idx, phase in [(i_flood, "flood"), (i_ebb, "ebb")]:
            sp = np.hypot(f["u"][idx], f["v"][idx])
            fig, ax = viz.new_ax((7.4, 6.6), f"Tidal flow field - {phase} ({lab})",
                                 "Easting (m)", "Northing (m)")
            pc = ax.pcolormesh(X, Y, sp, cmap=viz.VEL_CMAP, shading="auto",
                               vmin=0, vmax=0.8)
            sk2 = max(1, g.nx // 26)
            ax.quiver(X[::sk2, ::sk2], Y[::sk2, ::sk2], f["u"][idx][::sk2, ::sk2],
                      f["v"][idx][::sk2, ::sk2], color="#1b2a4a", scale=10, width=0.0028)
            ax.contourf(X, Y, (g.h <= 2.01).astype(float), levels=[0.5, 1.5], colors=["#d9c9a8"])
            cb = fig.colorbar(pc, ax=ax, shrink=0.85); cb.set_label("Current speed (m/s)")
            _outfall_marker(ax, C.MEDIUM_FIELD["outfall_xy"])
            ax.set_aspect("equal")
            H.register("figure", viz.save(fig, f"{H.FIG_DIR}/B_flowfield_{phase}.png"),
                       f"Depth-averaged tidal flow field at {phase} ({lab} tide).", sec)
    return dict(med_n=med_n, med_s=med_s, far_n=far_n, far_s=far_s)


# ===========================================================================
#  TRANSPORT helpers
# ===========================================================================
def layer_for_height(layers: tr.LayerScheme, target_h, mean_depth):
    edges = np.concatenate([[0], np.cumsum(layers.fractions)]) * mean_depth
    mids = 0.5 * (edges[:-1] + edges[1:])
    return int(np.argmin(np.abs(mids - target_h)))


def run_transport_case(field, hyd, n_layers, brine, sc, label, tide,
                       snapshots=False):
    """Run one transport simulation; returns result dict + grid."""
    f, g, X, Y = drive_fields(hyd, field)
    layers = tr.LayerScheme.from_percent(C.SIGMA_LAYERS[n_layers])
    Sa = C.AMBIENT["background_salinity_psu"]
    S0, T0 = brine
    model = tr.TransportModel(g, layers, s_bg=Sa,
                              t_bg=C.AMBIENT["background_temperature_C"],
                              Dh=C.MODEL["horizontal_dispersion_m2s"],
                              Kz=C.MODEL["vertical_diffusivity_m2s"],
                              sink_coef=C.MODEL["gravity_current_coef"])
    # near-field impact salinity for this case
    amb = nf.Ambient(salinity=Sa, temperature=C.AMBIENT["background_temperature_C"],
                     depth=C.AMBIENT["outfall_depth_m"],
                     current=dict((x["label"], x["velocity"]) for x in C.TIDAL_STATES)[tide])
    d = nf.Discharge(flow_m3hr=sc["flow_m3hr"] / sc["n_ports"],
                     diameter_m=C.DIFFUSER["port_diameter_m"], salinity=S0, temperature=T0,
                     z0=C.DIFFUSER["port_height_m"], theta_deg=C.DIFFUSER["port_angle_deg"])
    rnf = nf.simulate(d, amb, n_ports=sc["n_ports"], port_spacing=C.DIFFUSER["port_spacing_m"])
    ox, oy = (C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD)["outfall_xy"]
    diff = df.Diffuser(n_ports_installed=sc["n_ports"], port_spacing=C.DIFFUSER["port_spacing_m"],
                       x0=ox, y0=oy, bearing_deg=C.DIFFUSER["diffuser_bearing_deg"])
    ports = diff.port_positions(sc["n_ports"])
    # near-field footprint scales with discharge (more flow / more active ports
    # -> larger diluted-plume cross-section delivered to the far field)
    footprint = max(g.dx * 1.2, 2.9 * np.sqrt(sc["flow_m3hr"]))
    model.add_line_source(ports, sc["flow_m3hr"], S0, Sa,
                          footprint_m=footprint,
                          impact_salinity=rnf.impact_salinity)
    snaps = list(C.SNAPSHOT_PHASES.values()) if snapshots else None
    out = tr.run_transport(model, f, n_cycles=C.MODEL["sim_cycles"],
                           cycle_seconds=C.MODEL["tidal_period_s"], snapshots_at=snaps)
    out["grid"] = g; out["X"] = X; out["Y"] = Y; out["layers"] = layers
    out["nf"] = rnf
    return out


BRINE_CAVERN = (C.BRINE["cavern_salinity_psu"],
                C.AMBIENT["background_temperature_C"] + C.BRINE["cavern_excess_temp_C"])
BRINE_RO = (C.BRINE["ro_salinity_psu"],
            C.AMBIENT["background_temperature_C"] + C.BRINE["ro_excess_temp_C"])
PHASE_NAME = {v: k for k, v in C.SNAPSHOT_PHASES.items()}


def profile_at(out, phase_key):
    """Salinity and layer mid-heights through the diffuser column at one phase."""
    g = out["grid"]; layers = out["layers"]
    j, i = nearest_cell(g, C.MEDIUM_FIELD["outfall_xy"])
    S = out["snapshots"][round(phase_key, 3)]["S"]
    depth = float(g.h[j, i])
    edges = np.concatenate([[0], np.cumsum(layers.fractions)]) * depth
    return S[:, j, i], 0.5 * (edges[:-1] + edges[1:])


# Okabe-Ito: distinguishable in print, on a projector and to colour-blind viewers.
PHASE_COLOURS = {"high_water": "#0072B2", "mid_ebb": "#009E73",
                 "low_water": "#D55E00", "mid_flood": "#CC79A7"}


def vertical_profile_plot(out, phase_key, title, fname, sec, xlim=None):
    """Vertical salinity profile through the diffuser column at one tidal phase.

    Each phase used to autoscale to its own salinity range, so four genuinely
    different profiles rendered as the same shape and could not be compared.  The
    caller now passes one salinity axis shared by every phase.
    """
    sal, mids = profile_at(out, phase_key)
    fig, ax = viz.new_ax((5.2, 6.2), title, "Salinity (psu)", "Height above seabed (m)")
    ax.plot(sal, mids, "-o", lw=2.6, ms=6, color="#0072B2", mec="white", mew=0.8)
    ax.axvline(C.THRESHOLDS["salinity_threshold_psu"], color="#000000", ls="--", lw=1.8,
               label=f"{C.THRESHOLDS['salinity_threshold_psu']} psu threshold")
    ax.axvline(C.AMBIENT["background_salinity_psu"], color="#555555", ls=":", lw=1.8,
               label=f"{C.AMBIENT['background_salinity_psu']} psu background")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0, 6)
    ax.annotate(f"seabed {sal[0]:.2f} psu", xy=(0.97, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=10, fontweight="bold", color=viz.INK)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(0.97, 0.89))
    ax.grid(True, color=viz.GRIDC, alpha=0.6)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/{fname}"), title, sec)


def vertical_profiles_plot(out, title, fname, sec):
    """One profile plus the spread across tidal phases.

    Drawn as four separate panels these profiles were indistinguishable: each was
    autoscaled to its own salinity range, and the bed cell at the diffuser column is
    source-controlled, so every phase reads the same seabed value.  Overlaying four
    near-coincident curves only traded four identical panels for one muddled figure.
    A single mean profile with a min-max band across the phases shows both the shape
    and how little the tide changes it.
    """
    names = list(C.SNAPSHOT_PHASES)
    sals = np.array([profile_at(out, C.SNAPSHOT_PHASES[n])[0] for n in names])
    mids = profile_at(out, C.SNAPSHOT_PHASES[names[0]])[1]
    mean, lo, hi = sals.mean(axis=0), sals.min(axis=0), sals.max(axis=0)
    bed = sals[:, 0]

    fig, ax = viz.new_ax((7.6, 4.0), title, "Salinity (psu)", "Height above seabed (m)")
    ax.fill_betweenx(mids, lo, hi, color="#0072B2", alpha=0.22, lw=0,
                     label="spread across the four phases")
    ax.plot(mean, mids, "-o", lw=3.0, ms=8, color="#0072B2", mec="white", mew=1.0,
            label="mean profile")
    ax.axvline(C.THRESHOLDS["salinity_threshold_psu"], color="#000000", ls="--", lw=2.0,
               label=f"{C.THRESHOLDS['salinity_threshold_psu']} psu threshold")
    ax.axvline(C.AMBIENT["background_salinity_psu"], color="#555555", ls=":", lw=2.0,
               label=f"{C.AMBIENT['background_salinity_psu']} psu background")
    # The plume is bottom-trapped: above ~5 m the column sits at the background salinity,
    # so a 0-14 m axis squeezes the entire signal into the bottom sliver.
    ax.set_ylim(0, 6)
    ax.set_title("Vertical salinity profile at the diffuser column\n"
                 "(neap, saturated cavern brine)",
                 fontsize=14, color=viz.INK, fontweight="bold")
    ax.set_xlabel("Salinity (psu)", fontsize=13)
    ax.set_ylabel("Height above seabed (m)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10, loc="upper right", framealpha=1.0, borderpad=0.6)
    _spread = float(np.abs(hi - lo).max())
    ax.annotate(f"Seabed salinity is {bed.min():.2f} psu at every phase; the widest gap between "
                f"phases anywhere in the column is {_spread:.2f} psu.",
                xy=(0.5, -0.30), xycoords="axes fraction", ha="center", va="top",
                fontsize=11, color=viz.INK)
    ax.grid(True, color=viz.GRIDC, alpha=0.6)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/{fname}"), title, sec)
    # The four per-phase panels this figure replaces were the only home for the
    # individual phase curves; the mean-and-band plot cannot reproduce them.  Write
    # them out so no number lives only inside a picture.
    prof = pd.DataFrame({"height_above_bed_m": mids})
    for n in names:
        prof[f"salinity_{n}_psu"] = profile_at(out, C.SNAPSHOT_PHASES[n])[0]
    prof["salinity_mean_psu"] = mean
    H.write_csv(prof.round(4), "C_vertical_profiles.csv",
                "Vertical salinity profile at the diffuser column for each neap tidal phase "
                "(the data behind the mean-and-spread figure).", sec)


# ===========================================================================
#  PART C - MEDIUM-FIELD DISPERSION
# ===========================================================================
def part_C_medium(hydro_all):
    print("[C] medium-field dispersion ...")
    sec = "Medium-field dispersion"
    med_n, med_s = hydro_all["med_n"], hydro_all["med_s"]
    ox = C.MEDIUM_FIELD["outfall_xy"]
    mz = C.THRESHOLDS["mixing_zone_radius_m"]
    sc3 = C.SCENARIOS[-1]
    gm, Xm, Ym, _ = fine_grid("medium"); set_basemap(gm, Xm, Ym)
    MZOOM = ([ox[0] - 980, ox[0] + 520], [ox[1] - 700, ox[1] + 700])

    # --- sigma-layer schematic ---------------------------------------------
    fig, ax = viz.new_ax((7, 5), "Sigma-layer discretisation of the water column (9 layers)",
                         "Relative position across domain", "Depth below surface (m)")
    depths = np.linspace(6, 20, 40)
    fr = tr.LayerScheme.from_percent(C.SIGMA_LAYERS[9]).fractions
    cum = np.concatenate([[0], np.cumsum(fr)])
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(fr)))
    xs = np.linspace(0, 1, len(depths))
    # LayerScheme.fractions is ordered bed -> surface, so L1 is the thin bed cell.
    # The y axis is depth below surface, hence layer k spans (1 - cum[k+1]) to
    # (1 - cum[k]).  Stacking cum[k] downward from the surface drew the column
    # upside down, putting the 1% bed layer at the free surface.
    for k in range(len(fr)):
        lo = (1.0 - cum[k + 1]) * depths; hi = (1.0 - cum[k]) * depths
        if k == 0:
            lbl = f"L1 (bed, {fr[0] * 100:.0f}%)"
        elif k == len(fr) - 1:
            lbl = f"L{len(fr)} (surface, {fr[-1] * 100:.0f}%)"
        else:
            lbl = None
        ax.fill_between(xs, lo, hi, color=cols[k], alpha=0.8, label=lbl)
    ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/C_sigma_layers.png"),
               "Terrain-following sigma layers versus water depth (9-layer scheme). Layers are "
               "numbered from the seabed upward, so L1 is the thin bed cell that resolves the "
               "dense bottom plume and L9 is the thick surface cell.", sec)

    # --- layer-resolution sensitivity (4 / 9 / 14) at full flow, neap -------
    sens = {}
    for nl in (4, 9, 14):
        out = run_transport_case("medium", med_n, nl, BRINE_CAVERN, sc3, "spring_neap", "neap")
        sens[nl] = out
        plot_salinity_field(out["X"], out["Y"], out["env"][0],
            f"Maximum seabed salinity - {nl}-layer model (full flow, neap, saturated cavern brine)",
            f"C_layersens_{nl}.png", sec, ox, mz=mz)
    RESULTS["layer_sensitivity"] = {nl: float(sens[nl]["env"][0].max()) for nl in sens}

    # --- 9-layer reference run with snapshots (neap) ------------------------
    out9n = run_transport_case("medium", med_n, 9, BRINE_CAVERN, sc3, "spring_neap",
                               "neap", snapshots=True)
    g = out9n["grid"]; X, Y = out9n["X"], out9n["Y"]
    mean_depth = float(g.h[nearest_cell(g, ox)])
    layers = out9n["layers"]
    # envelopes at heights above bed - ONE multi-panel figure showing the
    # vertical decay of the dense plume (avoids 4 near-identical standalone maps)
    hpanels = []
    for hgt in C.SALINITY_HEIGHTS_M:
        k = layer_for_height(layers, hgt, mean_depth)
        hpanels.append({"S": out9n["env"][k], "X": X, "Y": Y, "grid": g,
                        "label": f"{hgt:.0f} m above bed"})
    plot_salinity_panels(hpanels,
        "Maximum salinity vs height above the seabed - neap, full flow, saturated cavern brine "
        "(dense plume is bottom-trapped and decays upward)",
        "C_env_heights_panel.png", sec, ox, mz=mz, zoom_box=MZOOM, ncols=2)
    # snapshots: salinity + current vectors (bottom layer)
    fkn, _, _, _ = drive_fields(med_n, "medium")
    for name, ph in C.SNAPSHOT_PHASES.items():
        snp = out9n["snapshots"][round(ph, 3)]
        plot_salinity_field(X, Y, snp["S"][0],
            f"Seabed salinity & current vectors at {name.replace('_',' ')} - neap (saturated cavern brine)",
            f"C_snap_neap_{name}.png", sec, ox, mz=mz,
            vectors=(snp["u"], snp["v"]), phase=snp.get("phase", ph),
            zoom_box=MZOOM)

    # One salinity axis shared by every phase, so the four panels are comparable.
    _allsal = np.concatenate([profile_at(out9n, ph)[0] for ph in C.SNAPSHOT_PHASES.values()])
    _lo = min(_allsal.min(), C.AMBIENT["background_salinity_psu"])
    _pad = 0.06 * (_allsal.max() - _lo) or 0.5
    _vxlim = (_lo - _pad, _allsal.max() + _pad)
    for name, ph in C.SNAPSHOT_PHASES.items():
        vertical_profile_plot(out9n, ph,
            f"Vertical salinity profile at {name.replace('_',' ')} - neap (saturated cavern brine)",
            f"C_vprof_neap_{name}.png", sec, xlim=_vxlim)

    vertical_profiles_plot(out9n,
        "Vertical salinity profiles at the diffuser column - all tidal phases "
        "(neap, saturated cavern brine)", "C_vprof_neap.png", sec)

    # --- spring run with snapshots ------------------------------------------
    out9s = run_transport_case("medium", med_s, 9, BRINE_CAVERN, sc3, "spring_neap",
                               "spring", snapshots=True)
    for name, ph in C.SNAPSHOT_PHASES.items():
        snp = out9s["snapshots"][round(ph, 3)]
        plot_salinity_field(out9s["X"], out9s["Y"], snp["S"][0],
            f"Seabed salinity & current vectors at {name.replace('_',' ')} - spring (saturated cavern brine)",
            f"C_snap_spring_{name}.png", sec, ox, mz=mz,
            vectors=(snp["u"], snp["v"]), phase=snp.get("phase", ph),
            zoom_box=MZOOM)

    # --- spring-neap maximum envelope ---------------------------------------
    env_sn = np.maximum(out9n["env"][0], out9s["env"][0])
    plot_salinity_field(X, Y, env_sn,
        "Maximum seabed salinity over a full spring-neap cycle (full flow, saturated cavern brine)",
        "C_env_springneap.png", sec, ox, mz=mz)

    RESULTS["medium"] = dict(env_bed_neap=out9n["env"][0], env_bed_spring=out9s["env"][0],
                             env_sn=env_sn, X=X, Y=Y, mean_depth=mean_depth,
                             nf_impact=out9n["nf"].impact_salinity)
    RESULTS["_med_grid_h"] = g.h
    return dict(out9n=out9n, out9s=out9s, sens=sens)


# ===========================================================================
#  PART D - FAR-FIELD DISPERSION
# ===========================================================================
def part_D_far(hydro_all):
    print("[D] far-field dispersion ...")
    sec = "Far-field dispersion"
    far_n, far_s = hydro_all["far_n"], hydro_all["far_s"]
    ox = C.FAR_FIELD["outfall_xy"]
    mz = C.THRESHOLDS["mixing_zone_radius_m"]
    gf, Xf, Yf, _ = fine_grid("far"); set_basemap(gf, Xf, Yf)
    far_levels = np.round(np.concatenate([
        np.arange(36.5, 38.01, 0.25), np.arange(38.5, 41.01, 0.5)]), 2)

    far_zoom = ([ox[0]-1150, ox[0]+2600], [ox[1]-2200, ox[1]+2200])
    far_runs = {}
    fpanels = []
    for sc in C.SCENARIOS:
        out = run_transport_case("far", far_n, 4, BRINE_RO, sc, "spring_neap", "neap")
        far_runs[sc["id"]] = out
        fpanels.append({"S": out["env"][0], "X": out["X"], "Y": out["Y"], "grid": out["grid"],
                        "label": f"{sc['id']}: {sc['flow_m3hr']} m3/hr "
                                 f"({sc['n_ports']} port{'' if sc['n_ports'] == 1 else 's'})"})
    # ONE 3-panel scenario comparison (avoids 3 near-identical standalone maps)
    plot_salinity_panels(fpanels,
        "Far-field maximum seabed salinity by discharge scenario - neap tide (RO brine)",
        "D_far_neap_scenarios.png", sec, ox, mz=mz, levels=far_levels,
        zoom_box=far_zoom, ncols=3)
    # spring full flow
    out_s = run_transport_case("far", far_s, 4, BRINE_RO, C.SCENARIOS[-1], "spring_neap", "spring")
    plot_salinity_field(out_s["X"], out_s["Y"], out_s["env"][0],
        "Far-field maximum seabed salinity - spring, full flow (RO brine)",
        "D_far_spring_S3.png", sec, ox, mz=mz, levels=far_levels,
        zoom_box=([ox[0]-1150, ox[0]+2600], [ox[1]-2200, ox[1]+2200]))
    # spring-neap envelope
    env_sn = np.maximum(far_runs["S3"]["env"][0], out_s["env"][0])
    plot_salinity_field(out_s["X"], out_s["Y"], env_sn,
        "Far-field maximum seabed salinity - spring-neap envelope (full flow, RO brine)",
        "D_far_springneap.png", sec, ox, mz=mz, levels=far_levels,
        zoom_box=([ox[0]-1150, ox[0]+2600], [ox[1]-2200, ox[1]+2200]))
    # cavern worst case
    out_cav = run_transport_case("far", far_n, 4, BRINE_CAVERN, C.SCENARIOS[-1], "spring_neap", "neap")
    plot_salinity_field(out_cav["X"], out_cav["Y"], out_cav["env"][0],
        "Far-field maximum seabed salinity - phase-2 saturated cavern brine (worst case)",
        "D_far_cavern.png", sec, ox, mz=mz,
        zoom_box=([ox[0]-1150, ox[0]+2600], [ox[1]-2200, ox[1]+2200]))

    # --- influence-area & diffusion-distance table (PDF2 Table 2 style) -----
    g = far_runs["S3"]["grid"]; Sa = C.AMBIENT["background_salinity_psu"]
    rows = []
    for sc in C.SCENARIOS:
        env = far_runs[sc["id"]]["env"][0]
        areas = metrics.increment_areas(env, Sa, g.dx, g.dy)
        par, tra = metrics.diffusion_distance(env, far_runs[sc["id"]]["X"][0],
                    far_runs[sc["id"]]["Y"][:, 0], ox, Sa, g.dx, g.dy, 0.5)
        mzr = metrics.mixing_zone_radius(env, far_runs[sc["id"]]["X"][0],
                    far_runs[sc["id"]]["Y"][:, 0], ox, C.THRESHOLDS["salinity_threshold_psu"])
        rows.append(dict(scenario=sc["id"], flow_m3hr=sc["flow_m3hr"],
                         area_0p5psu_km2=round(areas[0.5], 4),
                         area_1p0psu_km2=round(areas[1.0], 4),
                         area_1p5psu_km2=round(areas[1.5], 4),
                         diff_dist_parallel_m=round(par, 0),
                         diff_dist_transverse_m=round(tra, 0),
                         mixing_zone_radius_m=round(mzr, 0),
                         max_seabed_salinity_psu=round(float(env.max()), 2)))
    dft = pd.DataFrame(rows)
    H.write_csv(dft, "D_influence_area_diffusion.csv",
                "Influence areas of salinity increments and diffusion distances per flow scenario (far field).", sec)
    RESULTS["far_table"] = dft
    RESULTS["far"] = dict(env_S3=far_runs["S3"]["env"][0], X=far_runs["S3"]["X"],
                          Y=far_runs["S3"]["Y"])
    return far_runs


# ===========================================================================
#  PART E - DESIGN COMPARISON, PROFILES, CURRENT CURVES, LAYOUT
# ===========================================================================
def part_E_design(med):
    print("[E] design comparison & profiles ...")
    sec = "Design comparison & optimisation"
    out9n = med["out9n"]; X, Y = out9n["X"], out9n["Y"]; g = out9n["grid"]
    ox = C.MEDIUM_FIELD["outfall_xy"]; Sa = C.AMBIENT["background_salinity_psu"]

    # salinity profile along principal transect through the diffuser (PDF2 8d)
    j0, i0 = nearest_cell(g, ox)
    xs = X[0]; sal_line = out9n["env"][0][j0, :]
    fig, ax = viz.new_ax((8, 4.6),
                         "Seabed salinity through the diffuser (cross-shore transect, saturated cavern brine)",
                         "Easting (m)", "Maximum seabed salinity (psu)")
    ax.plot(xs, sal_line, color="#d1495b", lw=2.4)
    ax.axhline(Sa, color="#7b5ea7", ls=":", label="Background")
    ax.axhline(C.THRESHOLDS["salinity_threshold_psu"], color="#3a9278", ls="--", label="Threshold")
    ax.axvline(ox[0], color="#2a6f97", ls="-.", alpha=0.6, label="Diffuser")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/E_salinity_transect.png"),
               "Maximum seabed salinity along a transect through the diffuser "
               "(neap, full flow, saturated cavern brine).", sec)
    pd.DataFrame({"easting_m": xs, "max_seabed_salinity_psu": sal_line}).to_csv(
        f"{H.CSV_DIR}/E_salinity_transect.csv", index=False)
    H.register("csv", f"{H.CSV_DIR}/E_salinity_transect.csv",
               "Seabed salinity transect data.", sec)

    # max current-speed curve near intake & outlet (PDF2 Fig 9)
    fkn, gg, Xx, Yy = drive_fields(med["out9n"] and RESULTS and med["out9n"]["grid"] and med, "medium") if False else (None,)*4
    # build directly from cached spring hydro
    med_s = RESULTS.get("_dummy")  # not used
    # use the spring hydro fields stored in part B via re-run cache
    hyd = run_hydro_state("medium", "spring")
    f, g2, X2, Y2 = drive_fields(hyd, "medium")
    spd = np.hypot(f["u"], f["v"]).max(axis=0)
    j0, i0 = nearest_cell(g2, ox)
    offshore = X2[0]
    fig, ax = viz.new_ax((8, 4.6), "Maximum tidal current speed along the outfall corridor (spring)",
                         "Distance offshore (Easting, m)", "Max current speed (m/s)")
    ax.plot(offshore, spd[j0, :], color="#2a6f97", lw=2.4, label="Along diffuser transect")
    ax.axvline(ox[0], color="#d1495b", ls="--", label="Diffuser")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/E_current_curve.png"),
               "Maximum current speed along the outfall corridor (spring tide).", sec)

    # site development / constraints map
    gm, Xm, Ym, _ = fine_grid("medium")
    fig, ax = viz.new_ax((7.4, 6.6), "Marine development & environmental constraints",
                         "Easting (m)", "Northing (m)")
    ax.contourf(Xm, Ym, (gm.h <= 2.01).astype(float), levels=[0.5, 1.5], colors=["#d9c9a8"])
    ax.add_patch(plt.Circle((ox[0]+700, ox[1]+650), 250, color="#3a9278", alpha=0.4, label="Aquaculture"))
    ax.add_patch(plt.Circle((ox[0]-500, ox[1]-700), 200, color="#7b5ea7", alpha=0.4, label="Bathing water"))
    ax.add_patch(plt.Rectangle((ox[0]-150, ox[1]-1100), 300, 180, color="#e09f3e", alpha=0.5, label="Port"))
    _outfall_marker(ax, ox, C.THRESHOLDS["mixing_zone_radius_m"])
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/E_constraints.png"),
               "Distribution of marine development and environmental constraints relative to the mixing zone.", sec)

    # optimised layout schematic
    fig, ax = viz.new_ax((7.4, 6.6), "Recommended optimised diffuser layout",
                         "Easting (m)", "Northing (m)")
    ax.contourf(Xm, Ym, (gm.h <= 2.01).astype(float), levels=[0.5, 1.5], colors=["#d9c9a8"])
    npg = C.DIFFUSER["n_ports_installed"]
    diff = df.Diffuser(n_ports_installed=npg, port_spacing=C.DIFFUSER["port_spacing_m"],
                       x0=ox[0], y0=ox[1], bearing_deg=C.DIFFUSER["diffuser_bearing_deg"])
    ports = diff.port_positions(npg)
    ax.plot(ports[:, 0], ports[:, 1], "-", color="#1b2a4a", lw=3)
    ax.plot(ports[:, 0], ports[:, 1], "o", color="#e6c84f", mec="#9b2226", ms=10, label=f"Diffuser ports ({npg} x 6\")")
    ax.plot([ox[0]-260, ox[0]-260], [ox[1]-80, ox[1]+80], "s-", color="#2a6f97", label="Intake")
    _outfall_marker(ax, ox, C.THRESHOLDS["mixing_zone_radius_m"])
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal")
    ax.set_xlim(ox[0]-350, ox[0]+350); ax.set_ylim(ox[1]-300, ox[1]+300)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/E_optimised_layout.png"),
               f"Recommended optimised diffuser/intake layout (inclined {npg}-port diffuser).", sec)


# ===========================================================================
#  PART F - TABLES & IMPURITIES
# ===========================================================================
def part_F_tables():
    print("[F] tables & impurities ...")
    sec = "Inputs, parameters & water quality"
    # Layer-scheme table.  SIGMA_LAYERS is ordered bottom-first, so index the rows
    # from the SEABED: L1 is the bed cell in every scheme.  Aligning the schemes at
    # the surface instead (as this table used to) puts the 4-layer bed cell on the
    # same row as the 14-layer mid-column cell, so reading across a row compares
    # layers at different heights - the opposite of the point the table makes, which
    # is that resolution concentrates at the bed.
    rows = []
    maxn = 14
    for k in range(maxn):
        # Only L1 is the bed in every scheme; the surface cell differs per scheme
        # (L14, L9, L4), so do not label a single row "surface".
        rows.append(dict(layer=f"L{k+1}" + (" (bed)" if k == 0 else ""),
                         pct_14=C.SIGMA_LAYERS[14][k],
                         pct_9=(C.SIGMA_LAYERS[9][k] if k < 9 else None),
                         pct_4=(C.SIGMA_LAYERS[4][k] if k < 4 else None)))
    H.write_csv(pd.DataFrame(rows), "F_layer_scheme.csv",
                "Sigma-layer schemes: percentage of water-column depth per layer, indexed from the "
                "seabed (L1 is the bed cell in every scheme; each scheme's surface cell is its last "
                "numbered entry - L14, L9 and L4 respectively). A dash means the scheme has no such "
                "layer. The bed cell is 0.1% of depth with 14 layers, 1.0% with 9 and 15% with 4.",
                sec)

    # model parameter table
    par = [("Minimum time step", f"{C.MODEL['min_timestep_s']} s"),
           ("CFL number", C.MODEL["cfl_number"]),
           ("Manning coefficient n", f"{C.MODEL['manning_n']} ({C.MODEL['manning_range'][0]}-{C.MODEL['manning_range'][1]})"),
           ("Horizontal eddy viscosity", f"{C.MODEL['horizontal_eddy_viscosity_m2s']} m2/s"),
           ("Smagorinsky constant", C.MODEL["smagorinsky_const"]),
           ("Horizontal dispersion", f"{C.MODEL['horizontal_dispersion_m2s']} m2/s"),
           ("Vertical diffusivity", f"{C.MODEL['vertical_diffusivity_m2s']} m2/s"),
           ("Gravity-current coefficient", C.MODEL["gravity_current_coef"]),
           ("Background salinity", f"{C.AMBIENT['background_salinity_psu']} psu"),
           ("Background temperature", f"{C.AMBIENT['background_temperature_C']} C"),
           ("RO brine salinity", f"{C.BRINE['ro_salinity_psu']} psu"),
           ("Cavern brine salinity", f"{C.BRINE['cavern_salinity_psu']} psu"),
           ("Outfall depth", f"{C.AMBIENT['outfall_depth_m']} m"),
           # The near-field closure constants and the diffuser geometry belong in
           # the parameter table: without them the near-field results in this
           # report cannot be reproduced from it.
           ("Shear entrainment coefficient (jet)", nf.ALPHA_JET),
           ("Shear entrainment coefficient (plume)", round(nf.ALPHA_PLUME, 4)),
           ("Forced entrainment coefficient", nf.ALPHA_FORCED),
           ("Ports installed", C.DIFFUSER["n_ports_installed"]),
           ("Port diameter", f"{C.DIFFUSER['port_diameter_m']} m (6 inch)"),
           ("Port discharge angle", f"{C.DIFFUSER['port_angle_deg']} deg"),
           ("Port spacing", f"{C.DIFFUSER['port_spacing_m']} m"),
           ("Port height above bed", f"{C.DIFFUSER['port_height_m']} m")]
    H.write_csv(pd.DataFrame(par, columns=["Parameter", "Value"]),
                "F_model_parameters.csv", "Hydrodynamic and transport model parameters.", sec)

    # impurities / EQS compliance
    Sa = C.AMBIENT["background_salinity_psu"]
    # Use the MINIMUM near-field dilution over every brine, scenario and tidal
    # state.  Any other choice (a "representative" or mean value, still less the
    # study maximum) would understate the receiving-water concentration, since
    # concentration scales as 1/dilution.  This is the worst case the discharge
    # can present to the EQS.
    dil = float(RESULTS["nearfield"]["dilution"].min())
    _dil_src = RESULTS["nearfield"].loc[RESULTS["nearfield"]["dilution"].idxmin()]
    print(f"    impurity dilution = x{dil:.1f} (minimum: {_dil_src['brine']}, "
          f"{_dil_src['scenario']}, {_dil_src['tide']})")
    rows = []
    for param, (brine_c, sw_c, eqs) in C.BRINE["impurities_ugL"].items():
        comp = metrics.eqs_compliance(brine_c, dil, eqs, background_ugL=sw_c)
        rows.append(dict(parameter=param, brine_ugL=brine_c, seawater_ugL=sw_c,
                         # 1 dp: rounding to 0 dp printed x14 while the predicted
                         # concentrations were computed with x14.3, so the table
                         # did not reproduce its own numbers.
                         eqs_ugL=eqs, dilution=round(dil, 1),
                         predicted_ugL=round(comp["predicted_ugL"], 3),
                         margin_x=round(comp["margin"], 1),
                         compliant="Yes" if comp["compliant"] else "No"))
    dimp = pd.DataFrame(rows)
    H.write_csv(dimp, "F_impurities_eqs.csv",
                "Trace-impurity concentrations in the brine vs EQS after near-field dilution.", sec)
    RESULTS["impurities"] = dimp
    # impurity bar chart
    fig, ax = viz.new_ax((9, 4.8), "Predicted trace-impurity concentration vs EQS (after dilution)",
                         "Parameter", "Concentration (ug/L)")
    x = np.arange(len(dimp)); w = 0.38
    ax.bar(x - w/2, dimp.predicted_ugL, w, color="#2a6f97", label="Predicted in receiving water")
    ax.bar(x + w/2, dimp.eqs_ugL, w, color="#d1495b", label="EQS")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(dimp.parameter, rotation=35, ha="right")
    ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/F_impurities.png"),
               "Predicted receiving-water trace-impurity concentrations vs EQS thresholds.", sec)


# ===========================================================================
#  PART G - PLOT EVERY TABULAR CSV
# ===========================================================================
def part_G_csv_plots():
    print("[G] plotting CSV datasets ...")
    sec = "Data plots"
    # near-field dilution vs velocity (RO + cavern)
    d = RESULTS["nearfield"]
    fig, ax = viz.new_ax((7.5, 5), "Seabed dilution vs tidal velocity (by brine & scenario)",
                         "Tidal velocity (m/s)", "Seabed dilution (S/S0)")
    # S3 duplicates S1 exactly (same 300 m3/hr per port), so drawing all six series
    # stacked two identical curves per brine and left the legend swatches
    # indistinguishable.  Show the four distinct cases; the note carries S3.
    styles = {"RO concentrate": "-o", "Saturated cavern brine": "--s"}
    cols = {("RO concentrate", "S1"): "#12355b", ("RO concentrate", "S2"): "#1b9aaa",
            ("Saturated cavern brine", "S1"): "#c1121f",
            ("Saturated cavern brine", "S2"): "#f08c00"}
    short = {"RO concentrate": "RO", "Saturated cavern brine": "Cavern"}
    for bn in ["RO concentrate", "Saturated cavern brine"]:
        for sid in ["S1", "S2"]:
            sub = d[(d.brine == bn) & (d.scenario == sid)].sort_values("velocity_ms")
            ax.plot(sub.velocity_ms, sub.dilution, styles[bn], color=cols[(bn, sid)],
                    lw=2.2, ms=6, label=f"{short[bn]} {sid}")
    # Keep the note clear of the legend: "best" placement puts the legend top-left,
    # which is where the annotation used to be drawn, leaving it unreadable.
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.annotate("S3 (1200 m3/hr, 4 ports) reproduces S1 exactly: both discharge 300 m3/hr per port",
                xy=(0.98, 0.02), xycoords="axes fraction", fontsize=8,
                ha="right", va="bottom", color=viz.INK)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/G_dilution_vs_velocity.png"),
               "Seabed dilution as a function of tidal velocity for each brine type and scenario. "
               "S1 and S3 coincide because dilution is set by the per-port flow (300 m3/hr in both), "
               "not the total.", sec)

    # increment area vs flow
    dft = RESULTS["far_table"]
    _areas = ["area_0p5psu_km2", "area_1p0psu_km2", "area_1p5psu_km2"]
    _all_zero = float(dft[_areas].max().max()) == 0.0
    _bg = C.AMBIENT["background_salinity_psu"]
    _inc = dft.max_seabed_salinity_psu - _bg
    # Plotting three all-zero area curves produced three coincident lines on the axis:
    # no information, and legend swatches that cannot be told apart.  Plot the quantity
    # that explains the zeros instead - the peak seabed increment against the contour
    # thresholds it never reaches.
    fig, ax = viz.new_ax((7.5, 5),
                         "Peak seabed salinity increment vs discharge flow "
                         "(why every influence area is 0.000 km2)",
                         "Brine discharge (m3/hr)", "Peak seabed increment above ambient (psu)")
    xs = np.arange(len(dft))
    bars = ax.bar(xs, _inc, width=0.55, color="#0072B2", edgecolor="#0b2545",
                  linewidth=1.2, zorder=3)
    # Values sit inside the bar tops.  Above them, the tallest (+0.39 psu) collided
    # with the 0.5 psu threshold line, which is the one line it must be seen to clear.
    for x, v, sid in zip(xs, _inc, dft.scenario):
        ax.annotate(f"+{v:.2f} psu", (x, v), xytext=(0, -8), textcoords="offset points",
                    ha="center", va="top", fontsize=11, fontweight="bold",
                    color="white", zorder=4)
    # Distinct hue AND dash pattern per threshold: three dashed lines in similar blues
    # and greens were not separable on a projector.
    for thr, col, ls in [(0.5, "#E69F00", "--"), (1.0, "#009E73", "-."), (1.5, "#C1121F", ":")]:
        ax.axhline(thr, color=col, ls=ls, lw=2.6, zorder=2,
                   label=f"{thr:.1f} psu contour - area 0.000 km2")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{sid}\n{int(f)} m3/hr" for sid, f in zip(dft.scenario, dft.flow_m3hr)])
    # Headroom above the 1.5 psu line for the legend; the note sits in the clear band
    # between the 0.5 and 1.0 psu contours.
    ax.set_ylim(0, 2.0)
    ax.legend(fontsize=9, loc="upper right", framealpha=1.0)
    if _all_zero:
        ax.annotate(f"On the {C.FAR_FIELD['dx_m']:.0f} m regional grid the seabed increment never "
                    f"reaches the lowest\n0.5 psu contour, so all three influence areas are exactly "
                    f"0.000 km2.\nThe near-field impact footprint does exceed it (see section 4).",
                    xy=(0.5, 0.36), xycoords="axes fraction", ha="center", va="center",
                    fontsize=9, color=viz.INK)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/G_area_vs_flow.png"),
               "Peak seabed salinity increment by discharge scenario, against the 0.5, 1.0 and "
               f"1.5 psu contour thresholds. The increment peaks at {_inc.max():.2f} psu, so every "
               "influence area is 0.000 km2; the zeros are grid-resolved, not absolute.", sec)

    # validation skill bar
    sk = RESULTS["skill"]
    fig, ax = viz.new_ax((8, 4.6), "Model skill vs the synthetic current reference, by station",
                         "Station", "Willmott index of agreement (d)")
    sub = sk[sk.tide == "spring"]
    ax.bar(sub.station, sub.willmott_d, color="#3a9278")
    ax.axhline(0.9, color="#d1495b", ls="--", label="d = 0.90")
    ax.set_ylim(0, 1.05); ax.legend()
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/G_skill.png"),
               "Index-of-agreement between the model and the synthetic current reference by station "
               "(spring). The reference is derived from the model solution, so this is not a field "
               "comparison.", sec)


# ===========================================================================
#  PART H - OUTFALL SITING & INTAKE RECIRCULATION (steady-current study)
# ===========================================================================
def _steady_fields(g, speed, bearing_deg, T):
    br = np.radians(bearing_deg)
    u = np.full((4, g.ny, g.nx), speed * np.cos(br))
    v = np.full((4, g.ny, g.nx), speed * np.sin(br))
    return {"u": u, "v": v, "t": np.linspace(0, T, 4)}


def _recirc_run(outfall_dist_m, current_label, speed, bearing, capture_ts=False, brine=None):
    """Steady-current brine run; returns salinity rise at the intake (ppt).

    Uses the coarse (50 m) medium grid - a steady cross-shore spreading study
    does not need the fine resolution, which keeps the case study fast.

    ``brine`` selects the discharge: the intake signal scales with the excess
    salinity the plume delivers to the bed, so the phase-2 cavern brine (which
    lands at ~51 psu, an excess of ~15 psu) produces a far larger rise than the
    RO concentrate (~37 psu, an excess of ~0.5 psu).  Both must be assessed.
    """
    fld = C.MEDIUM_FIELD
    g, X, Y = build_bathymetry(fld["Lx_m"], fld["Ly_m"], 50.0, 50.0, fld["outfall_xy"])
    Sa = C.AMBIENT["background_salinity_psu"]
    S0, T0 = brine if brine is not None else BRINE_RO
    shore_x = 0.10 * fld["Lx_m"]
    ox = shore_x + outfall_dist_m
    oy = fld["Ly_m"] * 0.5
    intake_x = shore_x + 300.0                      # intake 300 m from shore
    f = _steady_fields(g, speed, bearing, C.MODEL["tidal_period_s"])
    layers = tr.LayerScheme.from_percent(C.SIGMA_LAYERS[9])
    # coastal cross-shore dispersion is stronger than the open-water value used
    # for the tidal envelopes; use a representative recirculation dispersion
    model = tr.TransportModel(g, layers, s_bg=Sa, t_bg=C.AMBIENT["background_temperature_C"],
                              Dh=10.0, Kz=C.MODEL["vertical_diffusivity_m2s"],
                              sink_coef=C.MODEL["gravity_current_coef"])
    amb = nf.Ambient(salinity=Sa, temperature=C.AMBIENT["background_temperature_C"],
                     depth=C.AMBIENT["outfall_depth_m"], current=max(speed, 0.05))
    n_ports = C.SCENARIOS[-1]["n_ports"]
    d = nf.Discharge(flow_m3hr=C.SCENARIOS[-1]["flow_m3hr"] / n_ports,
                     diameter_m=C.DIFFUSER["port_diameter_m"],
                     salinity=S0, temperature=T0, z0=C.DIFFUSER["port_height_m"], theta_deg=C.DIFFUSER["port_angle_deg"])
    rnf = nf.simulate(d, amb, n_ports=n_ports, port_spacing=C.DIFFUSER["port_spacing_m"])
    model.add_line_source([(ox, oy)], C.SCENARIOS[-1]["flow_m3hr"], S0, Sa,
                          footprint_m=max(g.dx * 1.2, 60), impact_salinity=rnf.impact_salinity)
    jin, iin = nearest_cell(g, (intake_x, oy)); jou, iou = nearest_cell(g, (ox, oy))
    T = C.MODEL["tidal_period_s"]; dt = model.cfl_dt(umax=max(speed, 0.1))
    n = int(np.ceil(2.2 * T / dt))
    ts_t, ts_in, ts_ou = [], [], []
    for k in range(n):
        model.step(f["u"][0], f["v"][0], dt)
        if capture_ts and k % 8 == 0:
            ts_t.append(k * dt / 3600.0)
            ts_in.append(float(model.S[0, jin, iin] - Sa))
            ts_ou.append(float(model.S[0, jou, iou] - Sa))
    rise = model.env[0] - Sa
    # rise at the intake: sample a small window around the intake
    win = rise[max(0, jin-2):jin+3, max(0, iin-2):iin+3]
    return dict(rise_avg=float(win.mean()), rise_min=float(win.min()), rise_max=float(win.max()),
                rise_field=rise, X=X, Y=Y, g=g, ox=ox, oy=oy, intake_x=intake_x,
                ts=(np.array(ts_t), np.array(ts_in), np.array(ts_ou)) if capture_ts else None)


def part_H_recirculation():
    print("[H] outfall siting & intake recirculation ...")
    sec = "Outfall siting & recirculation"
    # 1400 m is included because the phase-2 cavern brine does not meet the
    # criterion at any of the originally tested distances.
    dists = [600, 800, 1000, 1200, 1400]
    currents = [("northward 0.25 m/s", 0.25, 90), ("southward 0.25 m/s", 0.25, 270),
                ("weak southward 0.05 m/s", 0.05, 270)]
    brines = [("RO concentrate", BRINE_RO),
              ("Saturated cavern brine", BRINE_CAVERN)]
    crit = C.THRESHOLDS.get("recirc_limit_ppt", 0.1)
    rows = []
    fields = {}
    for bl, bv in brines:
        for cl, sp, brg in currents:
            for dd in dists:
                cap = (cl == "weak southward 0.05 m/s" and dd in (600, 1200))
                r = _recirc_run(dd, cl, sp, brg, capture_ts=cap, brine=bv)
                rows.append(dict(brine=bl, current=cl, outfall_dist_m=dd,
                                 rise_avg_ppt=round(r["rise_avg"], 3),
                                 rise_min_ppt=round(r["rise_min"], 3),
                                 rise_max_ppt=round(r["rise_max"], 3),
                                 # The criterion is an upper bound, so it is the peak
                                 # cell in the intake window that has to clear it - not
                                 # the window mean, which this previously graded on.
                                 recirc_risk="Yes" if r["rise_max"] > crit else "No"))
                fields[(bl, cl, dd)] = r
    dfr = pd.DataFrame(rows)
    H.write_csv(dfr, "H_intake_recirculation.csv",
                "Salinity rise above ambient at the intake vs outfall distance from shore, by brine "
                f"and current condition ({crit} ppt recirculation criterion, applied to the peak cell).",
                sec)
    RESULTS["recirc"] = dfr
    _fail = dfr[dfr.recirc_risk == "Yes"]
    if len(_fail):
        print(f"    recirculation: {len(_fail)} of {len(dfr)} cases exceed {crit} ppt "
              f"(brines: {sorted(_fail.brine.unique())})")

    # Table 1A / 1B / 1C style separate CSVs, now carrying both brines
    for cl, tag in [("southward 0.25 m/s", "H_table1A_southward"),
                    ("weak southward 0.05 m/s", "H_table1B_weak"),
                    ("northward 0.25 m/s", "H_table1C_northward")]:
        sub = dfr[dfr.current == cl][["brine", "outfall_dist_m", "rise_min_ppt",
                                      "rise_max_ppt", "rise_avg_ppt", "recirc_risk"]]
        H.write_csv(sub, f"{tag}.csv",
                    f"Rise in salinity at intake above ambient - {cl}, both brines.", sec)

    # Fig 6: rise at intake vs outfall distance.  Plot the MAXIMUM cell rise in the
    # intake window, not the window mean: the 0.1 ppt criterion is an upper bound, so
    # the peak cell is the quantity that has to clear it.  This also keeps the figure
    # on the same basis as the per-current tables (1A/1B/1C), which report the maximum.
    fig, ax = viz.new_ax((7.6, 5), "Maximum salinity rise at intake vs outfall distance from shore",
                         "Outfall distance from shore (m)", "Maximum salinity rise at intake (ppt)")
    # Give every series its own colour.  Sharing one colour per current and relying
    # on solid-vs-dashed to separate the brines made the six legend swatches read as
    # three: cool hues are now the RO concentrate, warm hues the cavern brine.
    cmap = {("RO concentrate", "northward 0.25 m/s"): "#12355b",
            ("RO concentrate", "southward 0.25 m/s"): "#1b9aaa",
            ("RO concentrate", "weak southward 0.05 m/s"): "#5fbb63",
            ("Saturated cavern brine", "northward 0.25 m/s"): "#7b2cbf",
            ("Saturated cavern brine", "southward 0.25 m/s"): "#f08c00",
            ("Saturated cavern brine", "weak southward 0.05 m/s"): "#c1121f"}
    style = {"RO concentrate": "-o", "Saturated cavern brine": "--s"}
    for bl, _bv in brines:
        for cl, sp, brg in currents:
            sub = dfr[(dfr.current == cl) & (dfr.brine == bl)].sort_values("outfall_dist_m")
            # a zero rise cannot be drawn on a log axis; floor it at the plot bottom
            y = sub.rise_max_ppt.clip(lower=1e-4)
            ax.plot(sub.outfall_dist_m, y, style[bl], color=cmap[(bl, cl)], lw=2.0, ms=5,
                    label=f"{'RO' if bl.startswith('RO') else 'Cavern'} - {cl}")
    ax.set_yscale("log")
    ax.axhline(crit, color="#333333", ls=":", lw=1.8, label=f"{crit} ppt recirculation limit")
    # Headroom above the worst curve so the legend never sits on the data.
    ax.set_ylim(4e-5, 60)
    ax.legend(fontsize=7, ncol=2, loc="upper center", framealpha=0.95)
    # The clipped points are floored, not measured; say so rather than let the flat
    # 1e-4 tails read as predictions.
    ax.annotate("Flat tails at 1e-4 ppt are floored: those cases round to 0.000 ppt "
                "and cannot be drawn on a log axis.",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7,
                ha="left", va="bottom", color=viz.INK)
    H.register("figure", viz.save(fig, f"{H.FIG_DIR}/H_intake_rise_curve.png"),
               "Variation of the maximum salinity rise at the intake with outfall distance from shore, "
               "for the RO concentrate and the phase-2 saturated cavern brine (log scale; "
               f"{crit} ppt recirculation criterion). Values of zero are plotted at the axis floor.", sec)

    # Fig 4A/4B: salinity-RISE contours for near vs far outfall (weak current)
    rise_levels = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    for dd, tag in [(600, "near"), (1200, "far")]:
        r = fields[("RO concentrate", "weak southward 0.05 m/s", dd)]
        fig, ax = viz.new_ax((7.4, 6.4),
            f"Contours of salinity rise - outfall {dd} m from shore (weak southward, RO brine)",
            "Easting (m)", "Northing (m)")
        cf = ax.contourf(r["X"], r["Y"], r["rise_field"], levels=rise_levels,
                         cmap=viz.RISE_CMAP, extend="max")
        cb = fig.colorbar(cf, ax=ax, shrink=0.85); cb.set_label("Salinity rise (ppt)")
        _draw_basemap(ax, r["g"], r["X"], r["Y"])
        ax.plot(r["ox"], r["oy"], "*", ms=15, mfc="#e6c84f", mec="#9b2226", label="Outfall", zorder=8)
        ax.plot(r["intake_x"], r["oy"], "s", ms=11, mfc="#2a6f97", mec="#1b2a4a", label="Intake", zorder=8)
        ax.legend(loc="upper right"); ax.set_aspect("equal")
        ax.set_xlim(r["intake_x"]-400, r["ox"]+800); ax.set_ylim(r["oy"]-1100, r["oy"]+1100)
        H.register("figure", viz.save(fig, f"{H.FIG_DIR}/H_rise_contour_{tag}.png"),
                   f"Salinity-rise contours, outfall {dd} m from shore, weak southward current "
                   f"(RO concentrate).", sec)

    # Fig 5A/5B: time history of salinity rise at intake & outfall
    for dd, tag in [(600, "near"), (1200, "far")]:
        r = fields[("RO concentrate", "weak southward 0.05 m/s", dd)]
        t, sin_, sou_ = r["ts"]
        fig, ax = viz.new_ax((8, 4.6),
            f"Time history of salinity rise - outfall {dd} m from shore (weak current, RO brine)",
            "Time (h)", "Salinity rise above ambient (ppt)")
        ax.plot(t, sou_, color="#d1495b", lw=2.2, label="At outfall")
        ax.plot(t, sin_, color="#2a6f97", lw=2.2, label="At intake")
        ax.axhline(0.1, color="#7b5ea7", ls="--", label="0.1 ppt limit")
        ax.legend()
        H.register("figure", viz.save(fig, f"{H.FIG_DIR}/H_timehist_{tag}.png"),
                   f"Time history of salinity rise at intake and outfall, outfall {dd} m from shore "
                   f"(weak southward current, RO concentrate).", sec)
        pd.DataFrame({"time_h": t, "rise_intake_ppt": sin_, "rise_outfall_ppt": sou_}).to_csv(
            f"{H.CSV_DIR}/H_timehist_{tag}.csv", index=False)
        H.register("csv", f"{H.CSV_DIR}/H_timehist_{tag}.csv",
                   f"Time-history data, outfall {dd} m from shore (RO concentrate).", sec)


if __name__ == "__main__":
    t0 = time.time()
    dfres, traj, diff = part_A_nearfield()
    part_A2_angle_opt(diff)
    hydro_all = part_B_hydro()
    med = part_C_medium(hydro_all)
    part_D_far(hydro_all)
    part_E_design(med)
    part_H_recirculation()
    part_F_tables()
    part_G_csv_plots()
    H.save_manifest()
    with open("outputs/results.pkl", "wb") as f:
        pickle.dump(RESULTS, f)
    n_fig = sum(1 for a in H._ARTIFACTS if a["kind"] == "figure")
    n_csv = sum(1 for a in H._ARTIFACTS if a["kind"] == "csv")
    print(f"ALL PARTS done in {time.time()-t0:.0f}s; figures={n_fig} csvs={n_csv}")
