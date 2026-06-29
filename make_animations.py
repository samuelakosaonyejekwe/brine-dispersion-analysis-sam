"""
Animation outputs for the UBDS case study
=========================================

Generates animated GIFs (and a static 'filmstrip' montage for the report) of
the time-evolving brine dispersion and tidal current field over a tidal cycle.
ffmpeg is not required - frames are written with Matplotlib's PillowWriter.

    python3 make_animations.py

Outputs:
    outputs/animations/*.gif
    outputs/figures/ANIM_*_filmstrip.png   (embedded in case.docx)

Author: Akosa Samuel Onyejekwe (Independent Researcher)
"""
import os, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import case_inputs as C
from case_geometry import build_bathymetry, nearest_cell
from ubds import transport as tr, nearfield as nf, diffuser as df, viz
import run_helpers as H
import run_case_study as R

ANIM_DIR = "outputs/animations"
os.makedirs(ANIM_DIR, exist_ok=True)
if os.path.exists(H.MANIFEST):
    H._ARTIFACTS.extend(json.load(open(H.MANIFEST)))


def capture_transport_frames(field, tide, brine, sc, n_layers=9, n_frames=48):
    """Run a transport simulation and capture bottom-salinity + current frames
    over the final tidal cycle."""
    hyd = R.run_hydro_state(field, tide)
    f, g, X, Y = R.drive_fields(hyd, field)
    layers = tr.LayerScheme.from_percent(C.SIGMA_LAYERS[n_layers])
    Sa = C.AMBIENT["background_salinity_psu"]
    S0, T0 = brine
    model = tr.TransportModel(g, layers, s_bg=Sa, t_bg=C.AMBIENT["background_temperature_C"],
                              Dh=C.MODEL["horizontal_dispersion_m2s"], Kz=C.MODEL["vertical_diffusivity_m2s"],
                              sink_coef=C.MODEL["gravity_current_coef"])
    amb = nf.Ambient(salinity=Sa, temperature=C.AMBIENT["background_temperature_C"],
                     depth=C.AMBIENT["outfall_depth_m"],
                     current=dict((x["label"], x["velocity"]) for x in C.TIDAL_STATES)[tide])
    d = nf.Discharge(flow_m3hr=sc["flow_m3hr"]/sc["n_ports"], diameter_m=C.DIFFUSER["port_diameter_m"],
                     salinity=S0, temperature=T0, z0=C.DIFFUSER["port_height_m"], theta_deg=C.DIFFUSER["port_angle_deg"])
    rnf = nf.simulate(d, amb, n_ports=sc["n_ports"], port_spacing=C.DIFFUSER["port_spacing_m"])
    ox, oy = (C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD)["outfall_xy"]
    diff = df.Diffuser(n_ports_installed=sc["n_ports"], port_spacing=C.DIFFUSER["port_spacing_m"],
                       x0=ox, y0=oy, bearing_deg=C.DIFFUSER["diffuser_bearing_deg"])
    model.add_line_source(diff.port_positions(sc["n_ports"]), sc["flow_m3hr"], S0, Sa,
                          footprint_m=max(g.dx*1.2, 2.9*np.sqrt(sc["flow_m3hr"])),
                          impact_salinity=rnf.impact_salinity)

    T = C.MODEL["tidal_period_s"]
    u_seq, v_seq, t_seq = f["u"], f["v"], f["t"]
    tspan = t_seq[-1] - t_seq[0]
    umax = float(np.max(np.hypot(u_seq, v_seq)))
    dt = model.cfl_dt(umax=max(umax, 0.3))
    n_cycles = C.MODEL["sim_cycles"]
    total = n_cycles * T
    n = int(np.ceil(total / dt))
    cap_start = total - T            # capture last cycle
    cap_every = max(1, int((T / n_frames) / dt))
    frames = []
    for k in range(n):
        t = k * dt
        phase = (t % T) / T
        idx = min(int(np.searchsorted(t_seq - t_seq[0], phase * tspan)), len(t_seq) - 1)
        model.step(u_seq[idx], v_seq[idx], dt)
        if t >= cap_start and k % cap_every == 0:
            frames.append(dict(S=model.S[0].copy(), u=u_seq[idx].copy(),
                               v=v_seq[idx].copy(), phase=phase))
    return frames, g, X, Y, (ox, oy), rnf.impact_salinity


def animate_salinity(frames, g, X, Y, outfall, title, gifname, zoom=450):
    ox, oy = outfall
    levels = viz.nonlinear_salinity_levels(C.AMBIENT["background_salinity_psu"])
    fig, ax = viz.new_ax((7.2, 6.4), title, "Easting (m)", "Northing (m)")
    sk = max(1, g.nx // 24)
    mz = C.THRESHOLDS["mixing_zone_radius_m"]

    def draw(i):
        ax.clear()
        fr = frames[i]
        cf = ax.contourf(X, Y, fr["S"], levels=levels, cmap=viz.SAL_CMAP, extend="max")
        ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], fr["u"][::sk, ::sk], fr["v"][::sk, ::sk],
                  color="#1b2a4a", scale=11, width=0.003, alpha=0.85)
        R._draw_basemap(ax, g, X, Y)
        R._outfall_marker(ax, outfall, mz)
        R._tidal_clock(ax, fr["phase"], (float(fr["u"].mean()), float(fr["v"].mean())))
        ax.set_xlim(ox - zoom, ox + zoom); ax.set_ylim(oy - zoom, oy + zoom)
        ax.set_title(f"{title}\nt = {fr['phase']*12.42:.1f} h into cycle",
                     color=viz.INK, fontweight="bold")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        return []
    cf0 = ax.contourf(X, Y, frames[0]["S"], levels=levels, cmap=viz.SAL_CMAP, extend="max")
    cb = fig.colorbar(cf0, ax=ax, shrink=0.85); cb.set_label("Seabed salinity (psu)")
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    path = f"{ANIM_DIR}/{gifname}"
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    H.register("animation", path, title, "Animations")
    return path


def animate_current(field, tide, title, gifname):
    hyd = R.run_hydro_state(field, tide)
    f, g, X, Y = R.drive_fields(hyd, field)
    ox, oy = (C.MEDIUM_FIELD if field == "medium" else C.FAR_FIELD)["outfall_xy"]
    nt = f["u"].shape[0]
    sk = max(1, g.nx // 26)
    fig, ax = viz.new_ax((7.2, 6.4), title, "Easting (m)", "Northing (m)")
    sp0 = np.hypot(f["u"][0], f["v"][0])
    pc = ax.pcolormesh(X, Y, sp0, cmap=viz.VEL_CMAP, vmin=0, vmax=0.8, shading="auto")
    cb = fig.colorbar(pc, ax=ax, shrink=0.85); cb.set_label("Current speed (m/s)")

    def draw(i):
        ax.clear()
        sp = np.hypot(f["u"][i], f["v"][i])
        ax.pcolormesh(X, Y, sp, cmap=viz.VEL_CMAP, vmin=0, vmax=0.8, shading="auto")
        ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], f["u"][i][::sk, ::sk], f["v"][i][::sk, ::sk],
                  color="#1b2a4a", scale=9, width=0.003)
        ax.contourf(X, Y, (g.h <= 2.01).astype(float), levels=[0.5, 1.5], colors=["#d9c9a8"])
        R._outfall_marker(ax, (ox, oy))
        ax.set_title(f"{title}\nframe {i+1}/{nt}", color=viz.INK, fontweight="bold")
        ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)"); ax.set_aspect("equal")
        return []
    anim = FuncAnimation(fig, draw, frames=min(nt, 36), blit=False)
    path = f"{ANIM_DIR}/{gifname}"
    anim.save(path, writer=PillowWriter(fps=7))
    plt.close(fig)
    H.register("animation", path, title, "Animations")
    return path


def filmstrip(frames, g, X, Y, outfall, title, fname, zoom=450, n=6):
    ox, oy = outfall
    levels = viz.nonlinear_salinity_levels(C.AMBIENT["background_salinity_psu"])
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8), layout="constrained")
    fig.suptitle(title, color=viz.INK, fontweight="bold")
    sk = max(1, g.nx // 20)
    for ax, ii in zip(axes.ravel(), idxs):
        fr = frames[ii]
        cf = ax.contourf(X, Y, fr["S"], levels=levels, cmap=viz.SAL_CMAP, extend="max")
        ax.quiver(X[::sk, ::sk], Y[::sk, ::sk], fr["u"][::sk, ::sk], fr["v"][::sk, ::sk],
                  color="#1b2a4a", scale=12, width=0.004, alpha=0.8)
        R._outfall_marker(ax, outfall, C.THRESHOLDS["mixing_zone_radius_m"])
        ax.set_xlim(ox - zoom, ox + zoom); ax.set_ylim(oy - zoom, oy + zoom)
        ax.set_title(f"t = {fr['phase']*12.42:.1f} h", fontsize=9, color=viz.INK)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
    path = f"{H.FIG_DIR}/{fname}"
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    H.register("figure", path, title + " (filmstrip of the animation).", "Animations")
    return path


if __name__ == "__main__":
    print("animating medium-field brine dispersion (spring) ...")
    frames, g, X, Y, of, imp = capture_transport_frames(
        "medium", "spring", R.BRINE_CAVERN, C.SCENARIOS[-1], n_layers=9, n_frames=44)
    animate_salinity(frames, g, X, Y, of,
                     "Brine dispersion over a tidal cycle (full flow, spring)",
                     "ANIM_brine_dispersion_spring.gif")
    filmstrip(frames, g, X, Y, of,
              "Brine dispersion over a tidal cycle (full flow, spring)",
              "ANIM_brine_dispersion_filmstrip.png")
    print("animating neap brine dispersion ...")
    frn, gn, Xn, Yn, ofn, _ = capture_transport_frames(
        "medium", "neap", R.BRINE_CAVERN, C.SCENARIOS[-1], n_layers=9, n_frames=44)
    animate_salinity(frn, gn, Xn, Yn, ofn,
                     "Brine dispersion over a tidal cycle (full flow, neap)",
                     "ANIM_brine_dispersion_neap.gif")
    print("animating tidal current field ...")
    animate_current("medium", "spring", "Tidal current field over a cycle (spring)",
                    "ANIM_current_field_spring.gif")
    H.save_manifest()
    n_anim = sum(1 for a in H._ARTIFACTS if a["kind"] == "animation")
    print(f"animations done: {n_anim} GIFs + filmstrip.")
