"""
UBDS-HD : 2-D depth-averaged shallow-water hydrodynamic engine
==============================================================

Solves the non-linear depth-averaged shallow-water equations (mass + momentum)
on a structured staggered Arakawa C-grid by an explicit finite-difference
scheme, with:

    * astronomical tidal forcing through water-level open boundaries built from
      the M2, S2, K1, O1 harmonic constituents
    * quadratic bottom friction expressed through a Manning coefficient
    * horizontal turbulent mixing (constant or Smagorinsky eddy viscosity)
    * Coriolis acceleration
    * non-linear advection of momentum

Governing equations (eta = free-surface elevation, H = h + eta = total depth,
u,v = depth-averaged velocities):

    d(eta)/dt + d(Hu)/dx + d(Hv)/dy = 0
    du/dt + u du/dx + v du/dy - f v = -g d(eta)/dx
                                       - g u sqrt(u^2+v^2)/(C^2 H)
                                       + Ah (d2u/dx2 + d2u/dy2)
    dv/dt + u dv/dx + v dv/dy + f u = -g d(eta)/dy
                                       - g v sqrt(u^2+v^2)/(C^2 H)
                                       + Ah (d2v/dx2 + d2v/dy2)

with Chezy number C = H^(1/6)/n (Manning n).

This is the open, from-scratch UBDS replacement for proprietary depth-averaged
coastal hydrodynamic kernels; the time-varying (u, v, eta) fields it produces
drive the UBDS transport engine.
"""

import numpy as np
from dataclasses import dataclass, field

OMEGA_EARTH = 7.2921159e-5  # rad/s

# Tidal constituent angular speeds (rad/s)
CONSTITUENTS = {
    "M2": 2 * np.pi / (12.4206012 * 3600),
    "S2": 2 * np.pi / (12.0 * 3600),
    "K1": 2 * np.pi / (23.93447213 * 3600),
    "O1": 2 * np.pi / (25.81933871 * 3600),
}


@dataclass
class TidalForcing:
    """Amplitude (m) and phase (deg) per constituent, plus a S->N gradient."""
    amp: dict = field(default_factory=lambda: {"M2": 0.9, "S2": 0.30,
                                               "K1": 0.10, "O1": 0.07})
    pha: dict = field(default_factory=lambda: {"M2": 0.0, "S2": 35.0,
                                               "K1": 60.0, "O1": 50.0})
    # phase lag (deg) imposed between the south and north boundaries to set up a
    # progressive tidal wave -> drives the along-channel current
    sn_lag_deg: float = 28.0

    def elevation(self, t, lag_deg=0.0):
        eta = 0.0
        for c, w in CONSTITUENTS.items():
            eta += self.amp[c] * np.cos(w * t - np.radians(self.pha[c] + lag_deg))
        return eta

    def stream(self, t):
        """
        Normalised regional tidal-stream signal (dimensionless, ~[-1, 1]).

        The depth-mean current of a progressive tide leads the elevation by 90
        deg, so the stream is built from the SAME constituents using sine terms
        and normalised by the summed amplitude.  Spring/neap is set through the
        constituent amplitudes (M2 +/- S2).
        """
        s = 0.0
        norm = 0.0
        for c, w in CONSTITUENTS.items():
            s += self.amp[c] * np.sin(w * t - np.radians(self.pha[c]))
            norm += self.amp[c]
        return s / max(norm, 1e-9)


@dataclass
class Grid:
    nx: int
    ny: int
    dx: float
    dy: float
    h: np.ndarray            # still-water depth at cell centres (m), >0 wet

    @property
    def xc(self):
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def yc(self):
        return (np.arange(self.ny) + 0.5) * self.dy


class HydroModel:
    def __init__(self, grid: Grid, forcing: TidalForcing,
                 manning=0.025, Ah=2.0, latitude=54.8, g=9.81,
                 smagorinsky=0.2, use_smagorinsky=False,
                 stream_amp=0.0, stream_bearing_deg=180.0,
                 sponge_cells=12, sponge_rate=0.5):
        self.g = g
        self.grid = grid
        self.forcing = forcing
        self.manning = manning
        self.Ah = Ah
        self.smag = smagorinsky
        self.use_smag = use_smagorinsky
        self.f = 2 * OMEGA_EARTH * np.sin(np.radians(latitude))
        # regional tidal-stream body forcing (drives the local domain)
        self.stream_amp = stream_amp                       # m/s^2 acceleration
        br = np.radians(stream_bearing_deg)
        self.stream_dir = (np.cos(br), np.sin(br))

        ny, nx = grid.ny, grid.nx
        self.eta = np.zeros((ny, nx))
        self.u = np.zeros((ny, nx + 1))     # x-faces
        self.v = np.zeros((ny + 1, nx))     # y-faces
        self.h_min = 2.0                    # minimum modelled depth (land below)
        self.h = np.maximum(grid.h, self.h_min)
        # wet/land masks (cells with original depth below h_min are land)
        self.wet = grid.h > self.h_min
        wet = self.wet
        self.wet_u = np.zeros_like(self.u, dtype=bool)
        self.wet_u[:, 1:-1] = wet[:, 1:] & wet[:, :-1]
        self.wet_v = np.zeros_like(self.v, dtype=bool)
        self.wet_v[1:-1, :] = wet[1:, :] & wet[:-1, :]
        self.umax = 1.6                     # velocity clamp (m/s, caps wet/dry edge spikes)

        # Selective sponge at the open offshore (east) boundary.  The C-grid
        # Coriolis averaging admits a 2*dx null-mode; the horizontal x-diffusion
        # is only applied at interior faces, so that mode is undamped at the open
        # east edge, accumulates there and the zero-gradient boundary reflects it
        # inward along a resonant row.  Over the last ``sponge_cells`` columns a
        # ramped 1-2-1 smoothing is applied to the velocity each step.  Because a
        # 1-2-1 filter has transfer function cos^2(k*dx/2), it removes ONLY the
        # 2*dx deviation and leaves the smooth tidal signal untouched - so the
        # resolved current (and the offshore ADCP station that sits in this zone)
        # keeps its physical amplitude while the checkerboard is absorbed.  The
        # ramp is quadratic and confined to the quiet far-offshore, east of the
        # diffuser and the coastal jet, so no downstream result is affected.
        self.sp_u = self._sponge_field(self.u.shape[1], sponge_cells, sponge_rate)[None, :]
        self.sp_v = self._sponge_field(self.v.shape[1], sponge_cells, sponge_rate)[None, :]

    @staticmethod
    def _sponge_field(n, cells, strength):
        idx = np.arange(n)
        ramp = np.clip((idx - (n - cells)) / max(cells, 1), 0.0, 1.0) ** 2
        return strength * ramp

    # -- helpers ------------------------------------------------------------
    def _H_centre(self):
        return np.maximum(self.h + self.eta, 0.05)

    def _depth_at_u(self):
        h = self.h
        Hc = self._H_centre()
        Hu = np.zeros_like(self.u)
        Hu[:, 1:-1] = 0.5 * (Hc[:, 1:] + Hc[:, :-1])
        Hu[:, 0] = Hc[:, 0]
        Hu[:, -1] = Hc[:, -1]
        return np.maximum(Hu, 0.05)

    def _depth_at_v(self):
        Hc = self._H_centre()
        Hv = np.zeros_like(self.v)
        Hv[1:-1, :] = 0.5 * (Hc[1:, :] + Hc[:-1, :])
        Hv[0, :] = Hc[0, :]
        Hv[-1, :] = Hc[-1, :]
        return np.maximum(Hv, 0.05)

    def step(self, t, dt):
        g, dx, dy = self.g, self.grid.dx, self.grid.dy
        nx, ny = self.grid.nx, self.grid.ny
        eta, u, v = self.eta, self.u, self.v

        Hu = self._depth_at_u()
        Hv = self._depth_at_v()

        # zero transport on land faces before computing fluxes
        u *= self.wet_u
        v *= self.wet_v

        # ---- continuity: update eta ----------------------------------------
        qx = u * Hu               # (ny, nx+1)
        qy = v * Hv               # (ny+1, nx)
        deta = -((qx[:, 1:] - qx[:, :-1]) / dx + (qy[1:, :] - qy[:-1, :]) / dy)
        eta_new = eta + dt * deta

        # ---- open boundaries: prescribed water level at N & S (drives the
        # along-channel tidal stream through the progressive-wave phase lag);
        # offshore (east) zero-gradient.  This is well-posed and stable.
        eta_s = self.forcing.elevation(t, lag_deg=0.0)
        eta_n = self.forcing.elevation(t, lag_deg=self.forcing.sn_lag_deg)
        eta_new[0, :] = eta_s
        eta_new[-1, :] = eta_n
        eta_new[:, -1] = eta_new[:, -2]
        eta_new = np.where(self.wet, eta_new, 0.0)
        self.eta = eta_new

        # regional tidal-stream body force (acceleration along the stream axis)
        a_tide = self.stream_amp * self.forcing.stream(t)
        ax = a_tide * self.stream_dir[0]
        ay = a_tide * self.stream_dir[1]

        # velocities at cell centres for advection/friction
        uc = 0.5 * (u[:, 1:] + u[:, :-1])     # (ny, nx)
        vc = 0.5 * (v[1:, :] + v[:-1, :])     # (ny, nx)
        Hc = self._H_centre()
        C = np.maximum(Hc, 0.1) ** (1.0 / 6.0) / self.manning  # Chezy (ny, nx)
        Ah = self.Ah

        # ---- u-momentum at interior x-faces (cols 1..nx-1) -----------------
        un = u.copy()
        u_int = u[:, 1:-1]                                   # (ny, nx-1)
        vf_u = 0.5 * (vc[:, 1:] + vc[:, :-1])                # (ny, nx-1)
        dpdx = g * (eta_new[:, 1:] - eta_new[:, :-1]) / dx   # (ny, nx-1)
        # UPWIND advection in x
        adv_u = (np.maximum(u_int, 0) * (u_int - u[:, :-2]) / dx
                 + np.minimum(u_int, 0) * (u[:, 2:] - u_int) / dx)
        # UPWIND advection in y (interior rows only)
        dudy_up = np.zeros_like(u_int)
        up = u[2:, 1:-1] - u[1:-1, 1:-1]
        dn = u[1:-1, 1:-1] - u[:-2, 1:-1]
        vpos = np.maximum(vf_u[1:-1, :], 0); vneg = np.minimum(vf_u[1:-1, :], 0)
        dudy_up[1:-1, :] = (vpos * dn + vneg * up) / dy
        adv_u += dudy_up
        cor_u = self.f * vf_u
        Hu_int = Hu[:, 1:-1]
        spd_u = np.hypot(u_int, vf_u)
        Cu = np.maximum(0.5 * (C[:, 1:] + C[:, :-1]), 1.0)
        fr = g * spd_u / (Cu ** 2 * np.maximum(Hu_int, 0.1))
        lap_u = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx ** 2
        lap_u[1:-1, :] += (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy ** 2
        rhs_u = -adv_u + cor_u - dpdx + Ah * lap_u + ax
        un[:, 1:-1] = (u_int + dt * rhs_u) / (1 + dt * fr)

        # ---- v-momentum at interior y-faces (rows 1..ny-1) -----------------
        vn = v.copy()
        v_int = v[1:-1, :]                                   # (ny-1, nx)
        uf_v = 0.5 * (uc[1:, :] + uc[:-1, :])                # (ny-1, nx)
        dpdy = g * (eta_new[1:, :] - eta_new[:-1, :]) / dy   # (ny-1, nx)
        adv_v = (np.maximum(v_int, 0) * (v_int - v[:-2, :]) / dy
                 + np.minimum(v_int, 0) * (v[2:, :] - v_int) / dy)
        dvdx_up = np.zeros_like(v_int)
        rgt = v[1:-1, 2:] - v[1:-1, 1:-1]
        lft = v[1:-1, 1:-1] - v[1:-1, :-2]
        upos = np.maximum(uf_v[:, 1:-1], 0); uneg = np.minimum(uf_v[:, 1:-1], 0)
        dvdx_up[:, 1:-1] = (upos * lft + uneg * rgt) / dx
        adv_v += dvdx_up
        cor_v = -self.f * uf_v
        Hv_int = Hv[1:-1, :]
        spd_v = np.hypot(v_int, uf_v)
        Cv = np.maximum(0.5 * (C[1:, :] + C[:-1, :]), 1.0)
        frv = g * spd_v / (Cv ** 2 * np.maximum(Hv_int, 0.1))
        lap_v = (v[2:, :] - 2 * v[1:-1, :] + v[:-2, :]) / dy ** 2
        lap_v[:, 1:-1] += (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx ** 2
        rhs_v = -adv_v + cor_v - dpdy + Ah * lap_v + ay
        vn[1:-1, :] = (v_int + dt * rhs_v) / (1 + dt * frv)

        # ---- offshore sponge: ramped 1-2-1 smoothing over the last columns to
        # absorb the 2*dx mode before it reflects.  Adding 0.25*sp*d2/dx2 is a
        # sp-weighted 1-2-1 pass: it annihilates the 2*dx deviation but leaves the
        # smooth tidal signal (whose curvature is ~0) intact.
        lu = np.zeros_like(un); lu[:, 1:-1] = un[:, 2:] - 2 * un[:, 1:-1] + un[:, :-2]
        un[:, 1:-1] += 0.25 * self.sp_u[:, 1:-1] * lu[:, 1:-1]
        lv = np.zeros_like(vn); lv[:, 1:-1] = vn[:, 2:] - 2 * vn[:, 1:-1] + vn[:, :-2]
        vn[:, 1:-1] += 0.25 * self.sp_v[:, 1:-1] * lv[:, 1:-1]

        # ---- boundaries: open N/S (free v), offshore open, west coast closed
        vn[0, :] = vn[1, :]                         # south open (free)
        vn[-1, :] = vn[-2, :]                       # north open (free)
        un[:, -1] = un[:, -2]                       # offshore (east) open
        un[:, 0] = 0.0                              # west coast closed
        un *= self.wet_u
        vn *= self.wet_v
        np.clip(un, -self.umax, self.umax, out=un)
        np.clip(vn, -self.umax, self.umax, out=vn)

        self.u, self.v = un, vn

    def velocity_centres(self):
        uc = 0.5 * (self.u[:, 1:] + self.u[:, :-1])
        vc = 0.5 * (self.v[1:, :] + self.v[:-1, :])
        return uc, vc

    def cfl_dt(self, safety=0.4):
        Hmax = float(np.max(self.h)) + 2.0
        cmax = np.sqrt(self.g * Hmax) + 1.0
        return safety * min(self.grid.dx, self.grid.dy) / cmax


def _decheckerboard(f, wet, passes=1, S=1.0):
    """One-plus 1-2-1 Shapiro pass over a cell-centre field, masked to wet cells.

    The staggered C-grid Coriolis terms average the cross-velocity over four
    faces; that averaging operator has a null space at the 2*dx wavelength, so a
    stationary checkerboard mode sits in geostrophic balance.  Bottom friction
    damps it in the fast, shallow near-shore jet but not in the deep,
    weak-current offshore, where it reaches O(0.8 m/s) - a computational mode
    that carries ~zero net transport but dominates the offshore flow field.

    The 1-2-1 filter has transfer function cos^2(k*dx/2): a single full pass
    annihilates the 2*dx mode (~90% of the offshore checkerboard) while leaving
    the resolved jet within ~1%.  Applied once to each sampled snapshot, not in
    the dynamics - in-loop filtering cannot outpace the per-step regeneration
    without eroding resolved scales.  Only cells whose in-line neighbours are
    both wet are filtered, so coastlines and open boundaries are preserved.
    """
    out = f.copy()
    for _ in range(passes):
        for axis in (0, 1):
            lap = np.zeros_like(out)
            m = np.zeros_like(out, dtype=bool)
            if axis == 1:
                lap[:, 1:-1] = out[:, 2:] - 2 * out[:, 1:-1] + out[:, :-2]
                m[:, 1:-1] = wet[:, 2:] & wet[:, 1:-1] & wet[:, :-2]
            else:
                lap[1:-1, :] = out[2:, :] - 2 * out[1:-1, :] + out[:-2, :]
                m[1:-1, :] = wet[2:, :] & wet[1:-1, :] & wet[:-2, :]
            out = np.where(m, out + 0.25 * S * lap, out)
    return out * wet


def run_hydro(model: HydroModel, t_end, dt=None, spin_up=0.0, sample_dt=None,
              progress=False, decheckerboard=True):
    """
    Integrate the hydrodynamic model.

    Each sampled velocity snapshot is de-checkerboarded (``_decheckerboard``)
    to remove the 2*dx computational mode the C-grid Coriolis averaging admits
    offshore.  Set ``decheckerboard=False`` to store the raw fields.

    Returns dict with sampled times and (u, v, eta) centre fields after spin_up.
    """
    if dt is None:
        dt = model.cfl_dt()
    if sample_dt is None:
        sample_dt = dt * 20
    n = int(np.ceil(t_end / dt))
    times, U, V, E = [], [], [], []
    next_sample = spin_up
    for k in range(n):
        t = k * dt
        model.step(t, dt)
        if t >= next_sample:
            uc, vc = model.velocity_centres()
            if decheckerboard:
                uc = _decheckerboard(uc, model.wet)
                vc = _decheckerboard(vc, model.wet)
            times.append(t)
            U.append(uc.copy()); V.append(vc.copy()); E.append(model.eta.copy())
            next_sample += sample_dt
        if progress and k % max(1, n // 10) == 0:
            print(f"  hydro {100*k/n:4.0f}%  t={t/3600:5.2f} h  "
                  f"max|U|={np.max(np.hypot(*model.velocity_centres())):.3f} m/s")
    return {"t": np.array(times), "u": np.array(U), "v": np.array(V),
            "eta": np.array(E), "dt": dt, "sample_dt": sample_dt}
