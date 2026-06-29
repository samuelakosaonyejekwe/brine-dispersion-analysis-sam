"""
UBDS diffuser geometry
======================

Describes a multi-port diffuser and assembles the near-field result for the
whole diffuser (handling port activation logic and plume merging).
"""

import numpy as np
from dataclasses import dataclass, field
from . import nearfield as nf


@dataclass
class Diffuser:
    """A line diffuser of identical ports."""
    n_ports_installed: int = 2
    port_spacing: float = 20.0          # m, centre-to-centre
    port_diameter: float = 0.1524       # m  (6 inch Tideflex)
    port_height: float = 0.75           # m above seabed
    theta_deg: float = 90.0             # vertical discharge
    x0: float = 0.0                     # diffuser mid-point easting offset (m)
    y0: float = 0.0
    bearing_deg: float = 0.0            # diffuser line orientation

    def active_ports(self, total_flow_m3hr, switch_flow=500.0):
        """Operating rule: 1 port below switch_flow, else all installed ports."""
        return 1 if total_flow_m3hr <= switch_flow else self.n_ports_installed

    def port_positions(self, n_active):
        """(x, y) of each active port, centred on the diffuser midpoint."""
        idx = np.arange(n_active) - (n_active - 1) / 2.0
        s = idx * self.port_spacing
        bx = np.cos(np.radians(self.bearing_deg))
        by = np.sin(np.radians(self.bearing_deg))
        return np.column_stack([self.x0 + s * bx, self.y0 + s * by])


def run_diffuser(total_flow_m3hr, amb, disch_template, diff: Diffuser,
                 switch_flow=500.0, n_active=None):
    """
    Run the near-field model for a diffuser at a given total flow.

    n_active : if given, use this many active ports; otherwise apply the
               flow-based port-activation rule.

    Returns
    -------
    dict with keys:
        n_active, per_port_flow, result (NearFieldResult), ports (positions)
    """
    if n_active is None:
        n_active = diff.active_ports(total_flow_m3hr, switch_flow)
    per_port = total_flow_m3hr / n_active
    d = nf.Discharge(flow_m3hr=per_port,
                     diameter_m=diff.port_diameter,
                     salinity=disch_template.salinity,
                     temperature=disch_template.temperature,
                     z0=diff.port_height,
                     theta_deg=diff.theta_deg)
    res = nf.simulate(d, amb, n_ports=n_active, port_spacing=diff.port_spacing)
    return {"n_active": n_active, "per_port_flow": per_port,
            "result": res, "ports": diff.port_positions(n_active),
            "exit_velocity": d.exit_velocity()}
