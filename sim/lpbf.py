"""LPBF melt pool physics ROM (Reduced-Order Model) -- the physics evaluator P
for the AI<->Physics loop, and the team's chosen formulas from the process
scenario doc (팀 결론, 2026-09-25):

  W = sqrt(8*A*P / (pi*e*rho*cp*v*(Tm-T0)))      -- Rosenthal-based width scaling
  (h/W)^2 + (t/D)^2 <= 1                          -- Lack-of-Fusion safety criterion

All physical quantities are SI (power in W, speed in m/s, lengths in m)
inside this module; callers on the um/mm scale (as the 316L dataset uses)
must convert at the boundary.
"""
import numpy as np

# 316L stainless steel, representative literature values
RHO = 7990.0      # kg/m^3, density
CP = 500.0        # J/kg-K, specific heat
T_M = 1673.0      # K, melting point (~1400C)
T_0 = 300.0       # K, ambient/preheat temperature
A_ABS = 0.35      # absorptivity (typical for SS under IR laser)


def melt_pool_width(power_w, speed_m_s):
    """Rosenthal scaling formula (team-derived), returns width in meters."""
    speed_m_s = np.maximum(speed_m_s, 1e-6)  # guard div-by-zero for stalled scans
    return np.sqrt(8 * A_ABS * power_w / (np.pi * np.e * RHO * CP * speed_m_s * (T_M - T_0)))


def melt_pool_depth(power_w, speed_m_s):
    """ponytail: conduction-mode semicircular assumption (D = W/2). Team's own
    doc flags simple heat-conduction models as unreliable in keyhole mode --
    this is exactly where that breaks down. Upgrade path: Eager-Tsai depth
    solution, or a data-fit correction once more measured depth data exists."""
    return melt_pool_width(power_w, speed_m_s) / 2


def volumetric_energy_density(power_w, speed_m_s, hatch_m, layer_t_m):
    """VED [J/m^3], standard AM process metric used to flag likely regime."""
    denom = np.maximum(speed_m_s * hatch_m * layer_t_m, 1e-18)
    return power_w / denom


def lof_risk(hatch_m, layer_t_m, width_m, depth_m):
    """Team's LOF criterion: (h/W)^2 + (t/D)^2 <= 1 is a safe (fused) track.
    >1 means adjacent tracks/layers likely fail to fuse."""
    width_m = np.maximum(width_m, 1e-9)
    depth_m = np.maximum(depth_m, 1e-9)
    return (hatch_m / width_m) ** 2 + (layer_t_m / depth_m) ** 2


def demo():
    # 200W, 800mm/s -- a commonly-cited good 316L process point (see search notes)
    power, speed = 200.0, 0.8
    w = melt_pool_width(power, speed)
    d = melt_pool_depth(power, speed)
    assert 0 < d < w, f"expected 0 < depth < width, got d={d}, w={w}"

    risk = lof_risk(hatch_m=100e-6, layer_t_m=30e-6, width_m=w, depth_m=d)
    print(f"demo ok: W={w*1e6:.1f}um D={d*1e6:.1f}um LOF-criterion={risk:.3f} "
          f"({'safe' if risk <= 1 else 'LOF risk'})")


if __name__ == "__main__":
    demo()
