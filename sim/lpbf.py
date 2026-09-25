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

# 316L stainless steel -- values match the team's own validated calculation
# assumptions (공정시나리오 탐색.md 3차 검증), not just generic literature picks.
RHO = 7900.0      # kg/m^3, density
CP = 650.0        # J/kg-K, specific heat
T_M = 1700.0      # K, melting point
T_0 = 300.0       # K, ambient/preheat temperature
A_ABS = 0.35      # absorptivity (team notes IN625 reports 0.28-0.43; 316L similar)


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


def area_energy_density(power_w, speed_m_s, spot_diameter_m):
    """P/(v*d) [J/m^2], area energy density -- a standard AM process proxy.
    ponytail: the team's own doc leaves the *proper* normalized-enthalpy
    convention for the conduction/keyhole threshold undecided (King 2014 vs
    Hann 2011 disagree on the sigma definition). This threshold is instead
    fit directly against the 316L dataset's own D/W>=0.5 split (~78% match),
    not derived from first principles -- replace with normalized enthalpy
    once the team locks a convention."""
    speed_m_s = np.maximum(speed_m_s, 1e-6)
    spot_diameter_m = np.maximum(spot_diameter_m, 1e-9)
    return power_w / (speed_m_s * spot_diameter_m)


AREA_ENERGY_KEYHOLE_THRESHOLD = 4.4e6  # J/m^2, calibrated against data/316L_meltpool.csv


def classify_regime(power_w, speed_m_s, spot_diameter_m):
    """'conduction' (Rosenthal ROM is valid) vs 'keyhole_or_transition'
    (ROM known to underestimate depth -- team measured ~-30% width bias
    even in conduction rows; it's worse once keyhole effects kick in)."""
    ed = area_energy_density(power_w, speed_m_s, spot_diameter_m)
    is_keyhole = ed > AREA_ENERGY_KEYHOLE_THRESHOLD
    return np.where(is_keyhole, "keyhole_or_transition", "conduction")


def lof_risk(hatch_m, layer_t_m, width_m, depth_m):
    """Team's LOF criterion: (h/W)^2 + (t/D)^2 <= 1 is a safe (fused) track.
    >1 means adjacent tracks/layers likely fail to fuse."""
    width_m = np.maximum(width_m, 1e-9)
    depth_m = np.maximum(depth_m, 1e-9)
    return (hatch_m / width_m) ** 2 + (layer_t_m / depth_m) ** 2


def suggest_process_params(target_depth_um, spot_diameter_um=100.0, hatch_um=100.0,
                            layer_thickness_um=30.0, power_range=(50.0, 500.0), n_power=25):
    """Inverse design (team's '목표 제어' experiment): given a target melt-pool
    depth, invert the Rosenthal formula (assuming D=W/2, so target W = 2*D)
    for scan speed at each of a grid of laser powers, then pick the fastest
    candidate that lands in the conduction regime with a safe LOF margin.
    Analytic + a small grid search -- no trained model needed, matching the
    team's own "목표 D -> P·v 제안 -> DT(ROM)로 확인" flow."""
    target_width_m = 2 * target_depth_um * 1e-6
    spot_m = spot_diameter_um * 1e-6
    hatch_m = hatch_um * 1e-6
    layer_m = layer_thickness_um * 1e-6

    candidates = []
    for power in np.linspace(power_range[0], power_range[1], n_power):
        speed_m_s = 8 * A_ABS * power / (np.pi * np.e * RHO * CP * target_width_m ** 2 * (T_M - T_0))
        if speed_m_s <= 0:
            continue
        w = melt_pool_width(power, speed_m_s)
        d = melt_pool_depth(power, speed_m_s)
        regime = str(classify_regime(power, speed_m_s, spot_m))
        risk = float(lof_risk(hatch_m, layer_m, w, d))
        candidates.append({
            "laser_power_w": round(float(power), 1),
            "scan_speed_mm_s": round(float(speed_m_s) * 1000, 1),
            "predicted_depth_um": round(float(d) * 1e6, 1),
            "predicted_width_um": round(float(w) * 1e6, 1),
            "regime": regime,
            "lof_criterion": round(risk, 3),
            "safe_fusion": risk <= 1.0,
        })

    safe_conduction = [c for c in candidates if c["regime"] == "conduction" and c["safe_fusion"]]
    if safe_conduction:
        best = max(safe_conduction, key=lambda c: c["scan_speed_mm_s"])  # fastest safe option
        best["note"] = "conduction 영역에서 안전하게 목표 깊이를 만족하는, 가장 빠른(생산성 높은) 조건"
    else:
        conduction_only = [c for c in candidates if c["regime"] == "conduction"]
        pool = conduction_only or candidates
        best = min(pool, key=lambda c: c["lof_criterion"])
        best["note"] = "conduction+안전 조건을 동시에 만족하는 후보가 없어 최선의 근사값을 반환함 -- 주의 필요"
    return best


def demo():
    # 200W, 800mm/s -- a commonly-cited good 316L process point (see search notes)
    power, speed = 200.0, 0.8
    w = melt_pool_width(power, speed)
    d = melt_pool_depth(power, speed)
    assert 0 < d < w, f"expected 0 < depth < width, got d={d}, w={w}"

    risk = lof_risk(hatch_m=100e-6, layer_t_m=30e-6, width_m=w, depth_m=d)
    regime = classify_regime(power, speed, spot_diameter_m=100e-6)
    print(f"demo ok: W={w*1e6:.1f}um D={d*1e6:.1f}um LOF-criterion={risk:.3f} "
          f"({'safe' if risk <= 1 else 'LOF risk'}) regime={regime}")

    suggestion = suggest_process_params(target_depth_um=60.0)
    assert abs(suggestion["predicted_depth_um"] - 60.0) < 1.0, suggestion
    print(f"demo ok: target 60um depth -> {suggestion}")


if __name__ == "__main__":
    demo()
