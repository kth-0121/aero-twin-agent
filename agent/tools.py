"""Tools the AI agent can call. Small enough that a dict dispatch beats a framework."""
from sim.lpbf import melt_pool_width, melt_pool_depth, lof_risk


def predict_melt_pool(laser_power_w, scan_speed_mm_s, hatch_um=100.0, layer_thickness_um=30.0):
    """Rosenthal-ROM estimate of melt pool depth/width for given LPBF settings,
    plus the LOF (Lack-of-Fusion) safety check. All physical lengths returned in um."""
    speed_m_s = scan_speed_mm_s / 1000.0
    width_m = melt_pool_width(laser_power_w, speed_m_s)
    depth_m = melt_pool_depth(laser_power_w, speed_m_s)
    risk = lof_risk(hatch_um * 1e-6, layer_thickness_um * 1e-6, width_m, depth_m)
    return {
        "depth_um": round(float(depth_m) * 1e6, 1),
        "width_um": round(float(width_m) * 1e6, 1),
        "lof_criterion": round(float(risk), 3),
        "safe_fusion": bool(risk <= 1.0),
    }


TOOLS = {
    "predict_melt_pool": predict_melt_pool,
}


def demo():
    result = predict_melt_pool(laser_power_w=200, scan_speed_mm_s=800)
    assert "depth_um" in result and "width_um" in result
    assert result["depth_um"] > 0 and result["width_um"] > 0
    print(f"demo ok: {result}")


if __name__ == "__main__":
    demo()
