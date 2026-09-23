"""Tools the AI agent can call. Small enough that a dict dispatch beats a framework."""
from sim.cure import simulate_cure


def run_whatif(ramp_rate_c_per_min, hold_temp_c, hold_seconds=3600):
    """Run the cure simulator under a given cycle and summarize the outcome."""
    def profile(t):
        start = 20 + 273.15
        hold = hold_temp_c + 273.15
        ramp_rate = ramp_rate_c_per_min / 60
        return min(start + ramp_rate * t, hold)

    out = simulate_cure(profile, t_span=(0, hold_seconds))
    center = out["alpha"].shape[0] // 2
    final_alpha = out["alpha"][center, -1]
    return {
        "final_degree_of_cure": round(float(final_alpha), 3),
        "fully_cured": bool(final_alpha > 0.95),
    }


TOOLS = {
    "run_whatif": run_whatif,
}


def demo():
    result = run_whatif(ramp_rate_c_per_min=3.0, hold_temp_c=180, hold_seconds=1800)
    assert "final_degree_of_cure" in result
    assert 0.0 <= result["final_degree_of_cure"] <= 1.0
    print(f"demo ok: {result}")


if __name__ == "__main__":
    demo()
