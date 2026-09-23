"""1D composite cure simulator: Arrhenius reaction kinetics + heat conduction.

Generates training data for the physics-aware core, and is also the ground
truth we compare What-if scenarios against. Only node 0 (surface) is treated
as sensor-observable downstream -- the interior degree-of-cure is the hidden
state the model has to infer.
"""
import numpy as np

R = 8.314        # gas constant J/mol-K

# Arrhenius cure kinetics params (representative epoxy prepreg values)
A = 1.53e7       # pre-exponential (1/s)
E = 6.65e4       # activation energy (J/mol)
N_ORDER = 1.2    # reaction order
H_TOTAL = 3.5e5  # total heat of reaction (J/kg)


def cure_rate(alpha, T):
    """Arrhenius cure rate d(alpha)/dt."""
    alpha = np.clip(alpha, 0, 0.999)
    return A * np.exp(-E / (R * T)) * (1 - alpha) ** N_ORDER


def simulate_cure(surface_temp_fn, n_nodes=5, thickness=0.01, k=0.8, rho=1600,
                   cp=1200, t_span=(0, 7200), dt=1.0):
    """
    surface_temp_fn(t) -> boundary temperature [K] applied at both faces.
    Returns dict(t, T[node,time], alpha[node,time]).
    """
    dx = thickness / (n_nodes - 1)
    alpha_diff = k / (rho * cp)

    T = np.full(n_nodes, surface_temp_fn(0.0))
    alpha = np.zeros(n_nodes)

    steps = int(t_span[1] / dt)
    T_hist = np.zeros((n_nodes, steps))
    alpha_hist = np.zeros((n_nodes, steps))
    t_hist = np.zeros(steps)

    for i in range(steps):
        t = i * dt
        T[0] = T[-1] = surface_temp_fn(t)

        da = cure_rate(alpha, T) * dt
        alpha = np.clip(alpha + da, 0, 1.0)
        heat_gen = rho * H_TOTAL * (da / dt)  # W/m^3, exothermic reaction

        lap = np.zeros(n_nodes)
        lap[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx ** 2
        dT = alpha_diff * lap[1:-1] + heat_gen[1:-1] / (rho * cp)
        T[1:-1] += dT * dt

        T_hist[:, i] = T
        alpha_hist[:, i] = alpha
        t_hist[i] = t

    return {"t": t_hist, "T": T_hist, "alpha": alpha_hist}


def demo():
    """Self-check: a standard ramp-hold cure cycle should fully cure the interior."""
    def ramp_hold(t):
        ramp_rate = 3 / 60  # K/s (~3C/min)
        hold_temp = 180 + 273.15
        start = 20 + 273.15
        return min(start + ramp_rate * t, hold_temp)

    out = simulate_cure(ramp_hold, n_nodes=5, t_span=(0, 3600), dt=2.0)
    final_alpha_center = out["alpha"][2, -1]
    assert 0.8 < final_alpha_center <= 1.0, f"expected near-full cure, got {final_alpha_center}"
    print(f"demo ok: center degree-of-cure at end = {final_alpha_center:.3f}")


if __name__ == "__main__":
    demo()
