import numpy as np
import pandas as pd

from .config import EVENTS_PER_JOB, N_JOBS, PROCESS_FEATURES, SEED, TARGETS


def robust_z(values):
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return (values - median) / scale


def make_simulated_stream(
    data,
    n_jobs=N_JOBS,
    events_per_job=EVENTS_PER_JOB,
    seed=SEED,
    drift_start_fraction=0.65,
):
    rng = np.random.default_rng(seed)
    process_values = data[PROCESS_FEATURES].to_numpy(float)
    target_values = data[TARGETS].to_numpy(float)
    normalized = robust_z(process_values)
    rows = []
    total_events = n_jobs * events_per_job
    global_index = 0
    q25, q75 = np.percentile(process_values, [25, 75], axis=0)
    iqr = np.maximum(q75 - q25, 1e-6)
    median = np.median(process_values, axis=0)

    for job in range(n_jobs):
        current = int(rng.integers(0, len(data)))
        chosen = []
        for step in range(events_per_job):
            if step == 0 or rng.random() < 0.18:
                current = int(rng.integers(0, len(data)))
            else:
                distances = np.mean((normalized - normalized[current]) ** 2, axis=1)
                nearest = np.argsort(distances)[:min(30, len(data))]
                current = int(rng.choice(nearest))
            chosen.append(current)

        latent = 0.0
        time_s = 0.0
        previous_time = 0.0
        for step, row_index in enumerate(chosen):
            if step > 0:
                time_s += float(rng.uniform(1.0, 10.0))
            latent = 0.92 * latent + rng.normal(0, 0.18)
            observed = process_values[row_index].copy() + rng.normal(0, 0.015, size=4) * iqr
            scaled = (observed - median) / iqr
            sensor_1 = 0.45 * scaled[0] - 0.25 * scaled[1] + 0.35 * latent + rng.normal(0, 0.12)
            sensor_2 = 0.20 * scaled[1] + 0.30 * scaled[2] + 0.25 * latent + rng.normal(0, 0.12)
            sensor_3 = 0.25 * scaled[0] + 0.15 * scaled[3] + 0.20 * latent + rng.normal(0, 0.12)
            drifted = global_index >= int(total_events * drift_start_fraction)
            if drifted:
                sensor_1 = 1.35 * sensor_1 + 0.55
                sensor_2 = 0.75 * sensor_2 - 0.35
                sensor_3 = 1.20 * sensor_3 + 0.25

            row = {
                "job_id": f"job_{job:03d}",
                "event_index": step,
                "global_index": global_index,
                "time_s": time_s,
                "delta_t": 0.0 if step == 0 else time_s - previous_time,
                "drift_injected": bool(drifted),
                "latent_state_sim": latent,
                **{PROCESS_FEATURES[i]: float(observed[i]) for i in range(4)},
                "sensor_1": float(sensor_1),
                "sensor_2": float(sensor_2),
                "sensor_3": float(sensor_3),
                **{TARGETS[i]: float(target_values[row_index, i]) for i in range(3)},
                "source_row": int(row_index),
            }
            rows.append(row)
            previous_time = time_s
            global_index += 1

    result = pd.DataFrame(rows)
    result["data_origin"] = "SIMULATED_FROM_STATIC_DATA"
    return result