from pathlib import Path

import numpy as np
import torch


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "316L_meltpool.csv"

PROCESS_FEATURES = ["P_laser", "v_scan", "d_laser", "t_powder"]
TARGETS = ["melt_width", "melt_depth", "melt_area"]
INPUT_FEATURES = PROCESS_FEATURES + ["sensor_1", "sensor_2", "sensor_3", "delta_t"]

WINDOW_EVENTS = 8
N_JOBS = 24
EVENTS_PER_JOB = 25
EPOCHS = 60
BATCH_SIZE = 64
N_LOOPS = 4

T_AMBIENT_K = 300.0
T_LIQUIDUS_K = 1700.0
THERMAL_CONDUCTIVITY_W_MK = 25.0
THERMAL_DIFFUSIVITY_M2_S = 4.0e-6
GRID_DY_UM = 5.0
GRID_DZ_UM = 5.0
GRID_HALF_WIDTH_UM = 350.0
GRID_DEPTH_UM = 300.0


def seed_everything(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)