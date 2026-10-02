# Looped Transformer V5 modules

This package contains the V5 streaming experiment extracted from `train/LT_v5.ipynb`. The notebook remains the experiment and visualization entry point; the reusable logic lives here.

## Module map

- `config.py`: feature contracts, defaults, paths, and deterministic seeding.
- `data.py` and `simulation.py`: CSV validation, chronological scaling, and simulated replay events.
- `physics.py`: train-only calibration and prediction for the Rosenthal geometry anchor.
- `windows.py`, `model.py`, and `training.py`: sequence creation, the residual-loop network, training, and metrics.
- `monitoring.py`: online history state and drift score.
- `service.py`: event validation, inference, SQLite persistence, and optional queue consumer.
- `artifacts.py`: model, scaler, and configuration export.

## Digital twin integration

Run imports from the repository root. A live event must provide the four process inputs (`P_laser`, `v_scan`, `d_laser`, `t_powder`) and the three sensor values (`sensor_1`, `sensor_2`, `sensor_3`). Values must use the same units and feature definitions used during training. Sensor values in the notebook are synthetic proxies; replace them with validated adapter measurements before using the service in a real integration.

```python
from model.LoopedTransformer.v5 import AdaptivePhysicsResidualLT, RosenthalPhysicsAnchor
from model.LoopedTransformer.v5.config import DEVICE, INPUT_FEATURES, N_LOOPS
from model.LoopedTransformer.v5.data import load_dataset, transform_stream_splits
from model.LoopedTransformer.v5.model import AdaptivePhysicsResidualLT
from model.LoopedTransformer.v5.service import StreamingLTService

# Supply a trained model, fitted input/output scalers, calibrated anchor, and
# training-reference input matrix to StreamingLTService.
service = StreamingLTService(
    model=model.to(DEVICE),
    x_scaler=x_scaler,
    y_scaler=y_scaler,
    physics_anchor=physics_anchor,
    reference_X=train_stream[INPUT_FEATURES].to_numpy(),
    db_path="physical_ai_stream.sqlite",
)
result = service.process_event({
    "event_id": "job-001:42",
    "job_id": "job-001",
    "P_laser": 200.0,
    "v_scan": 800.0,
    "d_laser": 0.08,
    "t_powder": 30.0,
    "sensor_1": 0.52,
    "sensor_2": 0.31,
    "sensor_3": 0.74,
})
```

The service returns geometry predictions, the physics anchor, per-loop predictions, approximate uncertainty, drift diagnostics, and event identifiers. It does not update model weights or issue equipment-control commands. Calibrate the physics anchor from training data only and validate the complete integration against the target machine before operational use.
