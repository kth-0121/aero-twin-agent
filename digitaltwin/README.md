# LPBF Build Supervision Console

The FastAPI console uses an operator-gated job sequence:

1. Upload an STL and create a validated CLI scan plan.
2. Start the FiPy qualitative thermal replay explicitly.
3. Start the LT V5 Agent explicitly after thermal replay completes. It predicts geometry per scan vector, evaluates physical constraints, and checks candidate adjustments with the Rosenthal ROM.
4. Review the aggregated process-variable recommendation and approve or reject it. Approval stores the values for the next virtual job only.

The application starts idle. It does not replay a synthetic build, run the Agent, or send machine commands at startup. If the operator starts the workflow without selecting an STL, the console loads the bundled `nozzle.preview.cli` path and `nozzle.stl` model when those files are present. The view renders the complete CLI path using a compact 0.01 mm binary coordinate stream; thermal/Agent analysis uses a deterministic sample of at most 100,000 vectors. Uploading an STL continues to use that STL and the regular slicing flow.

Run from the repository root:

```bash
python -m uvicorn digitaltwin.backend.main:app --host 0.0.0.0 --port 8000
```

The interface follows a restrained service-supervision layout: compact workflow cards, status summaries, and an equipment/build view. Its system font stack is Arial/Noto Sans KR; it does not download IBM Plex fonts. STL models are displayed at their actual scale relative to the configured 250 × 250 mm build plate rather than being enlarged to nearly fill the plate.

The hatch generator scans the source polygon directly along alternating X/Y axes, avoiding per-vector geometry rotation. Layer progress is exposed while slicing and shown in the STL job card. On a local benchmark using a 1,280-face icosphere scaled to 20 × 20 × 4 mm (134 layers, 30,513 scan vectors), median slicing time fell from 4.361 s to 2.343 s; actual part times depend on mesh complexity, layer count, and hatch spacing.

## Workflow API

- `POST /api/builds?source_units=mm` — upload raw STL bytes and prepare scan vectors.
- `POST /api/builds/preview` — load the bundled nozzle CLI path; the response reports full and sampled vector counts.
- `GET /api/builds/{id}/toolpath` — return the bounded analysis sample as JSON.
- `GET /api/builds/{id}/toolpath/full` — return the complete bundled CLI path as signed 16-bit coordinates at 0.01 mm resolution.
- `POST /api/builds/{id}/thermal-step?count=16` — advance the 2D FiPy replay; repeat until `phase=thermal_complete`.
- `POST /api/builds/{id}/agent/start` — load and start the LT Agent; only accepted after thermal analysis.
- `POST /api/builds/{id}/agent/step?count=16` — process scan vectors; repeat until `phase=review`.
- `POST /api/builds/{id}/review` with `{"approved": true|false}` — approve/reject the recommendation.

The browser advances thermal and Agent work in bounded batches and displays progress. The API rejects out-of-order stage requests.

## Demo scenario

The default preset in `process.py` uses 316L to match the available melt-pool training data. Its laser, preheat, chamber, powder, support, and machine values are illustrative only. Supply the actual machine profile and powder certificate with `--profile-json`; chamber oxygen and powder oxygen are separate fields.

The support strategy, density, and contact size travel with each build event as metadata. This first slicer does not generate support geometry; support-containing geometry must be included in the STL or supplied by a production slicer.

## Modules

- `process.py`: validated powder, process, chamber, support, and machine settings.
- `stl_slicer.py`: watertight STL slicing, contour extraction, and alternating raster hatches.
- `cli_toolpath.py`: ASCII CLI reader/writer and normalized scan-segment contract.
- `build_job.py`: build-volume and oxygen-purge gates; emits ordered `BuildEvent` values.
- `fipy_dt.py`: coarse 2D transient heat-equation replay driven by scan events.
- `pybullet_dt.py`: build plate, STL visualization, and recent scan-track visualization.
- `pre_lt_dt.py`: standalone CLI runner for the earlier FiPy/PyBullet pre-LT visualization.
- `backend/main.py`: operator-gated FastAPI workflow; FiPy precedes the LT Agent.

STL has no dependable unit metadata, so the source unit is required as an explicit assumption (defaults to mm). The generated CLI uses mm. Imported CLI XY coordinates are centered on the build plate when they become build events. The current external CLI reader accepts ASCII CLI polyline records; vendor-specific or hatch-record variants need a matching parser adapter. No real part STL, machine-exported CLI, or powder certificate is included in this repository.

## Run

Install the DT dependencies from the repository root:

```bash
pip install -r digitaltwin/requirements.txt
```

`part.stl` below is a placeholder for your own watertight model. To smoke-test the pipeline with a tiny 1 x 1 x 0.06 mm box, create an STL first:

```bash
python -c "import trimesh; trimesh.creation.box(extents=(1.0, 1.0, 0.06)).export('/tmp/part.stl')"
python -m digitaltwin.pre_lt_demo /tmp/part.stl --source-units mm --cli-output /tmp/part.cli --event-limit 100
```

For a real build, replace `/tmp/part.stl` with the path to your STL. STL files do not reliably store units, so set `--source-units` to match the CAD export.

Pass a machine/material JSON profile to replace every illustrative process assumption:

```bash
python -m digitaltwin.pre_lt_demo /path/to/part.stl --source-units mm --profile-json /path/to/build-profile.json --cli-output /tmp/part.cli --event-limit 100
```

The JSON profile requires `job_id`, `material`, and objects `powder`, `process`, `supports`, `chamber`, and `machine`. `powder` requires `material`, `lot_id`, `particle_d10_um`, `particle_d50_um`, `particle_d90_um`, `oxygen_ppm`, `reuse_count`, and `data_origin`. `process` requires `laser_power_w`, `scan_speed_mm_s`, `hatch_spacing_mm`, `layer_thickness_um`, `spot_diameter_mm`, and `preheat_temperature_c`. `supports` requires `strategy`, `density_percent`, `contact_diameter_mm`, and `removal_access_required`; `chamber` requires `shielding_gas`, `oxygen_limit_ppm`, and `oxygen_measured_ppm`; `machine` requires `machine_id`, `build_width_mm`, `build_depth_mm`, and `build_height_mm`. Use values and provenance from the machine recipe, powder certificate, chamber readback, and support plan; do not copy the demo defaults as certified settings.

Use an existing ASCII CLI file from an external slicer:

```bash
python -m digitaltwin.pre_lt_demo /path/to/part.stl --cli-input /path/to/part.cli --source-units mm --gui --event-limit 100
```

Run the focused module tests:

```bash
python -m unittest digitaltwin.test_pre_lt_pipeline
```

## Modeling boundary

FiPy output is now reported as fractions of grid cells above the preheat-relative hot threshold and a reference-temperature threshold, not as degrees Celsius. These are qualitative, resolution-dependent indicators for comparing replayed paths under the same settings; they are not melt-pool area, defect probability, or process-safety estimates. The coarse 2D model does not model latent heat, melt-pool fluid flow, calibrated material behavior, or layer-by-layer 3D heat accumulation. PyBullet is used to visualize geometry and path replay, not as a calibrated LPBF physics solver. No machine commands are sent.

The console's LT input sensor values are a synthetic proxy supplied by the existing V5 streaming adapter. Trend direction is a comparison of recent predicted risk rates, not a calibrated long-horizon forecast. The demo profile, ROM recommendation, and FiPy field are simulation aids, not process certification or machine instructions. Approval changes only the process values saved for the next virtual job.