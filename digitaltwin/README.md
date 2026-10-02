# STL/CLI Pre-LT Digital Twin

This module set replays one illustrative 316L LPBF build from a watertight STL or an existing ASCII Common Layer Interface (CLI) file. It ends before the Looped Transformer: the current result is scan-path geometry plus a qualitative thermal field, not an LT prediction or a part-quality guarantee.

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
- `pre_lt_dt.py`: runs each event through FiPy and PyBullet; deliberately has no LT call.

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