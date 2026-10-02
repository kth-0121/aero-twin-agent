# STL/CLI Pre-LT Digital Twin

This module set replays one illustrative 316L LPBF build from a watertight STL or an existing ASCII Common Layer Interface (CLI) file. It ends before the Looped Transformer: the current result is scan-path geometry plus a qualitative thermal field, not an LT prediction or a part-quality guarantee.

## Demo scenario

The preset in `process.py` uses 316L to match the available melt-pool training data. The nominal settings are 200 W laser power, 800 mm/s scan speed, 0.10 mm hatch spacing, 30 um layer thickness, 0.10 mm spot diameter, 100 C plate preheat, argon shielding, and an 80 ppm measured oxygen level against a 100 ppm limit. The powder lot and particle distribution are explicitly illustrative placeholders and must be replaced with a real powder certificate before making process claims.

The support strategy, density, and contact size travel with each build event as metadata. This first slicer does not generate support geometry; support-containing geometry must be included in the STL or supplied by a production slicer.

## Modules

- `process.py`: validated powder, process, chamber, support, and machine settings.
- `stl_slicer.py`: watertight STL slicing, contour extraction, and alternating raster hatches.
- `cli_toolpath.py`: ASCII CLI reader/writer and normalized scan-segment contract.
- `build_job.py`: build-volume and oxygen-purge gates; emits ordered `BuildEvent` values.
- `fipy_dt.py`: coarse 2D transient heat-equation replay driven by scan events.
- `pybullet_dt.py`: build plate, STL visualization, and recent scan-track visualization.
- `pre_lt_dt.py`: runs each event through FiPy and PyBullet; deliberately has no LT call.

STL has no dependable unit metadata, so the source unit is required as an explicit assumption (defaults to mm). The generated CLI uses mm. Imported CLI XY coordinates are centered on the build plate when they become build events.

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

Use an existing ASCII CLI file from an external slicer:

```bash
python -m digitaltwin.pre_lt_demo /path/to/part.stl --cli-input /path/to/part.cli --source-units mm --gui --event-limit 100
```

Run the focused module tests:

```bash
python -m unittest digitaltwin.test_pre_lt_pipeline
```

## Modeling boundary

The FiPy model uses a coarse 2D build-plane approximation and assumed 316L properties aligned with the repository ROM. Its temperatures are qualitative comparative signals only and may exceed realistic process temperatures; do not use their absolute values for process decisions. It does not model latent heat, melt-pool fluid flow, layer-by-layer 3D heat accumulation, porosity, distortion, or residual stress. PyBullet is used to visualize geometry and path replay, not as a calibrated LPBF physics solver. No machine commands are sent.