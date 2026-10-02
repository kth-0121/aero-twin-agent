"""Qualitative transient thermal field driven by CLI scan events using FiPy."""

from dataclasses import dataclass
from math import ceil, exp, isfinite, sqrt

import numpy as np

from .build_job import BuildEvent
from .process import BuildSettings


@dataclass(frozen=True)
class ThermalSnapshot:
    event_id: str
    peak_temperature_c: float
    hot_cell_count: int
    melt_threshold_cell_count: int
    grid_shape: tuple[int, int]
    cell_size_mm: tuple[float, float]


class FiPyThermalTwin:
    """Advance a coarse 2D heat equation along each CLI scan vector.

    This reduced model is for path replay and qualitative thermal comparison;
    it is not a calibrated melt-pool or porosity predictor.
    """

    def __init__(
        self,
        settings: BuildSettings,
        *,
        build_width_mm: float,
        build_depth_mm: float,
        resolution_mm: float = 0.25,
        max_cells: int = 40_000,
        max_substeps_per_vector: int = 8,
    ):
        if settings.material != "316L":
            raise ValueError("the FiPy demo material properties are defined only for 316L")
        if not all(
            isfinite(value) and value > 0
            for value in (build_width_mm, build_depth_mm, resolution_mm)
        ):
            raise ValueError("thermal domain dimensions and resolution must be positive")
        if max_cells < 4 or max_substeps_per_vector < 1:
            raise ValueError("max_cells and max_substeps_per_vector are too small")

        try:
            from fipy import CellVariable, Grid2D, DiffusionTerm, TransientTerm
        except ImportError as exc:
            raise RuntimeError("FiPy DT requires fipy; install digitaltwin/requirements.txt") from exc

        self.settings = settings
        self._cell_variable = CellVariable
        self._diffusion_term = DiffusionTerm
        self._transient_term = TransientTerm
        self.width_m = build_width_mm / 1000.0
        self.depth_m = build_depth_mm / 1000.0
        target_cell_m = resolution_mm / 1000.0
        nx = max(2, int(ceil(self.width_m / target_cell_m)))
        ny = max(2, int(ceil(self.depth_m / target_cell_m)))
        if nx * ny > max_cells:
            scale = sqrt(nx * ny / max_cells)
            nx = max(2, int(nx / scale))
            ny = max(2, int(ny / scale))
        self.nx, self.ny = nx, ny
        self.dx_m = self.width_m / nx
        self.dy_m = self.depth_m / ny
        self.max_substeps_per_vector = max_substeps_per_vector
        mesh = Grid2D(dx=self.dx_m, dy=self.dy_m, nx=nx, ny=ny)

        # Values are aligned with the repository's 316L ROM assumptions.
        self.density_kg_m3 = 7900.0
        self.heat_capacity_j_kg_k = 650.0
        self.diffusivity_m2_s = 5e-6
        self.absorptivity = 0.35
        self.layer_depth_m = settings.process.layer_thickness_um * 1e-6
        self.preheat_k = settings.process.preheat_temperature_c + 273.15
        self.melt_threshold_k = 1700.0
        self.temperature = CellVariable(
            mesh=mesh,
            name="temperature",
            value=self.preheat_k,
            hasOld=True,
        )
        self.temperature.constrain(self.preheat_k, where=mesh.exteriorFaces)
        self.source = CellVariable(mesh=mesh, name="laser_source", value=0.0)
        self.mesh = mesh
        self._equation = TransientTerm() == DiffusionTerm(coeff=self.diffusivity_m2_s) + self.source

        centers = mesh.cellCenters.value
        self.cell_x_m = centers[0]
        self.cell_y_m = centers[1]
        self.cell_areas_m2 = mesh.cellVolumes

    def _advance_laser(self, x_mm: float, y_mm: float, power_w: float, spot_mm: float, dt: float):
        x_m = x_mm / 1000.0 + self.width_m / 2
        y_m = y_mm / 1000.0 + self.depth_m / 2
        sigma_m = max(spot_mm / 2000.0, min(self.dx_m, self.dy_m) / 2)
        weights = np.exp(
            -0.5
            * (
                ((self.cell_x_m - x_m) / sigma_m) ** 2
                + ((self.cell_y_m - y_m) / sigma_m) ** 2
            )
        )
        normalization = float(np.sum(weights * self.cell_areas_m2))
        if normalization <= 0 or not isfinite(normalization):
            return
        absorbed_power = self.absorptivity * power_w
        source_k_s = (
            absorbed_power
            * weights
            / (
                self.density_kg_m3
                * self.heat_capacity_j_kg_k
                * self.layer_depth_m
                * normalization
            )
        )
        self.source.setValue(source_k_s)
        self.temperature.updateOld()
        self._equation.solve(var=self.temperature, dt=dt)

    def consume(self, event: BuildEvent) -> ThermalSnapshot:
        process = event.process
        start_x, start_y, _ = event.start_mm
        end_x, end_y, _ = event.end_mm
        length_mm = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
        spot_mm = process["spot_diameter_mm"]
        sample_step_mm = max(spot_mm, min(self.dx_m, self.dy_m) * 1000)
        substeps = max(1, min(self.max_substeps_per_vector, int(ceil(length_mm / sample_step_mm))))
        dt = max(event.duration_s / substeps, 1e-9)
        for index in range(substeps):
            fraction = (index + 0.5) / substeps
            x_mm = start_x + fraction * (end_x - start_x)
            y_mm = start_y + fraction * (end_y - start_y)
            self._advance_laser(
                x_mm,
                y_mm,
                process["laser_power_w"],
                spot_mm,
                dt,
            )

        temperatures_k = self.temperature.value
        peak_k = float(np.max(temperatures_k))
        return ThermalSnapshot(
            event_id=event.event_id,
            peak_temperature_c=peak_k - 273.15,
            hot_cell_count=int(np.count_nonzero(temperatures_k > self.preheat_k + 25.0)),
            melt_threshold_cell_count=int(np.count_nonzero(temperatures_k >= self.melt_threshold_k)),
            grid_shape=(self.nx, self.ny),
            cell_size_mm=(self.dx_m * 1000, self.dy_m * 1000),
        )