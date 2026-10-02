"""Build configuration contracts for the 316L LPBF demonstration."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class PowderProfile:
    material: str
    lot_id: str
    particle_d10_um: float
    particle_d50_um: float
    particle_d90_um: float
    oxygen_ppm: float
    reuse_count: int
    data_origin: str

    def validate(self) -> None:
        if not self.material.strip() or not self.lot_id.strip():
            raise ValueError("powder material and lot_id are required")
        sizes = (self.particle_d10_um, self.particle_d50_um, self.particle_d90_um)
        if not all(isfinite(value) and value > 0 for value in sizes):
            raise ValueError("powder particle sizes must be finite and positive")
        if not self.particle_d10_um <= self.particle_d50_um <= self.particle_d90_um:
            raise ValueError("powder sizes must satisfy D10 <= D50 <= D90")
        if not isfinite(self.oxygen_ppm) or self.oxygen_ppm < 0:
            raise ValueError("powder oxygen_ppm must be finite and non-negative")
        if self.reuse_count < 0:
            raise ValueError("powder reuse_count must be non-negative")
        if not self.data_origin.strip():
            raise ValueError("powder data_origin is required")


@dataclass(frozen=True)
class ProcessSettings:
    laser_power_w: float
    scan_speed_mm_s: float
    hatch_spacing_mm: float
    layer_thickness_um: float
    spot_diameter_mm: float
    preheat_temperature_c: float

    def validate(self) -> None:
        positive = {
            "laser_power_w": self.laser_power_w,
            "scan_speed_mm_s": self.scan_speed_mm_s,
            "hatch_spacing_mm": self.hatch_spacing_mm,
            "layer_thickness_um": self.layer_thickness_um,
            "spot_diameter_mm": self.spot_diameter_mm,
        }
        for name, value in positive.items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.preheat_temperature_c) or self.preheat_temperature_c < 0:
            raise ValueError("preheat_temperature_c must be finite and non-negative")


@dataclass(frozen=True)
class SupportSettings:
    strategy: str
    density_percent: float
    contact_diameter_mm: float
    removal_access_required: bool = True

    def validate(self) -> None:
        if self.strategy not in {"none", "manual", "external_slicer"}:
            raise ValueError("support strategy must be none, manual, or external_slicer")
        if not isfinite(self.density_percent) or not 0 <= self.density_percent <= 100:
            raise ValueError("support density_percent must be between 0 and 100")
        if not isfinite(self.contact_diameter_mm) or self.contact_diameter_mm < 0:
            raise ValueError("support contact_diameter_mm must be finite and non-negative")


@dataclass(frozen=True)
class ChamberSettings:
    shielding_gas: str
    oxygen_limit_ppm: float
    oxygen_measured_ppm: float

    def validate(self) -> None:
        if self.shielding_gas not in {"argon", "nitrogen"}:
            raise ValueError("shielding_gas must be argon or nitrogen")
        for name, value in (
            ("oxygen_limit_ppm", self.oxygen_limit_ppm),
            ("oxygen_measured_ppm", self.oxygen_measured_ppm),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def purge_complete(self) -> bool:
        return self.oxygen_measured_ppm <= self.oxygen_limit_ppm


@dataclass(frozen=True)
class MachineSettings:
    machine_id: str
    build_width_mm: float
    build_depth_mm: float
    build_height_mm: float

    def validate(self) -> None:
        if not self.machine_id.strip():
            raise ValueError("machine_id is required")
        for name, value in (
            ("build_width_mm", self.build_width_mm),
            ("build_depth_mm", self.build_depth_mm),
            ("build_height_mm", self.build_height_mm),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class BuildSettings:
    job_id: str
    material: str
    powder: PowderProfile
    process: ProcessSettings
    supports: SupportSettings
    chamber: ChamberSettings
    machine: MachineSettings

    def validate(self) -> None:
        if not self.job_id.strip() or not self.material.strip():
            raise ValueError("job_id and material are required")
        if self.material != self.powder.material:
            raise ValueError("build material must match the powder material")
        self.powder.validate()
        self.process.validate()
        self.supports.validate()
        self.chamber.validate()
        self.machine.validate()


def demo_316l_settings() -> BuildSettings:
    """Return one explicitly illustrative aerospace LPBF job configuration."""
    return BuildSettings(
        job_id="AERO-BRACKET-316L-DEMO",
        material="316L",
        powder=PowderProfile(
            material="316L",
            lot_id="DEMO-LOT-001",
            particle_d10_um=15.0,
            particle_d50_um=30.0,
            particle_d90_um=45.0,
            oxygen_ppm=500.0,
            reuse_count=0,
            data_origin="illustrative demo assumption; replace with powder certificate",
        ),
        process=ProcessSettings(
            laser_power_w=200.0,
            scan_speed_mm_s=800.0,
            hatch_spacing_mm=0.1,
            layer_thickness_um=30.0,
            spot_diameter_mm=0.1,
            preheat_temperature_c=100.0,
        ),
        supports=SupportSettings(
            strategy="manual",
            density_percent=20.0,
            contact_diameter_mm=0.5,
        ),
        chamber=ChamberSettings(
            shielding_gas="argon",
            oxygen_limit_ppm=100.0,
            oxygen_measured_ppm=80.0,
        ),
        machine=MachineSettings(
            machine_id="LPBF-DEMO-01",
            build_width_mm=250.0,
            build_depth_mm=250.0,
            build_height_mm=300.0,
        ),
    )


def load_build_settings(path: str | Path) -> BuildSettings:
    """Load a complete machine/material profile from JSON and validate it."""
    profile_path = Path(path)
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("build profile root must be a JSON object")

    def section(name: str) -> dict:
        value = data.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"build profile must contain object '{name}'")
        return value

    def text(values: dict, name: str) -> str:
        value = values.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"profile field '{name}' must be a non-empty string")
        return value

    def number(values: dict, name: str) -> float:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"profile field '{name}' must be a number")
        return float(value)

    powder = section("powder")
    process = section("process")
    supports = section("supports")
    chamber = section("chamber")
    machine = section("machine")
    reuse_count = powder.get("reuse_count")
    removal_access_required = supports.get("removal_access_required")
    if isinstance(reuse_count, bool) or not isinstance(reuse_count, int):
        raise ValueError("profile field 'reuse_count' must be an integer")
    if not isinstance(removal_access_required, bool):
        raise ValueError("profile field 'removal_access_required' must be a boolean")

    settings = BuildSettings(
        job_id=text(data, "job_id"),
        material=text(data, "material"),
        powder=PowderProfile(
            material=text(powder, "material"),
            lot_id=text(powder, "lot_id"),
            particle_d10_um=number(powder, "particle_d10_um"),
            particle_d50_um=number(powder, "particle_d50_um"),
            particle_d90_um=number(powder, "particle_d90_um"),
            oxygen_ppm=number(powder, "oxygen_ppm"),
            reuse_count=reuse_count,
            data_origin=text(powder, "data_origin"),
        ),
        process=ProcessSettings(
            laser_power_w=number(process, "laser_power_w"),
            scan_speed_mm_s=number(process, "scan_speed_mm_s"),
            hatch_spacing_mm=number(process, "hatch_spacing_mm"),
            layer_thickness_um=number(process, "layer_thickness_um"),
            spot_diameter_mm=number(process, "spot_diameter_mm"),
            preheat_temperature_c=number(process, "preheat_temperature_c"),
        ),
        supports=SupportSettings(
            strategy=text(supports, "strategy"),
            density_percent=number(supports, "density_percent"),
            contact_diameter_mm=number(supports, "contact_diameter_mm"),
            removal_access_required=removal_access_required,
        ),
        chamber=ChamberSettings(
            shielding_gas=text(chamber, "shielding_gas"),
            oxygen_limit_ppm=number(chamber, "oxygen_limit_ppm"),
            oxygen_measured_ppm=number(chamber, "oxygen_measured_ppm"),
        ),
        machine=MachineSettings(
            machine_id=text(machine, "machine_id"),
            build_width_mm=number(machine, "build_width_mm"),
            build_depth_mm=number(machine, "build_depth_mm"),
            build_height_mm=number(machine, "build_height_mm"),
        ),
    )
    settings.validate()
    return settings