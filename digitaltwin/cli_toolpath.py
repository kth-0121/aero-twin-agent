"""Read and write the ASCII Common Layer Interface subset used by the demo."""

from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path
from typing import Iterable

Point2D = tuple[float, float]


@dataclass(frozen=True)
class LayerPaths:
    layer_index: int
    z_mm: float
    contours: tuple[tuple[Point2D, ...], ...]
    hatches: tuple[tuple[Point2D, Point2D], ...]


@dataclass(frozen=True)
class PathSegment:
    layer_index: int
    z_mm: float
    start_mm: Point2D
    end_mm: Point2D
    path_type: str

    @property
    def length_mm(self) -> float:
        return hypot(
            self.end_mm[0] - self.start_mm[0],
            self.end_mm[1] - self.start_mm[1],
        )


@dataclass(frozen=True)
class CLIJob:
    label: str
    units_mm: float
    segments: tuple[PathSegment, ...]

    @property
    def layer_count(self) -> int:
        return len({segment.layer_index for segment in self.segments})


def _polyline_record(
    points: Iterable[Point2D], *, closed: bool, units_mm: float
) -> str:
    coordinates = tuple(points)
    if closed and len(coordinates) < 3:
        raise ValueError("closed contour needs at least three points")
    if not closed and len(coordinates) < 2:
        raise ValueError("open polyline needs at least two points")
    scaled = []
    for x_mm, y_mm in coordinates:
        if not isfinite(x_mm) or not isfinite(y_mm):
            raise ValueError("CLI coordinates must be finite")
        scaled.extend((x_mm / units_mm, y_mm / units_mm))
    encoded = ",".join(f"{value:.6f}" for value in scaled)
    return f"$$POLYLINE/0,{int(closed)},0,{len(coordinates)},{encoded}"


def write_cli(
    path: str | Path,
    layers: Iterable[LayerPaths],
    *,
    label: str = "AERO-TWIN-316L-DEMO",
    units_mm: float = 1.0,
) -> Path:
    """Write contours and hatch vectors using millimeters as the source unit."""
    if not isfinite(units_mm) or units_mm <= 0:
        raise ValueError("units_mm must be finite and positive")
    safe_label = label.replace("/", "-").replace("\n", " ").replace("\r", " ")
    lines = [
        "$$HEADERSTART",
        "$$ASCII",
        f"$$UNITS/{units_mm:.6f}",
        "$$VERSION/200",
        f"$$LABEL/{safe_label}",
        "$$HEADEREND",
        "$$GEOMETRYSTART",
    ]
    for layer in layers:
        if not isfinite(layer.z_mm) or layer.z_mm < 0:
            raise ValueError("layer z_mm must be finite and non-negative")
        lines.append(f"$$LAYER/{layer.z_mm / units_mm:.6f}")
        for contour in layer.contours:
            lines.append(_polyline_record(contour, closed=True, units_mm=units_mm))
        for start, end in layer.hatches:
            lines.append(_polyline_record((start, end), closed=False, units_mm=units_mm))
        lines.append("$$LAYEREND")
    lines.append("$$GEOMETRYEND")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output_path


def read_cli(path: str | Path) -> CLIJob:
    """Parse ASCII CLI layers and expand polylines into ordered scan segments."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    units_mm = None
    label = Path(path).stem
    geometry_started = False
    active_layer = None
    layer_index = -1
    segments: list[PathSegment] = []

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or not line.startswith("$$"):
            continue
        if line == "$$GEOMETRYSTART":
            geometry_started = True
        elif line.startswith("$$UNITS/"):
            try:
                units_mm = float(line.split("/", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid CLI units at line {line_number}") from exc
        elif line.startswith("$$LABEL/"):
            label = line.split("/", 1)[1]
        elif line.startswith("$$LAYER/"):
            if not geometry_started:
                raise ValueError(f"layer appears before geometry at line {line_number}")
            try:
                z_value = float(line.split("/", 1)[1])
            except ValueError as exc:
                raise ValueError(f"invalid layer height at line {line_number}") from exc
            if not isfinite(z_value) or z_value < 0:
                raise ValueError(f"invalid layer height at line {line_number}")
            layer_index += 1
            active_layer = (layer_index, z_value)
        elif line == "$$LAYEREND":
            active_layer = None
        elif line.startswith("$$POLYLINE/"):
            if active_layer is None:
                raise ValueError(f"polyline outside a layer at line {line_number}")
            try:
                values = line.split("/", 1)[1].split(",")
                closed = int(values[1])
                point_count = int(values[3])
                coordinates = [float(value) for value in values[4:]]
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid polyline at line {line_number}") from exc
            if closed not in (0, 1) or point_count < (3 if closed else 2):
                raise ValueError(f"invalid polyline flags/count at line {line_number}")
            if len(coordinates) != point_count * 2 or not all(
                isfinite(value) for value in coordinates
            ):
                raise ValueError(f"invalid polyline coordinates at line {line_number}")
            points = [
                (coordinates[index] * (units_mm or 0), coordinates[index + 1] * (units_mm or 0))
                for index in range(0, len(coordinates), 2)
            ]
            kind = "contour" if closed else "hatch"
            pairs = list(zip(points, points[1:]))
            if closed:
                pairs.append((points[-1], points[0]))
            layer_id, z_value = active_layer
            segments.extend(
                PathSegment(layer_id, z_value * (units_mm or 0), start, end, kind)
                for start, end in pairs
                if start != end
            )

    if units_mm is None or not isfinite(units_mm) or units_mm <= 0:
        raise ValueError("CLI file must declare positive $$UNITS")
    if not geometry_started:
        raise ValueError("CLI file is missing $$GEOMETRYSTART")
    if not segments:
        raise ValueError("CLI file contains no scan vectors")
    return CLIJob(label=label, units_mm=units_mm, segments=tuple(segments))