"""Convert a watertight STL mesh into demo-oriented ASCII CLI scan vectors."""

from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
from typing import Callable

from .cli_toolpath import LayerPaths, write_cli


_UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4}


@dataclass(frozen=True)
class STLConversionResult:
    cli_path: Path
    source_units: str
    layer_count: int
    scan_vector_count: int
    build_width_mm: float
    build_depth_mm: float
    build_height_mm: float


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [line for child in geometry.geoms for line in _line_parts(child)]
    return []


def _hatch_polygon(polygon, spacing_mm: float, angle_degrees: float):
    from shapely.affinity import rotate
    from shapely.geometry import LineString

    rotated = rotate(polygon, angle_degrees, origin=(0.0, 0.0), use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    first_y = min_y + spacing_mm / 2
    scan_count = max(0, int(ceil((max_y - first_y) / spacing_mm)))
    hatches = []
    for scan_index in range(scan_count):
        y = first_y + scan_index * spacing_mm
        intersections = _line_parts(
            rotated.intersection(LineString(((min_x - spacing_mm, y), (max_x + spacing_mm, y))))
        )
        intersections.sort(key=lambda line: line.bounds[0])
        if scan_index % 2:
            intersections.reverse()
        for line in intersections:
            coordinates = list(line.coords)
            start = coordinates[0]
            end = coordinates[-1]
            if scan_index % 2:
                start, end = end, start
            unrotated = rotate(
                LineString((start, end)),
                -angle_degrees,
                origin=(0.0, 0.0),
                use_radians=False,
            )
            points = list(unrotated.coords)
            if points[0] != points[-1]:
                hatches.append((tuple(points[0]), tuple(points[-1])))
    return hatches


def slice_stl_to_cli(
    stl_path: str | Path,
    cli_path: str | Path,
    *,
    layer_thickness_um: float,
    hatch_spacing_mm: float,
    source_units: str = "mm",
    label: str = "AERO-TWIN-316L-DEMO",
    progress_callback: Callable[[int, int], None] | None = None,
) -> STLConversionResult:
    """Slice a closed STL along +Z and generate alternating raster hatch paths.

    STL has no reliable unit metadata. ``source_units`` is therefore an explicit
    caller assumption; the generated CLI is always expressed in millimeters.
    The part is centered on the XY build plate and translated to Z >= 0.
    """
    try:
        unit_scale = _UNIT_TO_MM[source_units]
    except KeyError as exc:
        raise ValueError(f"source_units must be one of {tuple(_UNIT_TO_MM)}") from exc
    if not isfinite(layer_thickness_um) or layer_thickness_um <= 0:
        raise ValueError("layer_thickness_um must be finite and positive")
    if not isfinite(hatch_spacing_mm) or hatch_spacing_mm <= 0:
        raise ValueError("hatch_spacing_mm must be finite and positive")

    try:
        import trimesh
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise RuntimeError(
            "STL slicing requires trimesh and shapely; install digitaltwin/requirements.txt"
        ) from exc

    mesh = trimesh.load_mesh(stl_path, force="mesh", process=True)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError("STL contains no triangles")
    if not mesh.is_watertight:
        raise ValueError("STL must be watertight before slicing")
    mesh.apply_scale(unit_scale)
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    if not all(isfinite(float(value)) and value > 0 for value in dimensions):
        raise ValueError("STL must have finite, non-zero dimensions on all axes")

    mesh.apply_translation(
        (-float((bounds[0][0] + bounds[1][0]) / 2),
         -float((bounds[0][1] + bounds[1][1]) / 2),
         -float(bounds[0][2]))
    )
    height_mm = float(dimensions[2])
    layer_thickness_mm = layer_thickness_um / 1000.0
    layer_count = max(1, int(ceil(height_mm / layer_thickness_mm)))
    layers = []
    vector_count = 0
    if progress_callback is not None:
        progress_callback(0, layer_count)
    for layer_index in range(layer_count):
        z_mm = min(
            (layer_index + 0.5) * layer_thickness_mm,
            max(height_mm / 2, height_mm - layer_thickness_mm / 2),
        )
        section = mesh.section(
            plane_origin=(0.0, 0.0, z_mm),
            plane_normal=(0.0, 0.0, 1.0),
        )
        if section is None:
            if progress_callback is not None:
                progress_callback(layer_index + 1, layer_count)
            continue

        contours = []
        polygonal_regions = []
        for discrete in section.discrete:
            coordinates = [(float(point[0]), float(point[1])) for point in discrete]
            if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
                coordinates.pop()
            if len(coordinates) < 3:
                continue
            polygon = Polygon(coordinates)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            contours.append(tuple(coordinates))
            polygonal_regions.append(polygon)
        if not polygonal_regions:
            if progress_callback is not None:
                progress_callback(layer_index + 1, layer_count)
            continue

        # Even-odd composition preserves holes represented by nested mesh loops.
        material_region = polygonal_regions[0]
        for polygon in polygonal_regions[1:]:
            material_region = material_region.symmetric_difference(polygon)
        if not material_region.is_valid:
            material_region = material_region.buffer(0)

        hatch_angle = 0.0 if layer_index % 2 == 0 else 90.0
        hatches = _hatch_polygon(material_region, hatch_spacing_mm, hatch_angle)
        vector_count += sum(max(0, len(contour)) for contour in contours) + len(hatches)
        layers.append(
            LayerPaths(
                layer_index=layer_index,
                z_mm=z_mm,
                contours=tuple(contours),
                hatches=tuple(hatches),
            )
        )
        if progress_callback is not None:
            progress_callback(layer_index + 1, layer_count)

    if not layers:
        raise ValueError("STL produced no closed contours at the requested layer thickness")
    output_path = write_cli(cli_path, layers, label=label)
    return STLConversionResult(
        cli_path=output_path,
        source_units=source_units,
        layer_count=len(layers),
        scan_vector_count=vector_count,
        build_width_mm=float(dimensions[0]),
        build_depth_mm=float(dimensions[1]),
        build_height_mm=height_mm,
    )