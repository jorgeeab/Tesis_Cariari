from __future__ import annotations

from collections import Counter
from uuid import uuid4

from .rainfall import build_rainfall_intervals
from .schemas import (
    CaptureConfig,
    DestinationKind,
    HydrographPoint,
    HydrographResult,
    HydrologyRunRequest,
    InfiltrationConfig,
    SurfaceGrid,
    TributaryArea,
)


def _coverage_for_cell(grid: SurfaceGrid, idx: int) -> str:
    if grid.coverage is None:
        return "base"
    return grid.coverage[idx]


def assign_tributary_cells(grid: SurfaceGrid) -> tuple[dict[str, list[int]], dict[int, DestinationKind]]:
    """Assign each cell to one terminal destination by following D8 receivers.

    receivers uses flattened cell indexes. A value of -1 means the cell exits
    the modeled domain or has no valid downstream receiver. Sink cells are
    terminal and take priority over receiver traversal.
    """
    sink_by_cell = {sink.cell_index: sink.id for sink in grid.sinks}
    tributaries = {sink.id: [] for sink in grid.sinks}
    destinations: dict[int, DestinationKind] = {}
    destination_sink: dict[int, str | None] = {}
    total = grid.nx * grid.ny

    for start in range(total):
        path: list[int] = []
        seen: set[int] = set()
        current = start
        terminal_kind: DestinationKind
        terminal_sink: str | None = None

        while True:
            if current in destination_sink:
                terminal_sink = destination_sink[current]
                terminal_kind = destinations[current]
                break
            if current in sink_by_cell:
                terminal_sink = sink_by_cell[current]
                terminal_kind = "sink"
                break
            if current in seen:
                terminal_kind = "loop"
                break
            seen.add(current)
            path.append(current)
            receiver = grid.receivers[current]
            if receiver == -1:
                terminal_kind = "border"
                break
            current = receiver

        for idx in path:
            destinations[idx] = terminal_kind
            destination_sink[idx] = terminal_sink
        if start in sink_by_cell:
            destinations[start] = "sink"
            destination_sink[start] = sink_by_cell[start]

        if terminal_sink:
            for idx in path:
                tributaries[terminal_sink].append(idx)
            if start in sink_by_cell and start not in tributaries[terminal_sink]:
                tributaries[terminal_sink].append(start)

    for cells in tributaries.values():
        cells.sort()
    return tributaries, destinations


def _captured_flow(flow_m3s: float, capture: CaptureConfig) -> tuple[float, float]:
    if not capture.active:
        return 0.0, flow_m3s
    if capture.capture_mode == "complete":
        return flow_m3s, 0.0
    capacity = capture.effective_capacity_m3s()
    assert capacity is not None
    captured = min(flow_m3s, capacity)
    bypass = max(0.0, flow_m3s - captured)
    return captured, bypass


def generate_hydrographs(
    request: HydrologyRunRequest,
    precomputed_tributaries: tuple[dict[str, list[int]], dict[int, DestinationKind]] | None = None,
) -> HydrographResult:
    """Generate physical sink hydrographs from rainfall, D8 receivers and cover.

    The normalized accumulation layer remains visual only. This function uses
    physical volume per cell:
        V_rain = (P_mm / 1000) * cell_area_m2
        V_runoff = V_rain * (1 - infiltration_fraction)

    precomputed_tributaries allows reusing a cached assign_tributary_cells
    result for large grids; it must correspond to the same surface grid.
    """
    grid = request.surface
    intervals = build_rainfall_intervals(request.rainfall)
    if precomputed_tributaries is not None:
        tributaries, destinations = precomputed_tributaries
    else:
        tributaries, destinations = assign_tributary_cells(grid)

    hydrographs: dict[str, list[HydrographPoint]] = {}
    tributary_areas: list[TributaryArea] = []
    warnings: list[str] = []
    sink_by_id = {sink.id: sink for sink in grid.sinks}

    total_rain_m3 = 0.0
    total_infiltrated_m3 = 0.0
    total_runoff_m3 = 0.0
    total_captured_m3 = 0.0
    total_bypass_m3 = 0.0
    total_surface_exit_m3 = 0.0
    total_storage_m3 = 0.0

    for sink_id, cells in tributaries.items():
        area_m2 = len(cells) * grid.cell_area_m2
        if not cells:
            warnings.append(f"Tragante sin area tributaria: {sink_id}")
        tributary_areas.append(
            TributaryArea(
                sink_id=sink_id,
                cell_count=len(cells),
                area_m2=area_m2,
                cell_indices=cells,
            )
        )

    sink_ids = [sink.id for sink in grid.sinks]
    cell_to_sink = {
        cell_index: sink_id
        for sink_id, cells in tributaries.items()
        for cell_index in cells
    }
    sink_storage_remaining = {
        sink.id: sink.capture.surface_storage_m3
        for sink in grid.sinks
    }
    for sink_id in sink_ids:
        hydrographs[sink_id] = []

    all_cells = range(grid.nx * grid.ny)
    for interval in intervals:
        sink_volumes = {sink_id: 0.0 for sink_id in sink_ids}
        surface_exit_volume = 0.0
        for idx in all_cells:
            depth_m = interval.depth_mm / 1000.0
            rain_volume = depth_m * grid.cell_area_m2
            total_rain_m3 += rain_volume
            coverage = _coverage_for_cell(grid, idx)
            infiltration_fraction = float(getattr(request.infiltration, coverage))
            infiltrated = rain_volume * infiltration_fraction
            runoff = rain_volume - infiltrated
            total_infiltrated_m3 += infiltrated
            total_runoff_m3 += runoff

            destination = destinations.get(idx, "no_valid_destination")
            if destination == "sink":
                sink_id = cell_to_sink.get(idx)
                if sink_id is not None:
                    sink_volumes[sink_id] += runoff
            else:
                surface_exit_volume += runoff

        total_surface_exit_m3 += surface_exit_volume
        for sink_id, volume in sink_volumes.items():
            flow = volume / interval.duration_s if interval.duration_s > 0 else 0.0
            capture = sink_by_id[sink_id].capture
            captured_flow, bypass_flow = _captured_flow(flow, capture)
            captured_volume = captured_flow * interval.duration_s
            bypass_volume = bypass_flow * interval.duration_s
            if not capture.allow_bypass and bypass_volume > 0:
                stored_volume = min(bypass_volume, sink_storage_remaining[sink_id])
                sink_storage_remaining[sink_id] -= stored_volume
                total_storage_m3 += stored_volume
                bypass_volume -= stored_volume
                bypass_flow = bypass_volume / interval.duration_s if interval.duration_s > 0 else 0.0
                if bypass_volume > 0:
                    warnings.append(
                        f"El tragante {sink_id} excedio su almacenamiento superficial; el excedente se reporta como bypass."
                    )
            total_captured_m3 += captured_volume
            total_bypass_m3 += bypass_volume
            hydrographs[sink_id].append(
                HydrographPoint(
                    minute=interval.start_min,
                    volume_m3=volume,
                    flow_m3s=flow,
                    captured_volume_m3=captured_volume,
                    captured_flow_m3s=captured_flow,
                    bypass_volume_m3=bypass_volume,
                    bypass_flow_m3s=bypass_flow,
                )
            )

    destination_counts = Counter(destinations.values())
    for kind in ("sink", "surface_outlet", "border", "no_valid_destination", "loop"):
        destination_counts.setdefault(kind, 0)

    mass_balance = {
        "rainfall_m3": total_rain_m3,
        "infiltrated_m3": total_infiltrated_m3,
        "runoff_m3": total_runoff_m3,
        "captured_m3": total_captured_m3,
        "bypass_m3": total_bypass_m3,
        "surface_exit_m3": total_surface_exit_m3,
        "storage_m3": total_storage_m3,
    }
    lhs = mass_balance["rainfall_m3"]
    rhs = (
        mass_balance["infiltrated_m3"]
        + mass_balance["captured_m3"]
        + mass_balance["bypass_m3"]
        + mass_balance["surface_exit_m3"]
        + mass_balance["storage_m3"]
    )
    mass_balance["error_pct"] = abs(lhs - rhs) / lhs * 100.0 if lhs > 0 else 0.0

    return HydrographResult(
        hydrology_id=str(uuid4()),
        rainfall=request.rainfall,
        infiltration=request.infiltration,
        hydrographs=hydrographs,
        tributary_areas=tributary_areas,
        destination_counts=dict(destination_counts),
        mass_balance=mass_balance,
        warnings=warnings,
    )
