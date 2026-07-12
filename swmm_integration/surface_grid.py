from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hydrographs import assign_tributary_cells
from .schemas import CaptureConfig, DestinationKind, SurfaceGrid, SurfaceSink


class SurfaceGridUnavailableError(RuntimeError):
    """El surface_grid.json exportado por el generador no está disponible."""


_PAYLOAD_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_TRIBUTARY_CACHE: dict[str, tuple[float, tuple[dict[str, list[int]], dict[int, DestinationKind]]]] = {}


def load_surface_grid_payload(path: str | Path) -> dict[str, Any]:
    """Carga el JSON exportado por export_surface_grid, cacheado por mtime."""
    grid_path = Path(path)
    if not grid_path.exists():
        raise SurfaceGridUnavailableError(
            f"No existe {grid_path.name}. Ejecute el ajuste de excavación "
            "(o el generador del mapa 3D) para exportar el grid de flujo."
        )
    mtime = grid_path.stat().st_mtime
    key = str(grid_path)
    cached = _PAYLOAD_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    payload = json.loads(grid_path.read_text(encoding="utf-8"))
    _PAYLOAD_CACHE[key] = (mtime, payload)
    return payload


def build_surface_grid(payload: dict[str, Any], capture: CaptureConfig | None = None) -> SurfaceGrid:
    """Convierte el payload exportado en un SurfaceGrid validado.

    La misma configuración de captura del escenario UI se aplica a todos los
    tragantes (el escenario actual no distingue captura por tragante).
    """
    legend = payload.get("coverage_legend") or ["base", "vegetation", "building", "street"]
    codes = payload.get("coverage_codes")
    coverage = [legend[code] for code in codes] if codes else None
    capture = capture or CaptureConfig()
    sinks = [
        SurfaceSink(
            id=sink["id"],
            cell_index=sink["cell_index"],
            lon=sink.get("lon"),
            lat=sink.get("lat"),
            capture=capture.model_copy(deep=True),
        )
        for sink in payload.get("sinks", [])
    ]
    return SurfaceGrid(
        nx=payload["nx"],
        ny=payload["ny"],
        cell_area_m2=payload["cell_area_m2"],
        receivers=payload["receivers"],
        coverage=coverage,
        sinks=sinks,
    )


def tributaries_for(path: str | Path, grid: SurfaceGrid) -> tuple[dict[str, list[int]], dict[int, DestinationKind]]:
    """assign_tributary_cells cacheado por (ruta, mtime).

    Es la parte cara a resolución de flujo (~435k celdas) y no depende de la
    configuración de captura, así que se paga una vez por regeneración del grid.
    """
    grid_path = Path(path)
    mtime = grid_path.stat().st_mtime if grid_path.exists() else -1.0
    key = str(grid_path)
    cached = _TRIBUTARY_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    result = assign_tributary_cells(grid)
    _TRIBUTARY_CACHE[key] = (mtime, result)
    return result
