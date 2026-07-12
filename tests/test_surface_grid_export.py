import json

import numpy as np

from generar_dem_3d_threejs_satelital import build_flow_hydrology, export_surface_grid
from swmm_integration.schemas import SurfaceGrid, SurfaceSink


def _setup(n=6):
    """DEM inclinado hacia el sur (fila 0 = sur) con capas de cobertura."""
    lons = np.linspace(-84.004, -84.0, n)
    lats = np.linspace(10.0, 10.004, n)
    dem = np.repeat(np.arange(n, dtype=float)[:, None] * 5.0, n, axis=1)
    surface_layers = {
        "vegetation": np.zeros((n, n), dtype=np.float32),
        "buildings": np.zeros((n, n), dtype=np.float32),
        "streets": np.zeros((n, n), dtype=np.float32),
    }
    surface_layers["vegetation"][0, :] = 1.0   # borde sur vegetado
    surface_layers["streets"][0, 0] = 1.0      # esquina SW es calle (prioridad)
    return dem, lons, lats, surface_layers


def test_export_produce_surface_grid_valido(tmp_path):
    dem, lons, lats, surface_layers = _setup()
    n = len(lats)
    sinks = [
        {"id": "T1", "lon": float(lons[2]), "lat": float(lats[0])},
        {"id": "T2", "lon": float(lons[4]), "lat": float(lats[0])},
    ]
    hyd = build_flow_hydrology(dem, lons, lats, float(lats[2]), surface_sinks=sinks)
    out = export_surface_grid(hyd, surface_layers, sinks, scale=1,
                              out_path=tmp_path / "surface_grid.json",
                              params={"street_depth": 0.5})

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["nx"] == n and payload["ny"] == n
    assert len(payload["receivers"]) == n * n
    assert len(payload["coverage_codes"]) == n * n
    assert payload["cell_area_m2"] > 0
    assert payload["params"]["street_depth"] == 0.5

    # Valida contra el esquema pydantic del servidor.
    legend = payload["coverage_legend"]
    grid = SurfaceGrid(
        nx=payload["nx"],
        ny=payload["ny"],
        cell_area_m2=payload["cell_area_m2"],
        receivers=payload["receivers"],
        coverage=[legend[c] for c in payload["coverage_codes"]],
        sinks=[SurfaceSink(**s) for s in payload["sinks"]],
    )
    assert {s.id for s in grid.sinks} == {"T1", "T2"}

    # Los tragantes del borde sur deben caer en la última fila del grid
    # norte-arriba (celdas >= (n-1)*n).
    for sink in grid.sinks:
        assert sink.cell_index >= (n - 1) * n

    # Coberturas volteadas a norte-arriba: el borde sur (última fila) es
    # vegetación, salvo la esquina SW que es calle.
    codes = np.array(payload["coverage_codes"]).reshape(n, n)
    assert codes[n - 1, 0] == 3
    assert codes[n - 1, 1] == 1
    assert codes[0, 0] == 0


def test_export_deduplica_tragantes_en_misma_celda(tmp_path):
    dem, lons, lats, surface_layers = _setup()
    sinks = [
        {"id": "T1", "lon": float(lons[2]), "lat": float(lats[0])},
        {"id": "T1b", "lon": float(lons[2]), "lat": float(lats[0])},
    ]
    hyd = build_flow_hydrology(dem, lons, lats, float(lats[2]), surface_sinks=sinks)
    out = export_surface_grid(hyd, surface_layers, sinks, scale=1,
                              out_path=tmp_path / "surface_grid.json")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["sinks"]) == 1
    assert payload["sinks"][0]["id"] == "T1"
    assert payload["warnings"]
