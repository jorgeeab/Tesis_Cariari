import numpy as np
import pytest

from generar_dem_3d_threejs_satelital import (
    INLET_INFLUENCE_RADIUS_M,
    build_flow_hydrology,
    downsample_hydrology_data,
)


def _plano_inclinado_hacia_el_sur(n=9):
    """DEM indexado con fila 0 = sur (lats ascendente); el borde sur es el mas bajo."""
    lons = np.linspace(-84.004, -84.0, n)
    lats = np.linspace(10.0, 10.004, n)
    dem = np.repeat(np.arange(n, dtype=float)[:, None] * 5.0, n, axis=1)
    return dem, lons, lats


def test_sink_en_borde_sur_recibe_la_acumulacion_de_su_columna():
    dem, lons, lats = _plano_inclinado_hacia_el_sur()
    ny = len(lats)
    sinks = [{"id": "T1", "lon": float(lons[4]), "lat": float(lats[0])}]

    hyd = build_flow_hydrology(
        dem, lons, lats, lat_center=float(lats[4]), surface_sinks=sinks
    )

    assert hyd["sink_count"] == 1
    # Toda la columna (ny celdas) drena al tragante del borde sur; si el
    # mapeo quedara espejado N-S, el tragante caeria en el borde alto y
    # solo acumularia su propia celda (1.0).
    assert hyd["sink_loads"]["T1"] == pytest.approx(float(ny))


def test_sink_assignment_marca_la_columna_del_tragante():
    dem, lons, lats = _plano_inclinado_hacia_el_sur()
    ny, nx = len(lats), len(lons)
    sinks = [{"id": "T1", "lon": float(lons[4]), "lat": float(lats[0])}]

    hyd = build_flow_hydrology(
        dem, lons, lats, lat_center=float(lats[4]), surface_sinks=sinks
    )

    assert hyd["sink_ids"] == ["T1"]
    assign = np.array(hyd["sink_assignment"]).reshape(ny, nx)
    # La columna del tragante (grid norte-arriba) drena completa hacia él;
    # las demás columnas salen por el borde sin tragante (-1).
    assert (assign[:, 4] == 0).all()
    assert (assign[:, 0] == -1).all()
    # sink_areas se redondea a 1 decimal
    assert hyd["sink_areas"]["T1"] == pytest.approx(ny * hyd["cell_area_m2"], abs=0.1)


def test_downsample_conserva_subcuencas_pequenas():
    # Grid 2×2 → 1×1 (scale 2): el único tragante NO cae en la celda central
    # del bloque; el downsample debe conservarlo igualmente.
    hydro = {
        "accumulation": [0.1, 0.2, 0.3, 0.4],
        "arrow_dx": [0.0, 0.0, 0.0, 0.0],
        "arrow_dz": [0.0, 0.0, 0.0, 0.0],
        "arrow_strength": [0.0, 0.0, 0.0, 0.0],
        "max_accumulation": 4,
        "sink_loads": {"T1": 1.0},
        "sink_count": 1,
        "sink_assignment": [0, -1, -1, -1],
        "sink_ids": ["T1"],
        "sink_areas": {"T1": 100.0},
    }

    out = downsample_hydrology_data(hydro, 1, 1, 2)

    assert out["sink_assignment"] == [0]
    assert out["sink_ids"] == ["T1"]
    assert out["sink_areas"] == {"T1": 100.0}


def test_sink_en_borde_norte_solo_acumula_su_celda():
    dem, lons, lats = _plano_inclinado_hacia_el_sur()
    sinks = [{"id": "T9", "lon": float(lons[4]), "lat": float(lats[-1])}]

    hyd = build_flow_hydrology(
        dem, lons, lats, lat_center=float(lats[4]), surface_sinks=sinks
    )

    assert hyd["sink_loads"]["T9"] == pytest.approx(1.0)


def _grid_metrico_un_metro(n=17):
    """DEM con celdas de ~1 m y pendiente hacia el oeste."""
    lat0 = 10.0
    lon0 = -84.0
    mpd_lon = 111320.0 * np.cos(np.radians(lat0))
    mpd_lat = 110540.0
    lons = lon0 + np.arange(n, dtype=float) / mpd_lon
    lats = lat0 + np.arange(n, dtype=float) / mpd_lat
    dem = np.tile(np.arange(n, dtype=float)[None, :], (n, 1))
    return dem, lons, lats


def test_area_de_influencia_forza_celdas_cercanas_hacia_tragante():
    dem, lons, lats = _grid_metrico_un_metro()
    center = len(lons) // 2
    sinks = [{
        "id": "tragante_centro",
        "kind": "tragante",
        "lon": float(lons[center]),
        "lat": float(lats[center]),
    }]

    hyd = build_flow_hydrology(
        dem, lons, lats, lat_center=float(lats[center]), surface_sinks=sinks
    )

    assign = np.array(hyd["sink_assignment"]).reshape(len(lats), len(lons))
    sink_cell = hyd["sink_cells"]["tragante_centro"]
    sj, si = divmod(sink_cell, len(lons))
    # La celda al oeste del tragante naturalmente drenaria al borde; el area
    # de influencia la fuerza hacia el tragante.
    assert assign[sj, si - 2] == 0

    forced_count = 0
    for j in range(assign.shape[0]):
        for i in range(assign.shape[1]):
            distance_m = np.hypot(i - si, j - sj)
            if distance_m <= INLET_INFLUENCE_RADIUS_M - 0.1:
                forced_count += 1
                assert assign[j, i] == 0
    assert forced_count > 20


def test_tragantes_cercanos_dividen_area_de_influencia_por_cercania():
    dem, lons, lats = _grid_metrico_un_metro()
    row = len(lats) // 2
    sinks = [
        {
            "id": "tragante_oeste",
            "kind": "tragante",
            "lon": float(lons[6]),
            "lat": float(lats[row]),
        },
        {
            "id": "tragante_este",
            "kind": "tragante",
            "lon": float(lons[10]),
            "lat": float(lats[row]),
        },
    ]

    hyd = build_flow_hydrology(
        dem, lons, lats, lat_center=float(lats[row]), surface_sinks=sinks
    )

    assign = np.array(hyd["sink_assignment"]).reshape(len(lats), len(lons))
    sink_ids = hyd["sink_ids"]
    oeste = sink_ids.index("tragante_oeste")
    este = sink_ids.index("tragante_este")
    sink_cell_oeste = hyd["sink_cells"]["tragante_oeste"]
    sj, si = divmod(sink_cell_oeste, len(lons))

    assert assign[sj, si + 1] == oeste
    assert assign[sj, si + 3] == este
