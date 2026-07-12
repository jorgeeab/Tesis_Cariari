import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from swmm_integration import PySwmmUnavailableError
from swmm_integration.schemas import NodeResultPoint, SwmmRunResult


client = TestClient(app_module.app)


def _write_grid(path: Path) -> None:
    payload = {
        "version": 1,
        "generated_at": "2026-07-07T00:00:00Z",
        "params": {"street_depth": 1.5},
        "nx": 2,
        "ny": 1,
        "cell_area_m2": 100.0,
        "receivers": [1, -1],
        "coverage_codes": [0, 0],
        "coverage_legend": ["base", "vegetation", "building", "street"],
        "sinks": [{"id": "T1", "cell_index": 1, "lon": -84.0, "lat": 10.0}],
        "warnings": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def sim_env(tmp_path, monkeypatch):
    grid_path = tmp_path / "surface_grid.json"
    _write_grid(grid_path)
    monkeypatch.setattr(app_module, "SURFACE_GRID_PATH", grid_path)
    monkeypatch.setattr(app_module, "UI_SCENARIO_PATH", tmp_path / "ui_scenario.json")
    monkeypatch.setattr(app_module, "LATEST_SIMULATION_ID", None)
    app_module.SIMULATION_FRAMES.clear()
    return grid_path


def test_grid_status_reporta_faltante_y_disponible(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "SURFACE_GRID_PATH", tmp_path / "no_existe.json")
    body = client.get("/api/simulation/grid-status").json()
    assert body["available"] is False
    assert "excavaci" in body["detail"]

    grid_path = tmp_path / "surface_grid.json"
    _write_grid(grid_path)
    monkeypatch.setattr(app_module, "SURFACE_GRID_PATH", grid_path)
    body = client.get("/api/simulation/grid-status").json()
    assert body["available"] is True
    assert body["nx"] == 2 and body["ny"] == 1
    assert body["sink_count"] == 1


def test_simulation_run_404_sin_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "SURFACE_GRID_PATH", tmp_path / "no_existe.json")
    response = client.post("/api/simulation/run", json={})
    assert response.status_code == 404


def test_simulation_run_degradada_sin_pyswmm(sim_env, monkeypatch):
    def raise_unavailable(request):
        raise PySwmmUnavailableError("PySWMM no esta instalado")

    monkeypatch.setattr(app_module.swmm_service, "run_simulation", raise_unavailable)

    response = client.post("/api/simulation/run", json={})
    assert response.status_code == 200
    body = response.json()

    assert body["swmm_available"] is False
    assert "PySWMM" in body["swmm_error"]
    # Escenario por defecto: 30 min cada 5 min → frames 0..30 (7).
    assert body["times_min"] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    assert len(body["rain_mm_h"]) == 7
    assert body["rain_mm_h"][0] == 60.0
    assert body["rain_mm_h"][-1] == 0.0
    assert body["cum_rain_frac"][-1] == 1.0

    node = body["nodes"]["T1"]
    assert node["source"] == "hydrology"
    assert node["status"][0] == "hydrology_only"
    assert node["inflow_m3s"][0] > 0
    assert node["inflow_m3s"][-1] == 0.0
    assert node["depth_m"] is None
    assert node["area_m2"] == 200.0
    # Sin cola de drenaje la humedad visual sigue a la lluvia acumulada.
    assert body["wetness"] == body["cum_rain_frac"]

    latest = client.get("/api/simulation/latest/frames")
    assert latest.status_code == 200
    assert latest.json()["simulation_id"] == body["simulation_id"]
    by_id = client.get(f"/api/simulation/{body['simulation_id']}/frames")
    assert by_id.status_code == 200


def test_simulation_run_con_swmm_mockeado(sim_env, monkeypatch):
    t0 = datetime(2026, 1, 1)

    def _point(minute, depth, flooding, status):
        return NodeResultPoint(
            id="T1",
            timestamp=t0 + timedelta(minutes=minute),
            total_inflow_m3s=0.1,
            lateral_inflow_m3s=0.1,
            depth_m=depth,
            hydraulic_head_m=1.0,
            flooding_m3s=flooding,
            ponded_volume_m3=0.0,
            status=status,
        )

    def fake_run(request):
        assert Path(request.inp_path).is_absolute()
        assert request.hydrographs and "T1" in request.hydrographs
        return SwmmRunResult(
            simulation_id="sim-swmm",
            inp_path=request.inp_path,
            node_results={
                "T1": [
                    _point(0, 0.2, 0.0, "normal"),
                    _point(5, 0.5, 0.1, "flooded"),
                    # cola de drenaje: el nodo se recupera al minuto 60
                    _point(60, 0.05, 0.0, "normal"),
                ],
                "crp_1": [
                    _point(0, 0.1, 0.0, "normal"),
                    _point(5, 0.9, 0.0, "normal"),
                    _point(60, 0.2, 0.0, "normal"),
                ],
            },
            link_results={},
            summary={},
        )

    monkeypatch.setattr(app_module.swmm_service, "run_simulation", fake_run)

    response = client.post("/api/simulation/run")
    assert response.status_code == 200
    body = response.json()

    assert body["swmm_available"] is True
    assert body["swmm_error"] is None
    assert body["swmm_simulation_id"] == "sim-swmm"

    node = body["nodes"]["T1"]
    assert node["source"] == "swmm"
    assert node["swmm_node"] == "T1"
    assert node["depth_m"][0] == 0.2
    assert node["depth_m"][1] == 0.5
    assert node["status"][0] == "normal"
    assert node["status"][1] == "flooded"
    # Retención constante hasta el siguiente reporte; al minuto 60 se recupera.
    assert node["status"][6] == "flooded"
    assert node["status"][-1] == "normal"
    assert node["flooding_m3s"][1] == 0.1
    assert body["nodes"]["crp_1"]["source"] == "swmm"
    assert body["nodes"]["crp_1"]["depth_m"][1] == 0.9

    # Humedad visual: llena al terminar la lluvia (min 30, frame 6) y decae
    # linealmente a 0 durante la cola de drenaje.
    assert body["wetness"][6] == 1.0
    assert body["wetness"][7] < 1.0
    assert body["wetness"][8] < body["wetness"][7]
    assert body["wetness"][-1] == 0.0


def _write_inp(path: Path, node_ids: list[str]) -> None:
    lines = ["[JUNCTIONS]", ";;Name Elev MaxDepth"]
    lines += [f"{node_id} 10.0 2.0" for node_id in node_ids]
    lines += ["[OUTFALLS]", "OUT_X 9.0 FREE NO"]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_simulation_automapea_tragantes_presentes_en_inp(sim_env, tmp_path, monkeypatch):
    inp_path = tmp_path / "red.inp"
    _write_inp(inp_path, ["T1"])
    captured = {}

    def fake_run(request):
        captured["node_mapping"] = dict(request.node_mapping)
        captured["duration_min"] = request.duration_min
        captured["last_point"] = request.hydrographs["T1"][-1]
        return SwmmRunResult(
            simulation_id="sim-auto",
            inp_path=request.inp_path,
            node_results={"T1": []},
            link_results={},
            summary={},
        )

    monkeypatch.setattr(app_module.swmm_service, "run_simulation", fake_run)

    response = client.post(
        "/api/simulation/run",
        # mapeo explícito obsoleto: debe filtrarse sin abortar SWMM
        json={"swmm": {"inp_path": str(inp_path), "node_mapping_text": "ZZ:QQ"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["swmm_available"] is True
    assert captured["node_mapping"] == {"T1": "T1"}
    # cola de drenaje: 30 min de lluvia + 30 min
    assert captured["duration_min"] == 60.0
    # el hidrograma se cierra en cero al terminar la lluvia
    assert captured["last_point"].minute == 30.0
    assert captured["last_point"].captured_flow_m3s == 0.0
    assert body["nodes"]["T1"]["swmm_node"] == "T1"


def test_simulation_sin_nodos_mapeables_degrada_con_mensaje(sim_env, tmp_path, monkeypatch):
    inp_path = tmp_path / "red.inp"
    _write_inp(inp_path, ["X9"])  # ningún tragante del grid existe en el INP

    def fail_run(request):
        raise AssertionError("SWMM no debería ejecutarse sin mapeo")

    monkeypatch.setattr(app_module.swmm_service, "run_simulation", fail_run)

    response = client.post(
        "/api/simulation/run",
        json={"swmm": {"inp_path": str(inp_path), "node_mapping_text": ""}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["swmm_available"] is False
    assert "Ningún tragante" in body["swmm_error"]
    assert body["nodes"]["T1"]["source"] == "hydrology"


def test_simulation_run_acepta_overrides_del_escenario(sim_env, monkeypatch):
    monkeypatch.setattr(
        app_module.swmm_service,
        "run_simulation",
        lambda request: (_ for _ in ()).throw(PySwmmUnavailableError("sin pyswmm")),
    )

    response = client.post(
        "/api/simulation/run",
        json={"rainfall": {"duration_min": 10, "interval_min": 10, "intensity_mm_h": 120}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["times_min"] == [0.0, 10.0]
    assert body["rain_mm_h"][0] == 120.0
