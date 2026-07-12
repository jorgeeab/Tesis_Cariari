from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

import app as app_module
from swmm_integration.schemas import LinkResultPoint, NodeResultPoint, SwmmRunResult


client = TestClient(app_module.app)


def _rainfall_body() -> dict:
    return {
        "scenario_name": "api-test",
        "start_time": "2026-01-01T00:00:00",
        "duration_min": 10,
        "interval_min": 10,
        "intensity_mm_h": 60,
    }


def _hydrology_body() -> dict:
    return {
        "rainfall": _rainfall_body(),
        "surface": {
            "nx": 2,
            "ny": 1,
            "cell_area_m2": 100,
            "receivers": [1, -1],
            "coverage": ["base", "base"],
            "sinks": [{"id": "T1", "cell_index": 1}],
        },
        "infiltration": {"base": 0.0},
    }


def test_excavation_tuner_uses_same_application():
    response = client.get("/ajustar-excavacion")

    assert response.status_code == 200
    assert "/api/excavation/run" in response.text
    assert "/resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html" in response.text
    assert "localhost:8765" not in response.text


def test_excavation_run_streams_generator_output(monkeypatch):
    class FakeProcess:
        stdout = iter(["Preparando DEM\n", "HTML generado\n"])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(app_module.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        app_module,
        "_read_map_terrain_params",
        lambda: {
            "street_depth": 1.8,
            "street_width": 1.0,
            "street_ramp": 7.0,
            "river_depth": 2.5,
            "river_width": 5.0,
        },
    )

    response = client.get(
        "/api/excavation/run",
        params={"street_depth": 1.8, "river_width": 5.0},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Preparando DEM" in response.text
    assert "VERIFICADO: mapa principal actualizado" in response.text
    assert "data: __done__" in response.text


def test_create_scenario_and_run_hydrology():
    app_module.SCENARIOS.clear()
    app_module.HYDROLOGY_RESULTS.clear()

    scenario_response = client.post("/api/scenarios", json={"rainfall": _rainfall_body()})
    assert scenario_response.status_code == 200
    assert scenario_response.json()["scenario_id"] in app_module.SCENARIOS

    hydrology_response = client.post("/api/hydrology/run", json=_hydrology_body())
    assert hydrology_response.status_code == 200
    body = hydrology_response.json()
    assert body["hydrology_id"] in app_module.HYDROLOGY_RESULTS
    assert body["hydrographs"]["T1"][0]["captured_flow_m3s"] > 0
    assert body["mass_balance"]["error_pct"] == 0.0


def test_validate_swmm_endpoint_accepts_base_model():
    response = client.post("/api/swmm/validate", json={"inp_path": "models/cariari_base.inp"})

    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_ui_scenario_endpoint_saves_normalized_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "UI_SCENARIO_PATH", tmp_path / "ui_scenario.json")

    response = client.post(
        "/api/ui/scenario",
        json={
            "infiltration_pct": {"base": 33, "vegetation": 88, "building": 4, "street": 7},
            "rainfall": {"scenario_name": "principal", "intensity_mm_h": 80, "duration_min": 40},
            "capture": {"capture_mode": "limited", "capture_efficiency": 0.75},
            "swmm": {"inp_path": "models/cariari_base.inp", "node_mapping_text": "T1:T1"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["config"]["infiltration_pct"]["base"] == 33
    assert body["config"]["rainfall"]["scenario_name"] == "principal"
    assert body["config"]["capture"]["capture_efficiency"] == 0.75
    assert body["config"]["swmm"]["node_mapping"] == {"T1": "T1"}
    assert client.get("/api/ui/scenario").json()["infiltration_pct"]["vegetation"] == 88


def test_run_swmm_uses_stored_hydrology_and_returns_result(monkeypatch):
    app_module.HYDROLOGY_RESULTS.clear()
    app_module.SWMM_RESULTS.clear()

    hydrology = client.post("/api/hydrology/run", json=_hydrology_body()).json()

    def fake_run(request):
        assert Path(request.inp_path).is_absolute()
        assert request.hydrographs is not None
        assert request.duration_min == 10
        timestamp = datetime(2026, 1, 1)
        return SwmmRunResult(
            simulation_id="sim-test",
            inp_path=request.inp_path,
            node_results={
                "T1": [
                    NodeResultPoint(
                        id="T1",
                        timestamp=timestamp,
                        total_inflow_m3s=0.1,
                        lateral_inflow_m3s=0.1,
                        depth_m=0.2,
                        hydraulic_head_m=1.2,
                        flooding_m3s=0.0,
                        ponded_volume_m3=0.0,
                        status="normal",
                    )
                ]
            },
            link_results={
                "C_T1_J1": [
                    LinkResultPoint(
                        id="C_T1_J1",
                        timestamp=timestamp,
                        flow_m3s=0.1,
                        depth_m=0.1,
                        velocity_m_s=0.5,
                        direction="forward",
                        status="normal",
                    )
                ]
            },
            summary={"ok": True},
        )

    monkeypatch.setattr(app_module.swmm_service, "run_simulation", fake_run)

    response = client.post(
        "/api/swmm/run",
        json={
            "inp_path": "models/cariari_base.inp",
            "hydrology_id": hydrology["hydrology_id"],
            "node_mapping": {"T1": "T1"},
        },
    )

    assert response.status_code == 200
    assert response.json()["simulation_id"] == "sim-test"
    assert client.get("/api/swmm/results/sim-test/nodes").json()["T1"][0]["status"] == "normal"
    assert client.get("/api/swmm/results/sim-test/links").json()["C_T1_J1"][0]["direction"] == "forward"
    assert client.get("/api/swmm/results/sim-test/summary").json()["ok"] is True
