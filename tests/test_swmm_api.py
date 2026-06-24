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
