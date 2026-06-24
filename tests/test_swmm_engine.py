from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from swmm_integration.engine import SwmmSimulationService
from swmm_integration.schemas import HydrographPoint, SwmmRunRequest


class FakeNode:
    calls: list[tuple[str, float]] = []

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.total_inflow = 0.0
        self.lateral_inflow = 0.0
        self.depth = 0.1
        self.hydraulic_head = 1.0
        self.flooding = 0.0
        self.ponded_volume = 0.0

    def generated_inflow(self, flow_m3s: float):
        self.total_inflow = flow_m3s
        self.lateral_inflow = flow_m3s
        self.calls.append((self.node_id, flow_m3s))


class FakeSimulation:
    def __init__(self, _inp_path: str):
        self.start = datetime(2026, 1, 1)
        self.current_time = self.start
        self.report_step_s = 300

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def step_advance(self, seconds: int):
        self.report_step_s = seconds

    def __iter__(self):
        for index in range(3):
            self.current_time = self.start + timedelta(seconds=index * self.report_step_s)
            yield self


class FakeNodes:
    def __init__(self, _simulation):
        self.nodes = {
            "T1": FakeNode("T1"),
            "J1": FakeNode("J1"),
        }

    def __getitem__(self, node_id: str) -> FakeNode:
        if node_id not in self.nodes:
            raise KeyError(node_id)
        return self.nodes[node_id]


class FakeLink:
    linkid = "C1"
    flow = 0.0
    depth = 0.0
    velocity = 0.0


class FakeLinks:
    def __init__(self, _simulation):
        self.links = {"C1": FakeLink()}

    def __iter__(self):
        return iter(self.links)

    def __getitem__(self, link_id: str):
        return self.links[link_id]


def _fake_pyswmm():
    return SimpleNamespace(Simulation=FakeSimulation, Nodes=FakeNodes, Links=FakeLinks)


def _hydrograph_points() -> list[HydrographPoint]:
    return [
        HydrographPoint(
            minute=0,
            volume_m3=30,
            flow_m3s=0.1,
            captured_volume_m3=30,
            captured_flow_m3s=0.1,
            bypass_volume_m3=0,
            bypass_flow_m3s=0,
        ),
        HydrographPoint(
            minute=5,
            volume_m3=60,
            flow_m3s=0.2,
            captured_volume_m3=60,
            captured_flow_m3s=0.2,
            bypass_volume_m3=0,
            bypass_flow_m3s=0,
        ),
    ]


def test_generated_inflow_is_sent_to_mapped_nodes(monkeypatch):
    FakeNode.calls = []
    monkeypatch.setattr("swmm_integration.engine._load_pyswmm", _fake_pyswmm)

    result = SwmmSimulationService().run_simulation(
        SwmmRunRequest(
            inp_path="models/cariari_base.inp",
            hydrographs={"sink-a": _hydrograph_points()},
            node_mapping={"sink-a": "T1"},
            start_time=datetime(2026, 1, 1),
            duration_min=10,
            report_step_min=5,
        )
    )

    assert FakeNode.calls == [("T1", 0.1), ("T1", 0.2), ("T1", 0.2)]
    assert result.node_results["T1"][-1].total_inflow_m3s == 0.2
    assert result.link_results["C1"]
    assert result.summary["generated_inflow_strategy"] == "piecewise_constant_hold"


def test_missing_mapped_node_is_reported(monkeypatch):
    monkeypatch.setattr("swmm_integration.engine._load_pyswmm", _fake_pyswmm)

    with pytest.raises(KeyError):
        SwmmSimulationService().run_simulation(
            SwmmRunRequest(
                inp_path="models/cariari_base.inp",
                hydrographs={"sink-a": _hydrograph_points()},
                node_mapping={"sink-a": "NOPE"},
                start_time=datetime(2026, 1, 1),
                duration_min=10,
            )
        )
