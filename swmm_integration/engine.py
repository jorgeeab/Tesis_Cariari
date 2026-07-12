from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from .results import summarize_link_results, summarize_node_results
from .schemas import (
    HydrographPoint,
    LinkResultPoint,
    NodeResultPoint,
    SwmmRunRequest,
    SwmmRunResult,
    ValidationReport,
)
from .validation import validate_inp_file


class PySwmmUnavailableError(RuntimeError):
    pass


def _load_pyswmm():
    try:
        from pyswmm import Links, Nodes, Simulation
    except Exception as exc:
        raise PySwmmUnavailableError(
            'PySWMM no esta instalado o no puede cargar EPA SWMM. Instale: pip install "pyswmm[swmm5.2.4]"'
        ) from exc
    return SimpleNamespace(Simulation=Simulation, Nodes=Nodes, Links=Links)


def flow_at_minute(points: list[HydrographPoint], minute: float, field: str = "captured_flow_m3s") -> float:
    """Piecewise-constant hydrograph lookup.

    The last value remains active until the next hydrograph point. This is the
    documented synchronization strategy between external runoff intervals and
    SWMM hydraulic steps.
    """
    if not points:
        return 0.0
    current = float(getattr(points[0], field))
    for point in points:
        if point.minute > minute:
            break
        current = float(getattr(point, field))
    return current


def _safe_float(obj, attr: str, default: float = 0.0) -> float:
    try:
        value = getattr(obj, attr)
        if callable(value):
            value = value()
        return float(value)
    except Exception:
        return default


def _iter_collection_ids(collection, attr_name: str) -> list[str]:
    """Return ids from a PySWMM collection, falling back when it is not iterable."""
    try:
        raw_items = list(collection)
    except Exception:
        return []

    ids: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            item_id = item
        else:
            item_id = getattr(item, attr_name, str(item))
        if item_id:
            ids.append(str(item_id))
    return ids


class SwmmSimulationService:
    def __init__(self, near_capacity_ratio: float = 0.80):
        self.near_capacity_ratio = near_capacity_ratio

    def validate_model(self, inp_path: str | Path) -> ValidationReport:
        return validate_inp_file(inp_path)

    def run_simulation(self, request: SwmmRunRequest) -> SwmmRunResult:
        inp_path = Path(request.inp_path)
        report = self.validate_model(inp_path)
        if not report.valid:
            details = "; ".join(issue.message for issue in report.issues if issue.level == "error")
            raise ValueError(f"Modelo SWMM invalido: {details}")

        pyswmm = _load_pyswmm()
        hydrographs = request.hydrographs or {}
        node_mapping = request.node_mapping or {sink_id: sink_id for sink_id in hydrographs}
        start_time = request.start_time or datetime.utcnow()
        duration_min = request.duration_min
        if duration_min is None:
            duration_min = max((points[-1].minute if points else 0.0) for points in hydrographs.values()) + request.report_step_min

        node_results: dict[str, list[NodeResultPoint]] = {}
        link_results: dict[str, list[LinkResultPoint]] = {}

        with pyswmm.Simulation(str(inp_path)) as sim:
            if hasattr(sim, "step_advance"):
                sim.step_advance(int(request.report_step_min * 60))
            nodes = pyswmm.Nodes(sim)
            links = pyswmm.Links(sim)
            mapped_nodes = {}
            for sink_id, node_id in node_mapping.items():
                try:
                    mapped_nodes[sink_id] = nodes[node_id]
                except Exception as exc:
                    raise KeyError(f"El nodo SWMM no existe para el tragante {sink_id}: {node_id}") from exc
            observed_node_ids = _iter_collection_ids(nodes, "nodeid")
            if not observed_node_ids:
                observed_node_ids = list(dict.fromkeys(node_mapping.values()))

            step_index = 0
            sim_origin = None
            for _ in sim:
                current_time = getattr(sim, "current_time", None)
                if not isinstance(current_time, datetime):
                    current_time = start_time + timedelta(minutes=step_index * request.report_step_min)
                # El minuto del hidrograma se mide desde el inicio de la propia
                # simulación: la fecha del INP puede no coincidir con start_time.
                if sim_origin is None:
                    sim_origin = current_time
                minute = max(0.0, (current_time - sim_origin).total_seconds() / 60.0)
                if minute > duration_min + 1e-9:
                    break

                for sink_id, node in mapped_nodes.items():
                    q = flow_at_minute(hydrographs.get(sink_id, []), minute)
                    node.generated_inflow(q)

                for node_id in observed_node_ids:
                    node = nodes[node_id]
                    total_inflow = _safe_float(node, "total_inflow")
                    flooding = _safe_float(node, "flooding")
                    depth = _safe_float(node, "depth")
                    status = "flooded" if flooding > 0 else "normal"
                    node_results.setdefault(node_id, []).append(
                        NodeResultPoint(
                            id=node_id,
                            timestamp=current_time,
                            total_inflow_m3s=total_inflow,
                            lateral_inflow_m3s=_safe_float(node, "lateral_inflow"),
                            depth_m=depth,
                            hydraulic_head_m=_safe_float(node, "hydraulic_head", _safe_float(node, "head")),
                            flooding_m3s=flooding,
                            ponded_volume_m3=_safe_float(node, "ponded_volume"),
                            status=status,
                        )
                    )

                try:
                    link_ids = list(links)
                except Exception:
                    link_ids = []
                for link_id in link_ids:
                    link = links[link_id] if isinstance(link_id, str) else link_id
                    actual_id = link_id if isinstance(link_id, str) else getattr(link, "linkid", str(link_id))
                    flow = _safe_float(link, "flow")
                    status = "reverse_flow" if flow < 0 else "normal"
                    link_results.setdefault(actual_id, []).append(
                        LinkResultPoint(
                            id=actual_id,
                            timestamp=current_time,
                            flow_m3s=flow,
                            depth_m=_safe_float(link, "depth"),
                            velocity_m_s=_safe_float(link, "velocity"),
                            capacity_ratio=None,
                            direction="reverse" if flow < 0 else "forward",
                            status=status,
                        )
                    )
                step_index += 1

        summary = {
            "nodes": summarize_node_results(node_results, request.report_step_min),
            "links": summarize_link_results(link_results),
            "duration_min": duration_min,
            "report_step_min": request.report_step_min,
            "generated_inflow_strategy": "piecewise_constant_hold",
        }
        return SwmmRunResult(
            simulation_id=str(uuid4()),
            inp_path=str(inp_path),
            node_results=node_results,
            link_results=link_results,
            summary=summary,
        )
