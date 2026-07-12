from __future__ import annotations

import math
from typing import Any

from .rainfall import build_rainfall_intervals
from .schemas import HydrographResult, LinkResultPoint, NodeResultPoint, SwmmRunResult


def _hold(minutes: list[float], values: list[Any], minute: float, default: Any):
    """Retención constante por tramos: el último valor sigue activo hasta el
    siguiente punto (misma estrategia que flow_at_minute en engine.py)."""
    current = default
    for m, v in zip(minutes, values):
        if m > minute + 1e-9:
            break
        current = v
    return current


def _node_minutes(points: list[NodeResultPoint]) -> list[float]:
    """Minutos relativos al primer reporte del nodo.

    Se usa el primer timestamp como origen (y no start_time) porque pyswmm
    reporta con la fecha de inicio del INP, que puede no coincidir con la
    fecha del escenario de lluvia.
    """
    if not points:
        return []
    origin = points[0].timestamp
    return [max(0.0, (p.timestamp - origin).total_seconds() / 60.0) for p in points]


def _result_minutes(points: list[NodeResultPoint] | list[LinkResultPoint]) -> list[float]:
    if not points:
        return []
    origin = points[0].timestamp
    return [max(0.0, (p.timestamp - origin).total_seconds() / 60.0) for p in points]


def build_simulation_frames(
    hydro: HydrographResult,
    swmm: SwmmRunResult | None,
    cfg: dict[str, Any],
    swmm_error: str | None = None,
) -> dict[str, Any]:
    """Remuestrea lluvia, hidrogramas y resultados SWMM a una rejilla uniforme
    de frames para el visor 3D (timeline + valores por tragante)."""
    rainfall = hydro.rainfall
    intervals = build_rainfall_intervals(rainfall)
    swmm_cfg = cfg.get("swmm") or {}
    node_mapping: dict[str, str] = swmm_cfg.get("node_mapping") or {}

    duration_min = float(rainfall.duration_min)
    node_minutes_by_swmm_id: dict[str, list[float]] = {}
    link_minutes_by_swmm_id: dict[str, list[float]] = {}
    if swmm is not None:
        step_min = float(swmm_cfg.get("report_step_min") or 5.0)
        for node_id, points in swmm.node_results.items():
            mins = _result_minutes(points)
            node_minutes_by_swmm_id[node_id] = mins
            if mins:
                duration_min = max(duration_min, mins[-1])
        for link_id, points in swmm.link_results.items():
            mins = _result_minutes(points)
            link_minutes_by_swmm_id[link_id] = mins
            if mins:
                duration_min = max(duration_min, mins[-1])
    else:
        step_min = float(rainfall.interval_min)

    n_frames = int(math.ceil(duration_min / step_min - 1e-9)) + 1
    times_min = [round(k * step_min, 4) for k in range(n_frames)]

    total_depth_mm = sum(iv.depth_mm for iv in intervals)
    rain_mm_h: list[float] = []
    cum_rain_frac: list[float] = []
    for t in times_min:
        intensity = 0.0
        cum_depth = 0.0
        for iv in intervals:
            if iv.start_min - 1e-9 <= t < iv.end_min - 1e-9:
                intensity = iv.intensity_mm_h
            if iv.end_min <= t + 1e-9:
                cum_depth += iv.depth_mm
            elif iv.start_min < t:
                cum_depth += iv.depth_mm * (t - iv.start_min) / max(iv.end_min - iv.start_min, 1e-9)
        rain_mm_h.append(round(intensity, 3))
        cum_rain_frac.append(round(cum_depth / total_depth_mm, 4) if total_depth_mm > 0 else 0.0)

    # "Humedad" visual: sube con la lluvia acumulada y, al terminar la lluvia,
    # decae linealmente durante la cola de drenaje — en el visor el agua se
    # retrae hacia los cauces y tragantes hasta desaparecer.
    rain_end_min = 0.0
    for interval in intervals:
        if interval.depth_mm > 0:
            rain_end_min = max(rain_end_min, interval.end_min)
    tail_span = max(duration_min - rain_end_min, 0.0)
    wetness: list[float] = []
    for idx, t in enumerate(times_min):
        if rain_end_min <= 0 or t <= rain_end_min + 1e-9 or tail_span <= 1e-9:
            wetness.append(cum_rain_frac[idx])
        else:
            wetness.append(round(max(0.0, 1.0 - (t - rain_end_min) / tail_span), 4))

    area_by_sink = {area.sink_id: area for area in hydro.tributary_areas}

    def _swmm_node_entry(swmm_node: str, swmm_points: list[NodeResultPoint], area: Any = None) -> dict[str, Any]:
        mins = node_minutes_by_swmm_id.get(swmm_node, [])
        return {
            "source": "swmm",
            "swmm_node": swmm_node,
            "area_m2": round(area.area_m2, 1) if area else 0.0,
            "cell_count": area.cell_count if area else 0,
            "inflow_m3s": [
                round(_hold(mins, [p.total_inflow_m3s for p in swmm_points], t, 0.0), 5)
                for t in times_min
            ],
            "depth_m": [
                round(_hold(mins, [p.depth_m for p in swmm_points], t, 0.0), 4)
                for t in times_min
            ],
            "flooding_m3s": [
                round(_hold(mins, [p.flooding_m3s for p in swmm_points], t, 0.0), 5)
                for t in times_min
            ],
            "ponded_m3": [
                round(_hold(mins, [p.ponded_volume_m3 for p in swmm_points], t, 0.0), 3)
                for t in times_min
            ],
            "status": [
                _hold(mins, [p.status for p in swmm_points], t, "normal")
                for t in times_min
            ],
        }

    nodes: dict[str, dict[str, Any]] = {}
    consumed_swmm_nodes: set[str] = set()
    for sink_id, points in hydro.hydrographs.items():
        swmm_node = node_mapping.get(sink_id)
        swmm_points = swmm.node_results.get(swmm_node) if (swmm is not None and swmm_node) else None
        area = area_by_sink.get(sink_id)
        if swmm_points:
            entry = _swmm_node_entry(swmm_node, swmm_points, area)
            consumed_swmm_nodes.add(swmm_node)
        else:
            h_mins = [p.minute for p in points]
            h_flows = [p.captured_flow_m3s for p in points]
            entry: dict[str, Any] = {
                "swmm_node": swmm_node,
                "area_m2": round(area.area_m2, 1) if area else 0.0,
                "cell_count": area.cell_count if area else 0,
                "source": "hydrology",
            }
            # Tras el fin de la lluvia el hidrograma escalonado termina: 0.
            entry["inflow_m3s"] = [
                round(_hold(h_mins, h_flows, t, 0.0), 5) if t < rainfall.duration_min - 1e-9 else 0.0
                for t in times_min
            ]
            entry["depth_m"] = None
            entry["flooding_m3s"] = None
            entry["ponded_m3"] = None
            entry["status"] = ["hydrology_only"] * n_frames
        nodes[sink_id] = entry

    links: dict[str, dict[str, Any]] = {}
    if swmm is not None:
        for swmm_node, swmm_points in swmm.node_results.items():
            if swmm_node in consumed_swmm_nodes or swmm_node in nodes:
                continue
            nodes[swmm_node] = _swmm_node_entry(swmm_node, swmm_points)
        for link_id, link_points in swmm.link_results.items():
            mins = link_minutes_by_swmm_id.get(link_id, [])
            links[link_id] = {
                "source": "swmm",
                "flow_m3s": [
                    round(_hold(mins, [p.flow_m3s for p in link_points], t, 0.0), 5)
                    for t in times_min
                ],
                "depth_m": [
                    round(_hold(mins, [p.depth_m for p in link_points], t, 0.0), 4)
                    for t in times_min
                ],
                "velocity_m_s": [
                    round(_hold(mins, [p.velocity_m_s for p in link_points], t, 0.0), 4)
                    for t in times_min
                ],
                "direction": [
                    _hold(mins, [p.direction for p in link_points], t, "forward")
                    for t in times_min
                ],
                "status": [
                    _hold(mins, [p.status for p in link_points], t, "normal")
                    for t in times_min
                ],
            }

    return {
        "start_time": rainfall.start_time.isoformat(),
        "scenario_name": rainfall.scenario_name,
        "duration_min": round(duration_min, 4),
        "step_min": step_min,
        "times_min": times_min,
        "rain_mm_h": rain_mm_h,
        "max_rain_mm_h": max(rain_mm_h) if rain_mm_h else 0.0,
        "cum_rain_frac": cum_rain_frac,
        "wetness": wetness,
        "nodes": nodes,
        "links": links,
        "swmm_available": swmm is not None,
        "swmm_error": swmm_error,
        "mass_balance": hydro.mass_balance,
        "warnings": list(hydro.warnings),
    }
