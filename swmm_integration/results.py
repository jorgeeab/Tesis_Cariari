from __future__ import annotations

from .schemas import LinkResultPoint, NodeResultPoint


def summarize_node_results(
    results: dict[str, list[NodeResultPoint]], report_step_min: float = 1.0
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    step_s = report_step_min * 60.0
    for node_id, points in results.items():
        if not points:
            continue
        summary[node_id] = {
            "max_depth_m": max(p.depth_m for p in points),
            "max_total_inflow_m3s": max(p.total_inflow_m3s for p in points),
            "max_flooding_m3s": max(p.flooding_m3s for p in points),
            "total_flooding_volume_m3": sum(p.flooding_m3s * step_s for p in points),
            "flooded_duration_min": sum(report_step_min for p in points if p.flooding_m3s > 0),
        }
    return summary


def summarize_link_results(results: dict[str, list[LinkResultPoint]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for link_id, points in results.items():
        if not points:
            continue
        summary[link_id] = {
            "max_flow_m3s": max(abs(p.flow_m3s) for p in points),
            "max_depth_m": max(p.depth_m for p in points),
            "max_velocity_m_s": max(abs(p.velocity_m_s) for p in points),
            "reverse_flow_steps": sum(1 for p in points if p.flow_m3s < 0),
        }
    return summary
