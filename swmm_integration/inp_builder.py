from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class SwmmNodeDef:
    id: str
    invert_elev_m: float
    max_depth_m: float = 2.0
    kind: str = "junction"  # junction | outfall
    x: float = 0.0
    y: float = 0.0
    data_source: str = "estimated"


@dataclass(frozen=True)
class SwmmConduitDef:
    id: str
    from_node: str
    to_node: str
    length_m: float
    diameter_m: float = 0.60
    roughness_n: float = 0.013
    data_source: str = "estimated"


def render_minimal_inp(
    nodes: list[SwmmNodeDef],
    conduits: list[SwmmConduitDef],
    start_time: datetime | None = None,
    duration_min: float = 60.0,
) -> str:
    """Render an INP that receives external inflows only.

    The file intentionally has no [SUBCATCHMENTS] section. Runoff is computed in
    Python and injected through PySWMM generated_inflow().
    """
    start_time = start_time or datetime(2026, 1, 1)
    end_time = start_time + timedelta(minutes=duration_min)
    junctions = [node for node in nodes if node.kind != "outfall"]
    outfalls = [node for node in nodes if node.kind == "outfall"]

    lines = [
        "[TITLE]",
        "Cariari external-inflow MVP model",
        "",
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "INFILTRATION         HORTON",
        "FLOW_ROUTING         DYNWAVE",
        f"START_DATE           {start_time:%m/%d/%Y}",
        f"START_TIME           {start_time:%H:%M:%S}",
        f"REPORT_START_DATE    {start_time:%m/%d/%Y}",
        f"REPORT_START_TIME    {start_time:%H:%M:%S}",
        f"END_DATE             {end_time:%m/%d/%Y}",
        f"END_TIME             {end_time:%H:%M:%S}",
        "SWEEP_START          01/01",
        "SWEEP_END            12/31",
        "DRY_DAYS             0",
        "REPORT_STEP          00:05:00",
        "WET_STEP             00:05:00",
        "DRY_STEP             00:05:00",
        "ROUTING_STEP         00:00:30",
        "ALLOW_PONDING        YES",
        "",
        "[JUNCTIONS]",
        ";;Name           Elevation  MaxDepth   InitDepth  SurDepth   Aponded",
    ]
    for node in junctions:
        lines.append(
            f"{node.id:<16} {node.invert_elev_m:.3f}     {node.max_depth_m:.3f}     0          0.50       0"
        )

    lines.extend(["", "[OUTFALLS]", ";;Name           Elevation  Type       Stage Data       Gated"])
    for node in outfalls:
        lines.append(f"{node.id:<16} {node.invert_elev_m:.3f}     FREE                         NO")

    lines.extend(["", "[CONDUITS]", ";;Name           From Node        To Node          Length     Roughness  InOffset OutOffset InitFlow MaxFlow"])
    for conduit in conduits:
        lines.append(
            f"{conduit.id:<16} {conduit.from_node:<16} {conduit.to_node:<16} "
            f"{conduit.length_m:.3f}   {conduit.roughness_n:.4f}     0        0         0        0"
        )

    lines.extend(["", "[XSECTIONS]", ";;Link           Shape       Geom1      Geom2      Geom3      Geom4      Barrels"])
    for conduit in conduits:
        lines.append(f"{conduit.id:<16} CIRCULAR    {conduit.diameter_m:.3f}     0          0          0          1")

    lines.extend(["", "[COORDINATES]", ";;Node           X-Coord          Y-Coord"])
    for node in nodes:
        lines.append(f"{node.id:<16} {node.x:.3f}          {node.y:.3f}")

    lines.extend(["", "[REPORT]", "INPUT      YES", "CONTROLS   NO", "SUBCATCHMENTS ALL", "NODES ALL", "LINKS ALL", ""])
    return "\n".join(lines)


def write_minimal_inp(
    path: str | Path,
    nodes: list[SwmmNodeDef],
    conduits: list[SwmmConduitDef],
    start_time: datetime | None = None,
    duration_min: float = 60.0,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_minimal_inp(nodes, conduits, start_time, duration_min), encoding="utf-8")
    return out

