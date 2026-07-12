"""Genera models/cariari_pluvial.inp desde la red pluvial trazada.

Fuentes: output/shapefiles_v6/red_pluvial_nodos.shp y red_pluvial_links.shp
(EPSG:5367, metros). Nodos con invert = fon_elev y profundidad = prof_m;
tramos con longitud = dist_m.

Reglas:
- Nodo terminal (solo aparece como destino) con un único tramo conectado
  → OUTFALL. Si tiene varios tramos, se conserva como junction y se agrega
  un outfall sintético `out_<id>` con un tramo corto por debajo del invert
  (SWMM solo permite un tramo por outfall).
- Conductos circulares D=0.60 m, n=0.013 (0.80 m para los sintéticos).

El INP resultante NO tiene [SUBCATCHMENTS]: la escorrentía se calcula en
Python y se inyecta vía generated_inflow (swmm_integration/engine.py).

Uso:  python scripts/generar_inp_pluvial.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import geopandas as gpd  # noqa: E402

from swmm_integration.inp_builder import SwmmConduitDef, SwmmNodeDef, write_minimal_inp  # noqa: E402
from swmm_integration.validation import validate_inp_file  # noqa: E402

NODOS_SHP = BASE_DIR / "output" / "shapefiles_v6" / "red_pluvial_nodos.shp"
LINKS_SHP = BASE_DIR / "output" / "shapefiles_v6" / "red_pluvial_links.shp"
OUT_INP = BASE_DIR / "models" / "cariari_pluvial.inp"

DEFAULT_DIAMETER_M = 0.60
SYNTH_DIAMETER_M = 0.80
MIN_DEPTH_M = 0.80  # que quepa el tubo de 0.60 m
MIN_LENGTH_M = 1.0


def _finite(value, fallback):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if math.isfinite(value) else fallback


def build_network_defs(nodos: gpd.GeoDataFrame, links: gpd.GeoDataFrame):
    src_ids = set(links["src_id"])
    link_count: dict[str, int] = {}
    for col in ("src_id", "dst_id"):
        for nid in links[col]:
            link_count[nid] = link_count.get(nid, 0) + 1

    nodes_out: list[SwmmNodeDef] = []
    extra_conduits: list[SwmmConduitDef] = []
    stats = {"junctions": 0, "outfalls": 0, "synthetic_outfalls": 0}

    for row in nodos.itertuples():
        nid = str(row.id)
        sup = _finite(row.sup_elev, 0.0)
        depth = max(_finite(row.prof_m, 1.0), MIN_DEPTH_M)
        invert = _finite(row.fon_elev, sup - depth)
        x, y = float(row.geometry.x), float(row.geometry.y)
        terminal = nid not in src_ids

        if terminal and link_count.get(nid, 0) == 1:
            nodes_out.append(SwmmNodeDef(
                id=nid, invert_elev_m=invert, max_depth_m=depth,
                kind="outfall", x=x, y=y, data_source=str(row.prof_src),
            ))
            stats["outfalls"] += 1
            continue

        nodes_out.append(SwmmNodeDef(
            id=nid, invert_elev_m=invert, max_depth_m=depth,
            kind="junction", x=x, y=y, data_source=str(row.prof_src),
        ))
        stats["junctions"] += 1

        if terminal:
            out_id = f"out_{nid}"
            nodes_out.append(SwmmNodeDef(
                id=out_id, invert_elev_m=invert - 0.5, max_depth_m=1.0,
                kind="outfall", x=x + 3.0, y=y - 3.0, data_source="synthetic",
            ))
            extra_conduits.append(SwmmConduitDef(
                id=f"C_{nid}_out", from_node=nid, to_node=out_id,
                length_m=5.0, diameter_m=SYNTH_DIAMETER_M,
            ))
            stats["synthetic_outfalls"] += 1

    conduits = [
        SwmmConduitDef(
            id=str(row.id),
            from_node=str(row.src_id),
            to_node=str(row.dst_id),
            length_m=max(_finite(row.dist_m, MIN_LENGTH_M), MIN_LENGTH_M),
            diameter_m=DEFAULT_DIAMETER_M,
        )
        for row in links.itertuples()
    ] + extra_conduits

    return nodes_out, conduits, stats


def main() -> int:
    nodos = gpd.read_file(NODOS_SHP)
    links = gpd.read_file(LINKS_SHP)
    print(f"Red pluvial: {len(nodos)} nodos, {len(links)} tramos")

    nodes_out, conduits, stats = build_network_defs(nodos, links)
    # END_TIME holgado (4 h): el engine corta en duration_min de cada corrida.
    write_minimal_inp(OUT_INP, nodes_out, conduits, duration_min=240.0)
    print(f"INP escrito: {OUT_INP}")
    print(f"  junctions={stats['junctions']}  outfalls={stats['outfalls']}  "
          f"outfalls sintéticos={stats['synthetic_outfalls']}  conduits={len(conduits)}")

    report = validate_inp_file(OUT_INP)
    if report.valid:
        print("  Validación: OK")
        return 0
    for issue in report.issues:
        print(f"  {issue.level}: {issue.message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
