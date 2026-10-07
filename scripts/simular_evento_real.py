#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simula la respuesta de la red pluvial de Ciudad Cariari ante lluvia con
respaldo documental:

  (a) "real"   — evento observado extraído de la serie horaria de reanálisis
                 ERA5 (Hersbach et al., 2020) descargada vía Open-Meteo para
                 la celda de Cariari (datos/lluvia/era5_cariari_horario_*.csv).
  (b) "diseno" — tormenta de diseño de 60 min por bloques alternos construida
                 con las intensidades de la estación 84-111 Santa Lucía,
                 Heredia (IMN, Curvas IDF de estaciones mecánicas, Cuadro 17).

Encadena la misma tubería que /api/simulation/run (hidrología D8 sobre el
grid del DEM excavado → PySWMM sobre models/cariari_pluvial.inp) pero
permite hietogramas variables en el tiempo (RainfallEvent.time_series).

Uso:
    python scripts/simular_evento_real.py                    # ambos escenarios
    python scripts/simular_evento_real.py --escenario real
    python scripts/simular_evento_real.py --fecha 2016-06-14 --interval 5

Salidas: results/simulaciones/<escenario>/resumen.json + hietograma.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from swmm_integration import SwmmSimulationService, generate_hydrographs
from swmm_integration.schemas import (
    CaptureConfig,
    HydrographPoint,
    HydrologyRunRequest,
    InfiltrationConfig,
    RainfallEvent,
    RainfallPoint,
    SwmmRunRequest,
)
from swmm_integration.surface_grid import (
    build_surface_grid,
    load_surface_grid_payload,
    tributaries_for,
)
from swmm_integration.ui_config import load_ui_scenario, normalize_ui_scenario
from swmm_integration.validation import read_inp_node_ids

ERA5_CSV = BASE_DIR / "datos" / "lluvia" / "era5_cariari_horario_2000_2026.csv"
SURFACE_GRID_PATH = BASE_DIR / "results" / "hydrology" / "surface_grid.json"
UI_SCENARIO_PATH = BASE_DIR / "results" / "swmm" / "ui_scenario.json"
OUT_BASE = BASE_DIR / "results" / "simulaciones"

# Intensidades máximas (mm/h) por duración, estación 84-111 Santa Lucía,
# Heredia, período 1983-2010 (IMN, CIDF estaciones mecánicas, Cuadro 17).
IDF_SANTA_LUCIA = {
    5: {5: 215.13, 10: 245.00, 25: 282.75},
    10: {5: 155.42, 10: 169.48, 25: 187.26},
    15: {5: 140.59, 10: 154.40, 25: 171.85},
    30: {5: 109.69, 10: 121.90, 25: 137.32},
    60: {5: 74.48, 10: 82.86, 25: 93.44},
}


def leer_era5() -> list[tuple[dt.datetime, float]]:
    rows: list[tuple[dt.datetime, float]] = []
    with open(ERA5_CSV, encoding="utf-8") as f:
        started = False
        for line in csv.reader(f):
            if not line:
                continue
            if line[0] == "time":
                started = True
                continue
            if not started:
                continue
            rows.append((dt.datetime.fromisoformat(line[0]), float(line[1])))
    return rows


def hietograma_evento_real(fecha: dt.date) -> tuple[list[RainfallPoint], float, dict]:
    """Serie horaria del día `fecha` (hora local), desde la primera hora con
    lluvia hasta la última, como puntos escalonados minuto→mm/h."""
    dia = [(t, p) for t, p in leer_era5() if t.date() == fecha]
    if not dia or all(p == 0 for _, p in dia):
        raise SystemExit(f"Sin lluvia registrada el {fecha} en {ERA5_CSV.name}")
    wet_hours = [i for i, (_, p) in enumerate(dia) if p > 0]
    i0, i1 = wet_hours[0], wet_hours[-1]
    ventana = dia[i0 : i1 + 1]
    puntos = [
        RainfallPoint(minute=60.0 * k, intensity_mm_h=p)
        for k, (_, p) in enumerate(ventana)
    ]
    duracion = 60.0 * len(ventana)
    meta = {
        "fuente": "ERA5 (reanálisis ECMWF) vía API Open-Meteo, celda 9.947 N, -84.173 O",
        "inicio_local": ventana[0][0].isoformat(),
        "fin_local": (ventana[-1][0] + dt.timedelta(hours=1)).isoformat(),
        "total_mm": round(sum(p for _, p in ventana), 1),
        "pico_mm_h": max(p for _, p in ventana),
    }
    return puntos, duracion, meta


def hietograma_bloques_alternos(T: int = 10, paso_min: float = 5.0) -> tuple[list[RainfallPoint], float, dict]:
    """Tormenta de diseño de 60 min por bloques alternos con las intensidades
    IDF de Santa Lucía (84-111) para el período de retorno T."""
    duraciones = sorted(IDF_SANTA_LUCIA)
    laminas = {d: IDF_SANTA_LUCIA[d][T] * d / 60.0 for d in duraciones}  # mm acumulados
    total = 60.0
    n = int(total / paso_min)
    incrementos: list[float] = []
    prev_d, prev_p = 0.0, 0.0
    for d in duraciones:
        bloques = int((d - prev_d) / paso_min)
        inc = (laminas[d] - prev_p) / bloques
        incrementos.extend([inc] * bloques)
        prev_d, prev_p = d, laminas[d]
    # bloques alternos: máximo al centro, luego alternando derecha/izquierda
    orden = sorted(incrementos, reverse=True)
    arreglo = [0.0] * n
    centro = n // 2 - 1
    pos = [centro]
    for k in range(1, n):
        pos.append(centro + (k + 1) // 2 * (1 if k % 2 else -1))
    for lamina, p in zip(orden, pos):
        arreglo[p] = lamina
    puntos = [
        RainfallPoint(minute=paso_min * k, intensity_mm_h=arreglo[k] * 60.0 / paso_min)
        for k in range(n)
    ]
    meta = {
        "fuente": "CIDF estación 84-111 Santa Lucía, Heredia (IMN, estaciones mecánicas, Cuadro 17)",
        "periodo_retorno_anios": T,
        "metodo": f"bloques alternos, paso {paso_min:g} min",
        "total_mm": round(sum(arreglo), 1),
        "pico_mm_h": round(max(p.intensity_mm_h for p in puntos), 1),
    }
    return puntos, total, meta


def correr_escenario(nombre: str, puntos: list[RainfallPoint], duracion_min: float,
                     meta: dict, interval_min: float, cola_min: float = 60.0) -> dict:
    cfg = normalize_ui_scenario(load_ui_scenario(UI_SCENARIO_PATH))
    payload = load_surface_grid_payload(SURFACE_GRID_PATH)
    capture = CaptureConfig(**cfg["capture"])
    grid = build_surface_grid(payload, capture)

    rainfall = RainfallEvent(
        scenario_name=nombre,
        start_time=dt.datetime.fromisoformat(meta.get("inicio_local", "2026-01-01T00:00:00")),
        duration_min=duracion_min,
        interval_min=interval_min,
        time_series=puntos,
    )
    infiltration = InfiltrationConfig(
        **{k: v / 100.0 for k, v in cfg["infiltration_pct"].items()}
    )
    print(f"[{nombre}] hidrología D8… ({len(puntos)} puntos, {duracion_min:g} min)")
    hydro = generate_hydrographs(
        HydrologyRunRequest(rainfall=rainfall, surface=grid, infiltration=infiltration),
        precomputed_tributaries=tributaries_for(SURFACE_GRID_PATH, grid),
    )

    vol_captado = sum(
        p.captured_volume_m3 for pts in hydro.hydrographs.values() for p in pts
    )
    caudal_pico_total = 0.0
    minutos = sorted({p.minute for pts in hydro.hydrographs.values() for p in pts})
    for m in minutos:
        q = sum(
            next((p.captured_flow_m3s for p in pts if p.minute == m), 0.0)
            for pts in hydro.hydrographs.values()
        )
        caudal_pico_total = max(caudal_pico_total, q)

    inp_path = BASE_DIR / cfg["swmm"]["inp_path"]
    inp_nodes = read_inp_node_ids(inp_path)
    node_mapping = {s: s for s in hydro.hydrographs if s in inp_nodes}

    zero_tail = HydrographPoint(
        minute=duracion_min, volume_m3=0.0, flow_m3s=0.0,
        captured_volume_m3=0.0, captured_flow_m3s=0.0,
        bypass_volume_m3=0.0, bypass_flow_m3s=0.0,
    )
    swmm_hydrographs = {s: list(p) + [zero_tail] for s, p in hydro.hydrographs.items()}

    print(f"[{nombre}] PySWMM sobre {inp_path.name} ({len(node_mapping)} tragantes mapeados)…")
    service = SwmmSimulationService()
    result = service.run_simulation(
        SwmmRunRequest(
            inp_path=str(inp_path),
            hydrographs=swmm_hydrographs,
            node_mapping=node_mapping,
            start_time=rainfall.start_time,
            duration_min=duracion_min + cola_min,
            report_step_min=cfg["swmm"]["report_step_min"],
        )
    )

    paso_s = cfg["swmm"]["report_step_min"] * 60.0
    nodos = []
    for nid, serie in result.node_results.items():
        vol_flood = sum(p.flooding_m3s for p in serie) * paso_s
        nodos.append({
            "nodo": nid,
            "prof_max_m": round(max((p.depth_m for p in serie), default=0.0), 3),
            "caudal_max_m3s": round(max((p.total_inflow_m3s for p in serie), default=0.0), 4),
            "vol_inundacion_m3": round(vol_flood, 1),
        })
    inundados = sorted(
        [n for n in nodos if n["vol_inundacion_m3"] > 0.5],
        key=lambda x: -x["vol_inundacion_m3"],
    )
    links_criticos = []
    for lid, serie in result.link_results.items():
        cr = max((p.capacity_ratio or 0.0 for p in serie), default=0.0)
        if cr >= 0.95:
            links_criticos.append({"tramo": lid, "capacidad_max": round(cr, 2)})
    links_criticos.sort(key=lambda x: -x["capacidad_max"])

    resumen = {
        "escenario": nombre,
        "lluvia": meta | {
            "duracion_min": duracion_min,
            "intervalo_calculo_min": interval_min,
        },
        "hidrologia": {
            "tragantes_con_hidrograma": len(hydro.hydrographs),
            "volumen_captado_m3": round(vol_captado, 0),
            "caudal_pico_total_m3s": round(caudal_pico_total, 2),
        },
        "swmm": {
            "inp": cfg["swmm"]["inp_path"],
            "tragantes_mapeados": len(node_mapping),
            "nodos_con_inundacion": len(inundados),
            "vol_total_inundacion_m3": round(sum(n["vol_inundacion_m3"] for n in inundados), 0),
            "top_nodos_inundados": inundados[:15],
            "tramos_a_capacidad": len(links_criticos),
            "top_tramos_criticos": links_criticos[:15],
            "resumen_engine": result.summary,
        },
    }

    outdir = OUT_BASE / nombre
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    with open(outdir / "hietograma.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["minuto", "intensidad_mm_h"])
        for p in puntos:
            w.writerow([p.minute, p.intensity_mm_h])
    print(f"[{nombre}] listo → {outdir/'resumen.json'}")
    return resumen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--escenario", choices=["real", "diseno", "ambos"], default="ambos")
    ap.add_argument("--fecha", default="2016-06-14", help="día del evento real (YYYY-MM-DD)")
    ap.add_argument("--retorno", type=int, default=10, choices=[5, 10, 25])
    ap.add_argument("--interval", type=float, default=5.0, help="paso hidrológico (min)")
    args = ap.parse_args()

    resumenes = []
    if args.escenario in ("real", "ambos"):
        fecha = dt.date.fromisoformat(args.fecha)
        puntos, dur, meta = hietograma_evento_real(fecha)
        resumenes.append(correr_escenario(
            f"evento_real_{fecha.isoformat()}", puntos, dur, meta, args.interval))
    if args.escenario in ("diseno", "ambos"):
        puntos, dur, meta = hietograma_bloques_alternos(T=args.retorno)
        resumenes.append(correr_escenario(
            f"diseno_T{args.retorno}_santa_lucia", puntos, dur, meta, args.interval))

    for r in resumenes:
        print()
        print(f"=== {r['escenario']} ===")
        print(f"  lluvia total: {r['lluvia']['total_mm']} mm | pico: {r['lluvia']['pico_mm_h']} mm/h")
        print(f"  volumen captado por tragantes: {r['hidrologia']['volumen_captado_m3']:,.0f} m³")
        print(f"  caudal pico total: {r['hidrologia']['caudal_pico_total_m3s']} m³/s")
        print(f"  nodos con inundación: {r['swmm']['nodos_con_inundacion']} | "
              f"volumen: {r['swmm']['vol_total_inundacion_m3']:,.0f} m³")
        print(f"  tramos a ≥95% de capacidad: {r['swmm']['tramos_a_capacidad']}")


if __name__ == "__main__":
    main()
