#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera las figuras del escrito "Construcción del modelo 3D" que no son
capturas del visor: render del plano G-08, mapa de capas fuente e
hietogramas de los escenarios simulados.

Salida: Documentos Tesis/figuras_mapa3d/
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "Documentos Tesis" / "figuras_mapa3d"
OUT.mkdir(parents=True, exist_ok=True)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

AZUL = "#1c7ed6"      # red pluvial (convención del visor)
NARANJA = "#e8590c"   # red residual
VIOLETA = "#7048e8"   # desfogues
VERDE = "#2f9e44"     # tragantes
GRIS_GUIA = "#495057"
GRIS_CURVA = "#c1c9d0"
TINTA = "#212529"
TINTA_SUAVE = "#666f76"


def fig_plano_g08() -> None:
    import fitz  # pymupdf

    pdf = BASE / "Arcgis shapes y curvas" / "G-08 SANITARIO OESTE.pdf"
    doc = fitz.open(pdf)
    page = doc[0]
    # página completa a ~150 dpi
    pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
    pix.save(OUT / "fig_plano_g08_completo.png")
    # recorte central ampliado (donde se leen las etiquetas S/ y F/)
    r = page.rect
    w, h = r.width, r.height
    clip = fitz.Rect(w * 0.35, h * 0.30, w * 0.65, h * 0.60)
    pix = page.get_pixmap(matrix=fitz.Matrix(400 / 72, 400 / 72), clip=clip)
    pix.save(OUT / "fig_plano_g08_detalle.png")
    print("plano G-08:", pix.width, "x", pix.height)


def fig_capas_fuente() -> None:
    import json

    import geopandas as gpd
    from shapely.geometry import LineString, box

    shp_dir = BASE / "Arcgis shapes y curvas"
    trag_dir = shp_dir / "Shapes Tragantes y alcantarillado"

    nodos = gpd.read_file(BASE / "output/shapefiles_v7/red_pluvial_nodos.shp")
    crs_local = nodos.crs
    minx, miny, maxx, maxy = nodos.total_bounds
    m = 200.0
    bbox = (minx - m, miny - m, maxx + m, maxy + m)
    bbox_4326 = tuple(
        gpd.GeoSeries([box(*bbox)], crs=crs_local).to_crs(4326).total_bounds
    )

    def load(path: Path, geographic: bool = False):
        try:
            gdf = gpd.read_file(path, bbox=bbox_4326 if geographic else bbox)
            if gdf.empty:
                return None
            return gdf.to_crs(crs_local) if geographic else gdf
        except Exception as exc:  # capa opcional: no aborta la figura
            print("  aviso: no se pudo cargar", path.name, "→", exc)
            return None

    # Curvas IGN 1:25k: mismas del WFS del SNIT, cacheadas por el generador
    lineas, cotas = [], []
    for nombre in ("indice_clipped.json", "intermedia_clipped.json"):
        cache = BASE / "resultados_refinados" / "_cache_curvas_clipped" / nombre
        if not cache.exists():
            continue
        for cota, coords in json.loads(cache.read_text(encoding="utf-8")):
            if len(coords) >= 2:
                lineas.append(LineString(coords))
                cotas.append(cota)
    curvas = (
        gpd.GeoDataFrame({"cota": cotas}, geometry=lineas, crs=4326).to_crs(crs_local)
        if lineas
        else None
    )

    cauce = load(shp_dir / "Cauce.shp", geographic=True)
    guias = load(trag_dir / "Alcantarillado.shp")
    tragantes = load(trag_dir / "Tragantes (2).shp")
    crp = load(trag_dir / "CRP actualizado" / "CRP actualizado.shp")
    crn = load(trag_dir / "CRN (2).shp")
    desf = load(shp_dir / "Desfogues.shp")

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    if curvas is not None:
        curvas.plot(ax=ax, color=GRIS_CURVA, linewidth=0.5, zorder=1)
    if cauce is not None:
        cauce.plot(ax=ax, color="#2b5f8a", linewidth=1.8, zorder=2)
    if guias is not None:
        guias.plot(ax=ax, color=GRIS_GUIA, linewidth=0.9, zorder=3)
    if tragantes is not None:
        tragantes.plot(ax=ax, color=VERDE, markersize=5, marker="o", zorder=4)
    if crp is not None:
        crp.plot(ax=ax, color=AZUL, markersize=14, marker="s", zorder=5)
    if crn is not None:
        crn.plot(ax=ax, color=NARANJA, markersize=14, marker="s", zorder=5)
    if desf is not None:
        desf.plot(ax=ax, color=VIOLETA, markersize=60, marker="v", zorder=6)

    from matplotlib.lines import Line2D

    leyenda = [
        Line2D([], [], color=GRIS_CURVA, lw=1.2, label="Curvas de nivel IGN 1:25 000 (SNIT)"),
        Line2D([], [], color="#2b5f8a", lw=2, label="Cauces (red fluvial IGN)"),
        Line2D([], [], color=GRIS_GUIA, lw=1.4, label="Trazos guía de alcantarillado (campo)"),
        Line2D([], [], color=AZUL, marker="s", lw=0, ms=7, label="Cajas de registro pluvial (CRP, 146)"),
        Line2D([], [], color=NARANJA, marker="s", lw=0, ms=7, label="Cajas de registro residual (CRN)"),
        Line2D([], [], color=VERDE, marker="o", lw=0, ms=5, label="Tragantes (254)"),
        Line2D([], [], color=VIOLETA, marker="v", lw=0, ms=9, label="Desfogues"),
    ]
    ax.legend(handles=leyenda, loc="lower right", fontsize=8.5, framealpha=0.95,
              edgecolor="#dde2e6")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Este CRTM05 (m)", fontsize=9, color=TINTA_SUAVE)
    ax.set_ylabel("Norte CRTM05 (m)", fontsize=9, color=TINTA_SUAVE)
    ax.tick_params(labelsize=8, colors=TINTA_SUAVE)
    for s in ax.spines.values():
        s.set_color("#dde2e6")
    ax.set_title("Capas fuente del modelo — Ciudad Cariari, Belén",
                 fontsize=12, color=TINTA, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_capas_fuente.png", facecolor="white")
    plt.close(fig)
    print("capas fuente listo")


def _leer_hietograma(path: Path) -> tuple[list[float], list[float]]:
    minutos, intensidades = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            minutos.append(float(row["minuto"]))
            intensidades.append(float(row["intensidad_mm_h"]))
    return minutos, intensidades


def fig_hietogramas() -> None:
    real = BASE / "results/simulaciones/evento_real_2016-06-14/hietograma.csv"
    dis = BASE / "results/simulaciones/diseno_T10_santa_lucia/hietograma.csv"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    for ax, (path, titulo, paso) in zip(axes, [
        (real, "Evento real 14-jun-2016 (ERA5)\n120,2 mm en 14 h", 60.0),
        (dis, "Tormenta de diseño T = 10 años (IDF Santa Lucía)\n82,9 mm en 60 min, bloques alternos", 5.0),
    ]):
        minutos, inten = _leer_hietograma(path)
        ax.bar([m + paso / 2 for m in minutos], inten, width=paso * 0.92,
               color=AZUL, edgecolor="white", linewidth=0.5, zorder=3)
        pico = max(inten)
        m_pico = minutos[inten.index(pico)] + paso / 2
        ax.annotate(f"{pico:.1f} mm/h", (m_pico, pico), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=TINTA, fontweight="bold")
        ax.set_title(titulo, fontsize=10.5, color=TINTA)
        ax.set_xlabel("Minuto del evento", fontsize=9, color=TINTA_SUAVE)
        ax.grid(axis="y", color="#e9edf0", zorder=0)
        ax.tick_params(labelsize=8.5, colors=TINTA_SUAVE)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#dde2e6")
        ax.set_ylim(0, 260)
    axes[0].set_ylabel("Intensidad (mm/h)", fontsize=9, color=TINTA_SUAVE)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hietogramas.png", facecolor="white")
    plt.close(fig)
    print("hietogramas listo")


if __name__ == "__main__":
    fig_plano_g08()
    fig_capas_fuente()
    fig_hietogramas()
    print("figuras en", OUT)
