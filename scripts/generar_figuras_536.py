#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figuras de la sección 5.3.6 "Representación gráfica de la propuesta":
plano de ubicación de los sitios SbN (capa SIG real) y dibujos técnicos
preliminares (planta / perfil / sección con capas constructivas, entrada,
salida, rebalse, vegetación y drenaje inferior) con las dimensiones del
dimensionamiento preliminar de la sección 5.3.5.

Salidas:
- Documentos Tesis/figuras_mapa3d/fig_536_*.png
- figuras_corregidas/figuras_49_52_sbn/figura_*.png|pdf
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "Documentos Tesis" / "figuras_mapa3d"
OUT_CORR = BASE / "figuras_corregidas" / "figuras_49_52_sbn"
REG_DIR = BASE / "figuras_corregidas" / "registros"
for _dir in (OUT, OUT_CORR, REG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
DPI_ENTREGA = 300
SALIDAS_REGISTRO: list[dict[str, object]] = []

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TINTA = "#212529"
SUAVE = "#66707a"
AGUA = "#7fb6d9"
SUSTRATO = "#c9a06b"
TIERRA = "#a8815a"
GRAVA = "#c3c8cf"
CONCRETO = "#dfe3e8"
ASIENTO = "#e6c98a"
VERDE = "#2f9e44"
PIEDRA = "#9aa0a6"

COLOR_TEC = {
    "jardin": "#2f9e44",
    "pozo": "#7048e8",
    "cemento": "#868e96",
    "zanja": "#e8590c",
}


# ── helpers de dibujo ────────────────────────────────────────────────
CAJA_TXT = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.88)


def dim_h(ax, x0, x1, y, texto, offset=0.05, fs=9):
    ax.annotate("", (x0, y), (x1, y), arrowprops=dict(arrowstyle="<->", color=TINTA, lw=1))
    ax.text((x0 + x1) / 2, y + offset, texto, ha="center", va="bottom", fontsize=fs,
            color=TINTA, bbox=CAJA_TXT, zorder=9)


def dim_v(ax, x, y0, y1, texto, offset=0.06, fs=9, rot=90, ha="left"):
    ax.annotate("", (x, y0), (x, y1), arrowprops=dict(arrowstyle="<->", color=TINTA, lw=1))
    ax.text(x + offset, (y0 + y1) / 2, texto, ha=ha, va="center", fontsize=fs,
            color=TINTA, rotation=rot, bbox=CAJA_TXT, zorder=9)


def flecha(ax, x0, y0, dx, dy, texto, color="#1c7ed6", fs=9, ha="left", tx=None, ty=None):
    ax.add_patch(FancyArrow(x0, y0, dx, dy, width=0.015, head_width=0.09,
                            head_length=0.09, color=color, length_includes_head=True))
    ax.text(tx if tx is not None else x0, ty if ty is not None else y0 + 0.12,
            texto, fontsize=fs, color=color, ha=ha, fontweight="bold",
            bbox=CAJA_TXT, zorder=10)


def plantitas(ax, xs, y, escala=0.12, color=VERDE):
    for x in xs:
        ax.plot([x, x], [y, y + escala], color=color, lw=1.4)
        ax.plot([x, x - escala * 0.6], [y + escala * 0.55, y + escala], color=color, lw=1.2)
        ax.plot([x, x + escala * 0.6], [y + escala * 0.55, y + escala], color=color, lw=1.2)


def tubo_perforado(ax, x, y, r=0.055):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=TINTA, lw=1.2, zorder=6))
    for dx, dy in ((-r * 0.45, 0), (r * 0.45, 0), (0, -r * 0.45)):
        ax.add_patch(Circle((x + dx, y + dy), r * 0.09, facecolor=TINTA, edgecolor="none", zorder=7))


def preparar(ax, xlim, ylim, titulo):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=11, color=TINTA, pad=8)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guardar(fig, nombre_base: str, nombre_corregido: str | None = None) -> None:
    legacy = OUT / f"{nombre_base}.png"
    fig.savefig(legacy, facecolor="white", dpi=DPI_ENTREGA)
    if nombre_corregido is None:
        return

    png = OUT_CORR / f"{nombre_corregido}.png"
    pdf = OUT_CORR / f"{nombre_corregido}.pdf"
    fig.savefig(png, facecolor="white", dpi=DPI_ENTREGA)
    fig.savefig(pdf, facecolor="white")
    ancho_in, alto_in = fig.get_size_inches()
    SALIDAS_REGISTRO.append({
        "archivo": f"figuras_49_52_sbn/{nombre_corregido}",
        "png_px": [round(ancho_in * DPI_ENTREGA), round(alto_in * DPI_ENTREGA)],
        "ancho_cm_a_300_ppp": round(ancho_in * 2.54, 2),
        "alto_cm_a_300_ppp": round(alto_in * 2.54, 2),
        "ppp": DPI_ENTREGA,
        "md5_png": _md5(png),
        "md5_pdf": _md5(pdf),
    })


def escribir_registro() -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=BASE,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = "desconocido"
    registro = {
        "script": str(Path(__file__).resolve()),
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "commit_repositorio": commit,
        "python": sys.version.split()[0],
        "matplotlib": matplotlib.__version__,
        "figuras": [49, 50, 51, 52],
        "criterio": "Regeneracion a 300 ppp y PDF vectorial; rotulos con coma decimal.",
        "salidas": SALIDAS_REGISTRO,
    }
    path = REG_DIR / "figuras_49_52_sbn.json"
    path.write_text(json.dumps(registro, indent=2, ensure_ascii=False), encoding="utf-8")
    print("registro:", path)


# ── F1: plano de ubicación ──────────────────────────────────────────
def fig_ubicacion():
    import geopandas as gpd

    shp_dir = BASE / "Arcgis shapes y curvas"
    trag = shp_dir / "Shapes Tragantes y alcantarillado"
    sitios = gpd.read_file(trag / "Area para SBN" / "Area para SBN.shp")
    links = gpd.read_file(BASE / "output/shapefiles_v7/red_pluvial_links.shp")
    sitios = sitios.to_crs(links.crs)
    area = gpd.read_file(trag / "area de estudio.shp").to_crs(links.crs)

    minx, miny, maxx, maxy = sitios.total_bounds
    m = 260.0
    bbox = (minx - m, miny - m, maxx + m, maxy + m)
    from shapely.geometry import box
    bbox4326 = tuple(gpd.GeoSeries([box(*bbox)], crs=links.crs).to_crs(4326).total_bounds)
    cauce = gpd.read_file(shp_dir / "Cauce.shp", bbox=bbox4326).to_crs(links.crs)

    def clase(tipo: str) -> str:
        t = tipo.lower()
        if "jardin" in t or "jardín" in t:
            return "jardin"
        if "pozo" in t:
            return "pozo"
        if "cemento" in t:
            return "cemento"
        return "zanja"

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=150)
    area.boundary.plot(ax=ax, color="#adb5bd", linewidth=1.2, linestyle="--", zorder=1)
    cauce.plot(ax=ax, color="#2b5f8a", linewidth=1.8, zorder=2)
    links.plot(ax=ax, color="#9ec9e8", linewidth=1.0, zorder=3)

    # sitios del corredor central: etiquetas escalonadas para evitar traslapes
    sitios = sitios.copy()
    sitios["cx"] = sitios.geometry.centroid.x
    sitios["cy"] = sitios.geometry.centroid.y
    aislados = {1, 8, 16}
    cluster = sitios[~sitios["id"].isin(aislados)].sort_values("cx")
    ciclo = [(0, 11), (0, -17), (0, 27), (0, -33)]
    offsets = {}
    for k, (idx, _) in enumerate(cluster.iterrows()):
        offsets[idx] = ciclo[k % len(ciclo)]

    for idx, fila in sitios.iterrows():
        c = COLOR_TEC[clase(str(fila["Tipo SBN"]))]
        gpd.GeoSeries([fila.geometry]).plot(ax=ax, facecolor=c, edgecolor=c, alpha=0.55, linewidth=1.5, zorder=4)
        etiqueta = "SBN-S/I" if fila["id"] != fila["id"] else f"SBN-{int(fila['id']):02d}"
        dx, dy = offsets.get(idx, (0, 8))
        ax.annotate(etiqueta, (fila["cx"], fila["cy"]), xytext=(dx, dy),
                    textcoords="offset points", ha="center", fontsize=8, color=TINTA,
                    fontweight="bold", zorder=6,
                    arrowprops=dict(arrowstyle="-", color="#98a1a8", lw=0.6) if abs(dy) > 18 else None)

    from matplotlib.lines import Line2D
    ley = [
        Line2D([], [], color="#9ec9e8", lw=1.6, label="Red pluvial digitalizada"),
        Line2D([], [], color="#2b5f8a", lw=2, label="Cauces (ríos Bermúdez y Virilla)"),
        Line2D([], [], color="#adb5bd", lw=1.4, ls="--", label="Área de estudio"),
        Line2D([], [], color=COLOR_TEC["jardin"], marker="s", lw=0, ms=10, label="Jardín de lluvia"),
        Line2D([], [], color=COLOR_TEC["pozo"], marker="s", lw=0, ms=10, label="Pozo de infiltración"),
        Line2D([], [], color=COLOR_TEC["cemento"], marker="s", lw=0, ms=10, label="Cemento permeable"),
        Line2D([], [], color=COLOR_TEC["zanja"], marker="s", lw=0, ms=10, label="Zanja infiltrante"),
    ]
    ax.legend(handles=ley, loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#dde2e6")
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect("equal")
    ax.set_xlabel("Este CRTM05 (m)", fontsize=9, color=SUAVE)
    ax.set_ylabel("Norte CRTM05 (m)", fontsize=9, color=SUAVE)
    ax.tick_params(labelsize=8, colors=SUAVE)
    for s in ax.spines.values():
        s.set_color("#dde2e6")
    ax.set_title("Ubicación de los sitios propuestos para SbN — Bosques de Doña Rosa, Belén",
                 fontsize=12, color=TINTA, pad=12)
    fig.tight_layout()
    guardar(fig, "fig_536_ubicacion")
    plt.close(fig)
    print("ubicacion lista")


# ── F2: jardín de lluvia ────────────────────────────────────────────
def fig_jardin():
    fig, (ap, asx) = plt.subplots(2, 1, figsize=(10, 9.6), dpi=150,
                                  gridspec_kw={"height_ratios": [1.05, 1]})
    # PLANTA 5,0 × 5,0 m
    preparar(ap, (-1.6, 6.9), (-1.1, 6.1), "Planta — jardín de lluvia (sitio SBN-01, 5,0 × 5,0 m)")
    ap.add_patch(Rectangle((0, 0), 5, 5, facecolor="#dff0e2", edgecolor=VERDE, lw=2))
    ap.add_patch(Rectangle((0.55, 0.55), 3.9, 3.9, facecolor="#c9e7cf", edgecolor=VERDE, lw=1, ls="--"))
    plantitas(ap, [0.9 + i * 0.72 for i in range(5)], 1.15, 0.3)
    plantitas(ap, [1.25 + i * 0.72 for i in range(5)], 2.35, 0.3)
    plantitas(ap, [0.9 + i * 0.72 for i in range(5)], 3.55, 0.3)
    for k in range(7):  # entrada con piedra disipadora
        ap.add_patch(Circle((-0.28 + 0.13 * (k % 2), 1.8 + k * 0.22), 0.085, facecolor=PIEDRA, edgecolor="none"))
    flecha(ap, -1.35, 2.5, 0.85, 0, "Entrada: escorrentía\nde calle/techos", tx=-1.5, ty=3.6)
    ap.text(-0.25, 1.35, "disipador\nde piedra", fontsize=8, color=SUAVE, ha="center")
    ap.add_patch(Rectangle((5.0, 3.6), 0.5, 0.8, facecolor="#f1f3f5", edgecolor=TINTA, lw=1.2))
    ap.text(5.25, 3.05, "caja de\nrebalse", fontsize=8, color=TINTA, ha="center")
    flecha(ap, 5.55, 4.0, 0.85, 0, "Salida / rebalse\na red pluvial", tx=5.6, ty=4.85)
    dim_h(ap, 0, 5, -0.45, "5,00 m", offset=-0.42)
    dim_v(ap, 0.3, 0, 5, "5,00 m", offset=0.07)
    ap.text(2.5, 5.35, "vegetación tolerante a saturación y sequía", fontsize=8.5, color=VERDE, ha="center")

    # SECCIÓN
    preparar(asx, (-1.6, 6.9), (-1.65, 1.15), "Sección transversal — capas constructivas")
    x0, x1 = 0, 5
    asx.add_patch(Rectangle((x0, -1.15), x1 - x0, 0.25, facecolor=GRAVA, edgecolor="none"))       # grava
    asx.add_patch(Rectangle((x0, -0.9), x1 - x0, 0.55, facecolor=SUSTRATO, edgecolor="none"))     # sustrato
    asx.add_patch(Rectangle((x0, -0.35), x1 - x0, 0.25, facecolor=AGUA, edgecolor="none"))        # lámina
    asx.plot([x0, x1], [-0.35 + 0.25, -0.35 + 0.25], color=TINTA, lw=1)                            # nivel máx
    asx.plot([x0 - 1.2, x0, x0, x1, x1, x1 + 1.2], [0.32, 0.32, -1.15, -1.15, 0.32, 0.32], color=TINTA, lw=2)
    asx.plot([x0, x1], [-0.35, -0.35], color=TINTA, lw=0.8, ls=":")
    asx.plot([x0, x1], [-0.9, -0.9], color=TINTA, lw=0.8, ls=":")
    asx.plot([x0 - 0.04, x0 - 0.04], [-1.15, 0.32], color="#555", lw=1.4, ls=(0, (2, 2)))
    asx.plot([x1 + 0.04, x1 + 0.04], [-1.15, 0.32], color="#555", lw=1.4, ls=(0, (2, 2)))
    asx.text(x1 + 0.2, -1.45, "geotextil perimetral", fontsize=8, color="#555")
    plantitas(asx, [0.6 + i * 0.62 for i in range(7)], -0.1, 0.34)
    for k in range(6):
        asx.add_patch(Circle((-0.62 + 0.16 * (k % 2), -0.28 + k * 0.11), 0.07, facecolor=PIEDRA, edgecolor="none"))
    flecha(asx, -1.45, 0.28, 0.62, 0, "Entrada", tx=-1.5, ty=0.42)
    asx.add_patch(Rectangle((5.35, -0.45), 0.5, 0.85, facecolor="#f1f3f5", edgecolor=TINTA, lw=1.2))
    asx.plot([5.0, 5.35], [-0.12, -0.12], color=TINTA, lw=2)
    flecha(asx, 5.9, -0.3, 0.7, 0, "Rebalse a\nred pluvial", tx=5.62, ty=0.72)
    tubo_perforado(asx, 2.5, -1.03)
    asx.annotate("drenaje inferior opcional:\ntubería perforada Ø100 mm",
                 (2.5, -1.03), xytext=(3.35, -1.5), fontsize=8.5, color=TINTA,
                 arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.9))
    asx.text(0.12, -0.24, "lámina útil", fontsize=8.5, color="#1b4b66")
    asx.text(0.12, -0.68, "sustrato filtrante", fontsize=8.5, color="#5a3d1e")
    asx.text(0.12, -1.06, "grava (transición/dren)", fontsize=8.5, color="#3c4148")
    dim_v(asx, 4.62, -0.35, -0.1, "0,25 m", offset=-0.12, fs=8, rot=0, ha="right")
    dim_v(asx, 4.62, -0.1, 0.32, "0,10 m", offset=-0.12, fs=8, rot=0, ha="right")
    asx.text(4.15, 0.46, "borde libre", fontsize=7.5, color=SUAVE, ha="center", bbox=CAJA_TXT)
    dim_v(asx, -0.95, -1.15, 0.32, "prof. constructiva ≈ 0,45–0,60 m", offset=-0.55, fs=8)
    fig.tight_layout()
    guardar(fig, "fig_536_jardin", "figura_49_jardin_lluvia")
    plt.close(fig)
    print("jardin listo")


# ── F3: pozo de infiltración ────────────────────────────────────────
def fig_pozo():
    fig, (ap, asx) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150,
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    # PLANTA
    preparar(ap, (-2.4, 3.6), (-1.2, 3.4), "Planta — pozo de infiltración (2,0 × 2,0 m)")
    ap.add_patch(Rectangle((-1.9, 0.55), 0.9, 0.9, facecolor="#f1f3f5", edgecolor=TINTA, lw=1.4))
    ap.text(-1.45, 1.75, "caja de\npretratamiento\n(rejilla y sedimentos)", fontsize=8, ha="center", color=TINTA)
    flecha(ap, -2.3, 1.0, 0.3, 0, "Entrada", tx=-2.35, ty=1.28)
    ap.plot([-1.0, 0], [1.0, 1.0], color=TINTA, lw=2)
    ap.add_patch(Rectangle((0, 0), 2, 2, facecolor=GRAVA, edgecolor=TINTA, lw=1.6, hatch="..."))
    ap.text(1.0, 2.5, "pozo relleno de grava limpia\n(porosidad ≈ 0,35)", fontsize=8.5, ha="center", color=TINTA)
    dim_h(ap, 0, 2, -0.4, "2,00 m", offset=-0.38)
    dim_v(ap, -0.35, 0, 2, "2,00 m", offset=-0.62)
    flecha(ap, 2.0, 1.55, 0.9, 0, "Rebalse a\ncaja pluvial", tx=2.15, ty=1.95)

    # SECCIÓN
    preparar(asx, (-2.6, 4.3), (-3.75, 1.3), "Sección — profundidad útil 2,0–2,5 m")
    asx.add_patch(Rectangle((0, -2.5), 2, 2.5, facecolor=GRAVA, edgecolor="none", hatch="..."))
    asx.plot([-2.4, 0, 0, 2, 2, 4.1], [0, 0, -2.5, -2.5, 0, 0], color=TINTA, lw=2)
    asx.plot([0.04, 0.04], [-2.5, 0], color="#555", lw=1.3, ls=(0, (2, 2)))
    asx.plot([1.96, 1.96], [-2.5, 0], color="#555", lw=1.3, ls=(0, (2, 2)))
    asx.text(2.1, -1.15, "geotextil", fontsize=8, color="#555", rotation=90)
    asx.add_patch(Rectangle((-1.7, -0.75), 0.9, 0.75, facecolor="#f1f3f5", edgecolor=TINTA, lw=1.3))
    asx.plot([-0.8, 0], [-0.45, -0.45], color=TINTA, lw=2)
    flecha(asx, -2.45, -0.18, 0.55, 0, "Entrada", tx=-2.5, ty=0.08)
    asx.text(-1.25, -1.1, "sedimentador\ncon rejilla", fontsize=8, ha="center", color=TINTA)
    flecha(asx, 2.0, -0.22, 1.15, 0, "Rebalse superior\na red pluvial", tx=2.1, ty=0.12)
    for xx in (0.4, 1.0, 1.6):
        asx.add_patch(FancyArrow(xx, -2.55, 0, -0.34, width=0.02, head_width=0.11,
                                 head_length=0.12, color="#1c7ed6", length_includes_head=True))
    asx.add_patch(FancyArrow(-0.05, -1.6, -0.36, 0, width=0.02, head_width=0.11,
                             head_length=0.12, color="#1c7ed6", length_includes_head=True))
    asx.add_patch(FancyArrow(2.05, -1.6, 0.36, 0, width=0.02, head_width=0.11,
                             head_length=0.12, color="#1c7ed6", length_includes_head=True))
    asx.text(1.0, -3.25, "infiltración por fondo y paredes\n(fondo ≥ 1,5 m sobre nivel freático)",
             fontsize=8.5, ha="center", color="#1b4b66", bbox=CAJA_TXT)
    dim_v(asx, 2.6, -2.5, 0, "2,0–2,5 m", offset=0.06)
    dim_h(asx, 0, 2, 0.35, "2,00 m", offset=0.05)
    fig.tight_layout()
    guardar(fig, "fig_536_pozo", "figura_50_pozo_infiltracion")
    plt.close(fig)
    print("pozo listo")


# ── F4: cemento permeable ───────────────────────────────────────────
def fig_pavimento():
    fig, asx = plt.subplots(figsize=(10, 5.6), dpi=150)
    preparar(asx, (-3.6, 9.6), (-2.5, 1.15),
             "Sección — cemento permeable sobre base granular (uso liviano; escala vertical exagerada)")
    x0, x1 = 0, 6
    # espesores reales 0,08 / 0,05 / 0,25 m dibujados ×3 para legibilidad
    hs, ha_, hr = 0.24, 0.15, 0.75
    y_sup, y_asi, y_res = -hs, -hs - ha_, -hs - ha_ - hr
    asx.add_patch(Rectangle((x0, y_res), x1 - x0, hr, facecolor=GRAVA, edgecolor="none", hatch="..."))
    asx.add_patch(Rectangle((x0, y_asi), x1 - x0, ha_, facecolor=ASIENTO, edgecolor="none"))
    asx.add_patch(Rectangle((x0, y_sup), x1 - x0, hs, facecolor=CONCRETO, edgecolor=TINTA, lw=0.6, hatch="////"))
    asx.plot([x0 - 3.0, x0], [0, 0], color=TINTA, lw=2)
    asx.plot([x1, x1 + 3.2], [0, 0], color=TINTA, lw=2)
    asx.plot([x0, x1], [y_res, y_res], color=TINTA, lw=1.4)
    asx.plot([x0, x0], [y_res, 0], color=TINTA, lw=1.2)
    asx.plot([x1, x1], [y_res, 0], color=TINTA, lw=1.2)
    asx.plot([x0 - 0.03, x1 + 0.03], [y_res - 0.05, y_res - 0.05], color="#555", lw=1.3, ls=(0, (2, 2)))
    asx.annotate("geotextil sobre\nsubrasante nivelada", (x0 + 0.45, y_res - 0.05),
                 xytext=(-3.45, y_res - 0.55), fontsize=8, color="#555", va="center",
                 arrowprops=dict(arrowstyle="-", color="#98a1a8", lw=0.7))
    # entrada (lluvia atraviesa la superficie)
    for xx in (1.0, 2.2, 3.4, 4.6, 5.6):
        asx.add_patch(FancyArrow(xx, 0.42, 0, -0.3, width=0.015, head_width=0.1,
                                 head_length=0.1, color="#1c7ed6", length_includes_head=True))
    asx.text(3.0, 0.62, "Entrada: lluvia y escorrentía atraviesan la superficie", fontsize=9,
             color="#1b4b66", ha="center")
    # salida principal por infiltración
    for xx in (1.4, 2.9, 4.4):
        asx.add_patch(FancyArrow(xx, y_res - 0.1, 0, -0.32, width=0.015, head_width=0.1,
                                 head_length=0.1, color="#1c7ed6", length_includes_head=True))
    asx.text(2.6, y_res - 0.75, "Salida principal: infiltración a la subrasante", fontsize=9,
             color="#1b4b66", ha="center")
    # capas: etiquetas a la izquierda con líneas guía
    etiquetas = [
        ("superficie permeable 0,08 m", y_sup + hs / 2),
        ("capa de asiento 0,05 m", y_asi + ha_ / 2),
        ("reservorio granular 0,25 m\n(vacíos ≈ 0,35)", y_res + hr / 2),
    ]
    for k, (txt, yy) in enumerate(etiquetas):
        asx.annotate(txt, (x0, yy), xytext=(-3.45, yy + (0.22 - 0.22 * k)),
                     fontsize=8.5, color=TINTA, va="center",
                     arrowprops=dict(arrowstyle="-", color="#98a1a8", lw=0.7))
    # subdrén y rebalse
    tubo_perforado(asx, 5.5, y_res + 0.18, r=0.09)
    asx.plot([5.5, 7.2], [y_res + 0.18, y_res + 0.18], color=TINTA, lw=1.6)
    flecha(asx, 7.2, y_res + 0.18, 0.85, 0, "Rebalse/subdrén a\ncuneta o caja pluvial",
           tx=7.0, ty=y_res + 0.62)
    asx.annotate("drenaje inferior (subdrén Ø100 mm)\nrecomendado si el suelo infiltra poco",
                 (5.5, y_res + 0.18), xytext=(5.05, -2.15), fontsize=8.5, color=TINTA,
                 arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.9))
    # cota total real (los espesores por capa van en las etiquetas de la izquierda)
    dim_v(asx, 6.5, y_res, 0, "0,38 m", offset=0.06, fs=8)
    dim_h(asx, x0, x1, 0.9, "franja permeable (accesos y tránsito liviano)", offset=0.05)
    asx.text(3.0, -2.4, "pendiente transversal ≤ 2 %; sin entrada de sedimentos finos durante la obra",
             fontsize=8, color=SUAVE, ha="center")
    fig.tight_layout()
    guardar(fig, "fig_536_pavimento", "figura_51_cemento_permeable")
    plt.close(fig)
    print("pavimento listo")


# ── F5: zanja infiltrante ───────────────────────────────────────────
def fig_zanja():
    fig = plt.figure(figsize=(11, 9.2), dpi=150)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 0.8, 1.15], hspace=0.42)
    apl = fig.add_subplot(gs[0])
    apf = fig.add_subplot(gs[1])
    asx = fig.add_subplot(gs[2])

    # PLANTA 70 m
    preparar(apl, (-6, 84), (-3.2, 4.6), "Planta — zanja infiltrante SBN-16 (70 m × 2,2 m)")
    apl.add_patch(Rectangle((0, 0), 70, 2.2, facecolor="#f6e3d3", edgecolor=COLOR_TEC["zanja"], lw=1.8))
    for xx in range(4, 68, 9):
        apl.add_patch(FancyArrow(xx, 3.6, 0, -1.1, width=0.25, head_width=1.3,
                                 head_length=0.5, color="#1c7ed6", length_includes_head=True))
    apl.text(33, 3.9, "Entrada distribuida: escorrentía lateral del corredor", fontsize=9, color="#1b4b66", ha="center")
    flecha(apl, 70, 1.1, 6.5, 0, "Rebalse aguas abajo\na red pluvial", tx=70.5, ty=1.9)
    dim_h(apl, 0, 70, -1.5, "70,0 m", offset=-1.4)
    dim_v(apl, -2.5, 0, 2.2, "2,2 m", offset=-1.4, fs=8.5)

    # PERFIL longitudinal
    preparar(apf, (-6, 84), (-3.6, 2.6), "Perfil longitudinal — pendiente suave hacia el rebalse")
    apf.plot([-4, 74], [1.6, 0.9], color=TINTA, lw=2)
    apf.text(-5.5, 1.75, "terreno", fontsize=8.5, color=SUAVE)
    apf.add_patch(Polygon([(0, 1.55), (70, 0.85), (70, -0.7), (0, 0.0)], closed=True,
                          facecolor=GRAVA, edgecolor=TINTA, lw=1.2, hatch="..."))
    apf.plot([0, 70], [-0.05, -0.75], color=TINTA, lw=1.6)
    tubo_perforado(apf, 35, -0.42, r=0.3)
    apf.annotate("tubería perforada Ø100 mm con pendiente al rebalse",
                 (35, -0.42), xytext=(8, -2.6), fontsize=8.5, color=TINTA,
                 arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.9))
    apf.add_patch(Rectangle((70, -1.0), 2.6, 2.0, facecolor="#f1f3f5", edgecolor=TINTA, lw=1.2))
    apf.text(71.3, 2.1, "caja de\nrebalse", fontsize=8, ha="center", color=TINTA)
    flecha(apf, 72.6, -0.2, 5.5, 0, "Salida a red pluvial", tx=73.4, ty=-1.9)

    # SECCIÓN trapezoidal
    preparar(asx, (-2.3, 4.6), (-1.55, 1.25), "Sección transversal — trapezoidal, prof. 0,45 m")
    xtl, xtr = 0.0, 2.2
    xbl, xbr = 0.55, 1.65
    prof = 0.9
    asx.add_patch(Polygon([(xtl, 0), (xtr, 0), (xbr, -prof), (xbl, -prof)], closed=True,
                          facecolor=GRAVA, edgecolor="none", hatch="..."))
    asx.add_patch(Polygon([(xtl, 0), (xtr, 0), (xtr - 0.14, -0.16), (xtl + 0.14, -0.16)], closed=True,
                          facecolor=TIERRA, edgecolor="none"))
    asx.plot([-2.0, xtl, xbl, xbr, xtr, 4.3], [0, 0, -prof, -prof, 0, 0], color=TINTA, lw=2)
    asx.plot([xtl + 0.05, xbl + 0.05, xbr - 0.05, xtr - 0.05],
             [-0.02, -prof + 0.04, -prof + 0.04, -0.02], color="#555", lw=1.3, ls=(0, (2, 2)))
    asx.text(2.45, -0.62, "geotextil", fontsize=8, color="#555")
    plantitas(asx, [0.28 + i * 0.4 for i in range(5)], 0.0, 0.2)
    asx.text(1.1, 0.42, "vegetación de protección en corona y taludes", fontsize=8.5, color=VERDE, ha="center")
    tubo_perforado(asx, 1.1, -0.74, r=0.075)
    asx.annotate("drenaje inferior Ø100 mm", (1.1, -0.74), xytext=(2.5, -1.15), fontsize=8.5,
                 color=TINTA, arrowprops=dict(arrowstyle="->", color=TINTA, lw=0.9))
    flecha(asx, -1.9, 0.28, 0.75, 0, "Entrada lateral", tx=-2.0, ty=0.45)
    dim_h(asx, xtl, xtr, 0.75, "2,20 m", offset=0.04, fs=8.5)
    dim_v(asx, 3.1, -prof, 0, "0,45 m (útil)", offset=0.06, fs=8.5)
    asx.text(1.1, -1.4, "grava limpia (porosidad ≈ 0,35) con capa filtrante superior", fontsize=8.5,
             color="#3c4148", ha="center")
    guardar(fig, "fig_536_zanja", "figura_52_zanja_infiltrante")
    plt.close(fig)
    print("zanja lista")


if __name__ == "__main__":
    fig_ubicacion()
    fig_jardin()
    fig_pozo()
    fig_pavimento()
    fig_zanja()
    escribir_registro()
    print("figuras 5.3.6 en", OUT)
