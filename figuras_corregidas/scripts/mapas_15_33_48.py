#!/usr/bin/env python3
"""Regenera los mapas corregidos de las figuras 15, 33 y 48."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from shapely.geometry import LineString, box


REPO = Path(__file__).resolve().parents[2]
SIG = REPO / "Arcgis shapes y curvas"
RED = REPO / "output" / "shapefiles_v7"
CRS = "EPSG:5367"
CM = 1 / 2.54


def leer(relativa: str, crs: str = CRS) -> gpd.GeoDataFrame:
    capa = gpd.read_file(REPO / relativa)
    return capa.to_crs(crs) if capa.crs != crs else capa


def cauces_en(vista: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    ventana = gpd.GeoSeries([box(*vista)], crs=CRS).to_crs("EPSG:4326")
    capa = gpd.read_file(SIG / "Cauce.shp", bbox=tuple(ventana.total_bounds))
    return capa.to_crs(CRS).clip(box(*vista))


def curvas_en(vista: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    fuente = REPO / "resultados_refinados" / "_cache_curvas_clipped" / "intermedia_clipped.json"
    datos = json.loads(fuente.read_text(encoding="utf-8"))
    geometrias = [LineString(coords) for _, coords in datos if len(coords) >= 2]
    cotas = [cota for cota, coords in datos if len(coords) >= 2]
    capa = gpd.GeoDataFrame({"cota": cotas}, geometry=geometrias,
                            crs="EPSG:4326").to_crs(CRS)
    return capa.clip(box(*vista))


def preparar_figura(ancho_cm: float, alto_cm: float):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.edgecolor": "#5b6570", "axes.linewidth": 0.7,
        "xtick.color": "#5b6570", "ytick.color": "#5b6570",
    })
    return plt.subplots(figsize=(ancho_cm * CM, alto_cm * CM))


def decorar(ax, vista, escala: int, columnas_leyenda: int,
            leyenda, bottom: float = 0.17) -> None:
    xmin, ymin, xmax, ymax = vista
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    ax.tick_params(axis="y", labelrotation=90, labelsize=7)
    ax.tick_params(axis="x", labelsize=7)

    # Norte, en coordenadas relativas para conservarlo al cambiar la extension.
    x, y = 0.95, 0.91
    norte = Polygon([[x, y + 0.015], [x - 0.012, y - 0.045], [x, y - 0.032],
                     [x + 0.012, y - 0.045]], closed=True,
                    transform=ax.transAxes, facecolor="white",
                    edgecolor="#202124", linewidth=0.8, zorder=20)
    ax.add_patch(norte)
    ax.add_patch(Polygon([[x, y + 0.015], [x, y - 0.032],
                          [x + 0.012, y - 0.045]], closed=True,
                         transform=ax.transAxes, facecolor="#202124",
                         edgecolor="#202124", linewidth=0.5, zorder=21))
    ax.text(x, y + 0.025, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=8, fontweight="bold")

    # Barra de escala alternada.
    sx = xmin + (xmax - xmin) * 0.04
    sy = ymin + (ymax - ymin) * 0.045
    alto = (ymax - ymin) * 0.009
    tramo = escala / 2
    for i, color in enumerate(("#202124", "white", "#202124", "white")):
        ax.add_patch(Rectangle((sx + i * tramo, sy), tramo, alto,
                               facecolor=color, edgecolor="#202124",
                               linewidth=0.5, zorder=20))
    ax.text(sx, sy + alto * 1.55, "0", fontsize=6.5, ha="center")
    ax.text(sx + escala, sy + alto * 1.55, str(escala), fontsize=6.5,
            ha="center")
    ax.text(sx + 2 * escala, sy + alto * 1.55, f"{2 * escala} m",
            fontsize=6.5, ha="center")
    ax.text(0.99, 0.015, "CRTM05 (EPSG:5367), coordenadas en metros",
            transform=ax.transAxes, ha="right", va="bottom",
            color="#5b6570", fontsize=6.5)
    ax.legend(handles=leyenda, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncols=columnas_leyenda, frameon=False, fontsize=7,
              handlelength=1.7, columnspacing=1.2)
    ax.figure.subplots_adjust(left=0.075, right=0.99, top=0.985, bottom=bottom)


def guardar(fig, salida: Path, nombre: str) -> None:
    base = salida / nombre
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(base)


def figura_33(salida: Path) -> None:
    vista = (481050, 1101860, 483410, 1103550)
    fig, ax = preparar_figura(15.9, 13.45)
    curvas_en(vista).plot(ax=ax, color="#cbd2d8", linewidth=0.45, zorder=1)
    cauces_en(vista).plot(ax=ax, color="#17639b", linewidth=1.4, zorder=3)
    alc = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Alcantarillado.shp")
    trag = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Tragantes (2).shp")
    crp = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/CRP actualizado/CRP actualizado.shp")
    crn = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/CRN (2).shp")
    desf = leer("Arcgis shapes y curvas/Desfogues.shp")
    alc.plot(ax=ax, color="#66717c", linewidth=0.7, zorder=4)
    trag.plot(ax=ax, color="#009e73", markersize=5, zorder=6)
    crp.plot(ax=ax, color="#0879b5", marker="s", markersize=6, zorder=6)
    crn.plot(ax=ax, color="#d95f02", marker="s", markersize=6, zorder=6)
    desf.plot(ax=ax, color="#cc79a7", marker="v", edgecolor="#202124",
              linewidth=0.4, markersize=32, zorder=7)
    leyenda = [
        Line2D([0], [0], color="#cbd2d8", label="Curvas de nivel IGN 1:25 000 (SNIT)"),
        Line2D([0], [0], color="#17639b", lw=1.5, label="Cauces"),
        Line2D([0], [0], color="#66717c", label=f"Trazos de alcantarillado ({alc.geometry.notna().sum()})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#009e73", markeredgecolor="none", label=f"Tragantes ({len(trag)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#0879b5", label=f"Cajas de registro pluvial ({len(crp)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#d95f02", label=f"Cajas de registro sanitario ({len(crn)})"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#cc79a7", markeredgecolor="#202124", label=f"Desfogues ({len(desf)})"),
    ]
    decorar(ax, vista, 250, 2, leyenda, 0.20)
    guardar(fig, salida, "figura_33_capas_fuente")


def clase_sbn(texto: str) -> str:
    texto = texto.lower()
    if "jardin" in texto:
        return "Jardín de lluvia"
    if "pozo" in texto:
        return "Pozo de infiltración"
    if "cemento" in texto:
        return "Cemento permeable"
    return "Zanja infiltrante"


def figura_48(salida: Path) -> None:
    vista = (481310, 1101880, 482690, 1103320)
    fig, ax = preparar_figura(15.9, 17.63)
    links = leer("output/shapefiles_v7/red_pluvial_links.shp")
    area = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/area de estudio.shp")
    sbn = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Area para SBN/Area para SBN.shp")
    sbn["clase"] = sbn["Tipo SBN"].map(clase_sbn)
    links.plot(ax=ax, color="#8bc0e3", linewidth=0.8, alpha=0.9, zorder=2)
    cauces_en(vista).plot(ax=ax, color="#17639b", linewidth=1.4, zorder=3)
    area.boundary.plot(ax=ax, color="#8b969f", linestyle="--", linewidth=0.9, zorder=4)
    estilos = {
        "Jardín de lluvia": ("#56bfa5", "D"),
        "Pozo de infiltración": ("#d9a3c8", "^"),
        "Cemento permeable": ("#a9a9a9", "s"),
        "Zanja infiltrante": ("#f1c75b", "s"),
    }
    for clase, (color, marcador) in estilos.items():
        grupo = sbn[sbn["clase"] == clase]
        grupo.plot(ax=ax, color=color, marker=marcador, edgecolor=color,
                   alpha=0.85, markersize=28, zorder=8)
        for fila in grupo.itertuples():
            sin_id = math.isnan(fila.id)
            codigo = "SBN-S/I" if sin_id else f"SBN-{int(fila.id):02d}"
            p = fila.geometry.representative_point()
            offsets = {
                2: (0, 18), 3: (-5, -24), 4: (-4, 10), 5: (5, 18),
                6: (0, 27), 7: (-12, 10), 9: (-2, -20), 11: (0, -30),
                12: (-12, -20), 13: (-10, -11), 14: (8, -30), 15: (10, -12),
            }
            despl = (0, -20) if sin_id else offsets.get(int(fila.id), (0, 8))
            ax.annotate(codigo, (p.x, p.y), xytext=despl,
                        textcoords="offset points", ha="center", va="center",
                        fontsize=5.5, fontweight="bold", color="#202124",
                        zorder=12)
    leyenda = [
        Line2D([0], [0], color="#8bc0e3", label="Red pluvial digitalizada"),
        Line2D([0], [0], color="#17639b", lw=1.5, label="Cauces"),
        Line2D([0], [0], color="#8b969f", ls="--", label="Área de estudio"),
    ]
    for clase, (color, _) in estilos.items():
        leyenda.append(Patch(facecolor=color,
                             label=f"{clase} ({(sbn['clase'] == clase).sum()})"))
    decorar(ax, vista, 100, 4, leyenda, 0.17)
    guardar(fig, salida, "figura_48_ubicacion_SbN")


def figura_15(salida: Path) -> None:
    vista = (481120, 1101860, 483470, 1103620)
    fig, ax = preparar_figura(15.9, 13.66)
    micro = leer("output/microcuencas/microcuencas_desfogue.shp")
    links = leer("output/shapefiles_v7/red_pluvial_links.shp")
    flechas = leer("output/shapefiles_v7/red_pluvial_flechas.shp")
    desf = leer("Arcgis shapes y curvas/Desfogues.shp")
    area = leer("Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/area de estudio.shp")
    for rio, mapa in (("Bermudez", plt.cm.Oranges), ("Virilla", plt.cm.Blues)):
        grupo = micro[micro["rio"] == rio].sort_values("codigo")
        colores = mapa(np.linspace(0.22, 0.68, len(grupo)))
        for color, fila in zip(colores, grupo.itertuples()):
            gpd.GeoSeries([fila.geometry], crs=CRS).plot(
                ax=ax, color=color, edgecolor="white", linewidth=0.8, zorder=1)
            p = fila.geometry.representative_point()
            ax.text(p.x, p.y, f"{fila.codigo}\n{str(round(fila.area_ha, 1)).replace('.', ',')} ha",
                    ha="center", va="center", fontsize=6.2, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78,
                          "pad": 1.2}, zorder=10)
    cauces_en(vista).plot(ax=ax, color="#17639b", linewidth=1.7, zorder=4)
    links.plot(ax=ax, color="#59636c", linewidth=0.45, alpha=0.8, zorder=5)
    ang = np.deg2rad(flechas["angle"].astype(float).to_numpy())
    ax.quiver(flechas.geometry.x, flechas.geometry.y, np.sin(ang), np.cos(ang),
              color="#59636c", zorder=6, scale=120, width=0.0012,
              headwidth=3.5, headlength=4)
    desf.plot(ax=ax, color="#d95f02", marker="v", edgecolor="#202124",
              linewidth=0.4, markersize=28, zorder=8)
    area.boundary.plot(ax=ax, color="#d95f02", linewidth=1.0, zorder=7)
    leyenda = [
        Patch(facecolor=plt.cm.Oranges(0.52), label="Microcuencas con descarga al río Bermúdez (MC-B)"),
        Patch(facecolor=plt.cm.Blues(0.52), label="Microcuencas con descarga al río Virilla (MC-V)"),
        Line2D([0], [0], color="#17639b", lw=1.5, label="Cauces"),
        Line2D([0], [0], color="#59636c", marker=">", label="Red pluvial y sentido de flujo"),
        Line2D([0], [0], marker="v", color="none", markerfacecolor="#d95f02", markeredgecolor="#202124", label="Desfogue"),
        Line2D([0], [0], color="#d95f02", label="Área de estudio"),
    ]
    decorar(ax, vista, 250, 3, leyenda, 0.20)
    guardar(fig, salida, "figura_15_microcuencas")


def generar(output_root: Path) -> None:
    salida = output_root / "mapas"
    salida.mkdir(parents=True, exist_ok=True)
    figura_33(salida)
    figura_48(salida)
    figura_15(salida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "figuras_corregidas")
    generar(parser.parse_args().output_root.resolve())
