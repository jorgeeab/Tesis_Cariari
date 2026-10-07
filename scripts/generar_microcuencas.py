# -*- coding: utf-8 -*-
"""Genera microcuencas urbanas agrupando tragantes por punto de desague.

Metodo:
 1. Se construye el grafo dirigido de la red pluvial (links src_id -> dst_id).
 2. Cada tragante se traza aguas abajo hasta su nodo terminal; en bifurcaciones
    se sigue el ramal de mayor caida (drop_m).
 3. Tragantes cuyo tramo muere sin llegar a un desfogue se asignan al grupo del
    tragante resuelto mas cercano (vecindad espacial).
 4. Se generan poligonos de Thiessen/Voronoi por tragante, se recortan al area
    de estudio y se disuelven por desfogue -> microcuencas.
 5. El rio receptor (Bermudez / Virilla) se asigna por cercania del desfogue a
    los cauces recortados al area.

Salidas:
  output/microcuencas/microcuencas_desfogue.shp
  output/microcuencas/tragantes_asignados.shp
  output/microcuencas/mapa_microcuencas.png
"""
from pathlib import Path
from collections import defaultdict

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, MultiPoint
from shapely.ops import voronoi_diagram, unary_union

BASE = Path(__file__).resolve().parents[1]
SHP = BASE / "output" / "shapefiles_v7"
OUT = BASE / "output" / "microcuencas"
OUT.mkdir(parents=True, exist_ok=True)
CRS = "EPSG:5367"

# ---------------------------------------------------------------- carga
tragantes = gpd.read_file(SHP / "tragantes_puntos_ids.shp")
desfogues = gpd.read_file(SHP / "desfogues_puntos_ids.shp")
links = gpd.read_file(SHP / "red_pluvial_links.shp")
area = gpd.read_file(
    BASE / "Arcgis shapes y curvas" / "Shapes Tragantes y alcantarillado"
    / "Area de estudio actaulizado" / "Area de estudio actaulizado.shp"
).to_crs(CRS)
area_geom = unary_union(area.geometry)

# ------------------------------------------------- trazado aguas abajo
salidas = defaultdict(list)  # nodo -> [(drop_m, destino), ...]
for _, r in links.iterrows():
    salidas[r.src_id].append((r.drop_m if pd.notna(r.drop_m) else 0.0, r.dst_id))


def trazar(nodo: str) -> str:
    """Sigue la red aguas abajo (ramal de mayor caida) hasta el nodo terminal."""
    visitados = set()
    while nodo in salidas and nodo not in visitados:
        visitados.add(nodo)
        nodo = max(salidas[nodo])[1]
    return nodo


tragantes["desfogue"] = [trazar(t) for t in tragantes["id"]]
resueltos = tragantes["desfogue"].str.startswith("desfogue")
print(f"Tragantes resueltos por red: {resueltos.sum()} / {len(tragantes)}")

# --------------------- asignacion espacial de tragantes sin desfogue
ok = tragantes[resueltos]
pend = tragantes[~resueltos]
if len(pend):
    joined = gpd.sjoin_nearest(
        pend.drop(columns="desfogue"), ok[["desfogue", "geometry"]],
        how="left", distance_col="dist_asig",
    )
    joined = joined[~joined.index.duplicated()]
    tragantes.loc[joined.index, "desfogue"] = joined["desfogue"]
    tragantes["asignacion"] = np.where(resueltos, "red", "vecindad")
    print(f"Asignados por vecindad: {len(pend)} "
          f"(dist. media {joined['dist_asig'].mean():.0f} m)")

# ------------------------------------------------- rio receptor
bb4326 = area.to_crs("EPSG:4326").total_bounds
margen = 0.004  # ~440 m, para incluir el Virilla al sur del limite
cauces = gpd.read_file(
    BASE / "Arcgis shapes y curvas" / "Cauce.shp",
    bbox=(bb4326[0] - margen, bb4326[1] - margen,
          bb4326[2] + margen, bb4326[3] + margen))
cauces = cauces.to_crs(CRS)
# recorte a la ventana del mapa; se descartan fragmentos y quebradas menores
vista = box(*gpd.GeoSeries([area_geom], crs=CRS).buffer(280).total_bounds)
cauces["geometry"] = cauces.geometry.intersection(vista)
cauces = cauces[~cauces.geometry.is_empty]
cauces = cauces[cauces.geometry.length >= 900]
# El Virilla bordea el limite sur; el Bermudez cruza el centro del area.
ymin, ymax = area_geom.bounds[1], area_geom.bounds[3]
ymed = (ymin + ymax) / 2


def rio_de(pt) -> str:
    cerca = cauces.distance(pt).idxmin()
    y = cauces.geometry[cerca].centroid.y
    return "Virilla" if y < ymed - (ymax - ymin) * 0.15 else "Bermudez"


desfogues["rio"] = [rio_de(p) for p in desfogues.geometry]

# ------------------------------------------------- voronoi + disolucion
envolvente = box(*area_geom.buffer(200).bounds)
celdas = voronoi_diagram(MultiPoint(tragantes.geometry.tolist()),
                         envelope=envolvente)
cel_gdf = gpd.GeoDataFrame(geometry=list(celdas.geoms), crs=CRS)
cel_gdf = gpd.sjoin(cel_gdf, tragantes[["id", "desfogue", "geometry"]],
                    predicate="contains").drop(columns="index_right")
cel_gdf["geometry"] = cel_gdf.geometry.intersection(area_geom)

micro = cel_gdf.dissolve(by="desfogue", as_index=False)[["desfogue", "geometry"]]

# ---- fusion de microcuencas menores (p.ej. desfogue_6, encajonada entre
# vecinas mas grandes): se une a la vecina con mayor frontera compartida
FUSIONAR = ["desfogue_6"]
for objetivo in FUSIONAR:
    fila = micro[micro.desfogue == objetivo]
    if fila.empty:
        continue
    geom = fila.geometry.iloc[0]
    otras = micro[micro.desfogue != objetivo]
    frontera = otras.geometry.apply(
        lambda g: g.boundary.intersection(geom.boundary).length)
    destino = otras.loc[frontera.idxmax(), "desfogue"]
    print(f"Fusionando {objetivo} -> {destino}")
    tragantes.loc[tragantes.desfogue == objetivo, "desfogue"] = destino
    cel_gdf.loc[cel_gdf.desfogue == objetivo, "desfogue"] = destino
    micro = cel_gdf.dissolve(by="desfogue", as_index=False)[
        ["desfogue", "geometry"]]
# suavizado leve de bordes dentados del voronoi
micro["geometry"] = micro.geometry.buffer(6).buffer(-6).intersection(area_geom)

micro = micro.merge(desfogues[["id", "rio"]], left_on="desfogue", right_on="id")
micro = micro.drop(columns="id")
micro["n_trag"] = micro["desfogue"].map(
    tragantes.groupby("desfogue").size())
micro["area_ha"] = micro.geometry.area / 10_000

# codigo tipo MC-B1.. / MC-V1.. ordenado oeste->este por rio
micro["cx"] = micro.geometry.centroid.x
micro = micro.sort_values(["rio", "cx"]).reset_index(drop=True)
codigos, cont = [], defaultdict(int)
for r in micro.itertuples():
    letra = "B" if r.rio == "Bermudez" else "V"
    cont[letra] += 1
    codigos.append(f"MC-{letra}{cont[letra]}")
micro["codigo"] = codigos
micro = micro.drop(columns="cx")

micro.to_file(OUT / "microcuencas_desfogue.shp", encoding="utf-8")
tragantes.to_file(OUT / "tragantes_asignados.shp", encoding="utf-8")
print(micro[["codigo", "desfogue", "rio", "n_trag", "area_ha"]]
      .to_string(index=False))
print(f"Area total: {micro.area_ha.sum():.1f} ha "
      f"(area de estudio: {area_geom.area/10_000:.1f} ha)")

# ------------------------------------------------- mapa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# tonos de una misma familia por rio receptor
n_b = (micro.rio == "Bermudez").sum()
n_v = (micro.rio == "Virilla").sum()
tonos_b = plt.cm.Oranges(np.linspace(0.30, 0.62, n_b))
tonos_v = plt.cm.Blues(np.linspace(0.30, 0.62, n_v))
colores, ib, iv = {}, 0, 0
for r in micro.itertuples():
    if r.rio == "Bermudez":
        colores[r.codigo] = tonos_b[ib]; ib += 1
    else:
        colores[r.codigo] = tonos_v[iv]; iv += 1

fig, ax = plt.subplots(figsize=(13, 10))
for r in micro.itertuples():
    gpd.GeoSeries([r.geometry], crs=CRS).plot(
        ax=ax, color=colores[r.codigo], edgecolor="white", linewidth=1.4,
        alpha=0.9)
    c = r.geometry.representative_point()
    ax.annotate(f"{r.codigo}\n({r.area_ha:.1f} ha)", (c.x, c.y),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#1a1a1a", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.65))

# rios: Virilla al sur, Bermudez cruzando el centro
rios_v = cauces[cauces.geometry.centroid.y < ymed - (ymax - ymin) * 0.15]
rios_b = cauces.drop(rios_v.index)
rios_b.plot(ax=ax, color="#0b4f9e", linewidth=3.0, zorder=4)
rios_v.plot(ax=ax, color="#0b4f9e", linewidth=3.0, zorder=4)


from shapely.ops import nearest_points


def etiqueta_rio(gdf, texto, dx=0, dy=0):
    if gdf.empty:
        return
    g = gdf.loc[gdf.geometry.length.idxmax()].geometry
    p = nearest_points(g, area_geom.centroid)[0]
    ax.annotate(texto, (p.x + dx, p.y + dy), fontsize=14, fontstyle="italic",
                fontweight="bold", color="#0b4f9e", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0b4f9e",
                          alpha=0.85))


etiqueta_rio(rios_b, "Río Bermúdez", dx=-320, dy=70)
etiqueta_rio(rios_v, "Río Virilla", dx=-250, dy=-90)

# red pluvial + flechas de direccion de flujo
links.plot(ax=ax, color="#4a4a4a", linewidth=0.6, alpha=0.7)
flechas = gpd.read_file(SHP / "red_pluvial_flechas.shp")
# `angle` es rumbo en grados horarios desde el norte -> u=sin, v=cos
ang = np.deg2rad(flechas["angle"].astype(float).values)
ax.quiver(flechas.geometry.x, flechas.geometry.y,
          np.sin(ang), np.cos(ang), color="#4a4a4a", zorder=3,
          scale=110, width=0.0011, headwidth=3.8, headlength=4.2, alpha=0.8)

desfogues.plot(ax=ax, color="red", marker="v", markersize=55, zorder=5,
               edgecolor="black", linewidth=0.7)
gpd.GeoSeries([area_geom.boundary], crs=CRS).plot(
    ax=ax, color="red", linewidth=2)

xmin, ymin_v, xmax, ymax_v = vista.bounds
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin_v, ymax_v)

leyenda = [
    Patch(facecolor=plt.cm.Oranges(0.45), label="Microcuencas → Río Bermúdez"),
    Patch(facecolor=plt.cm.Blues(0.45), label="Microcuencas → Río Virilla"),
    Line2D([0], [0], color="#0b4f9e", lw=3, label="Ríos"),
    Line2D([0], [0], color="#4a4a4a", lw=1, marker=">", markersize=6,
           label="Red pluvial (dirección de flujo)"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="red",
           markeredgecolor="black", markersize=9, label="Desfogue"),
    Line2D([0], [0], color="red", lw=2, label="Área de estudio"),
]
ax.legend(handles=leyenda, loc="lower right", fontsize=11, framealpha=0.92)
ax.set_title("Microcuencas de drenaje pluvial según río receptor — "
             "Ciudad Cariari, Belén", fontsize=15, fontweight="bold")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(OUT / "mapa_microcuencas.png", dpi=200, bbox_inches="tight")
print(f"Mapa: {OUT / 'mapa_microcuencas.png'}")
