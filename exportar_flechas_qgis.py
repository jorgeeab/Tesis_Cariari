"""
exportar_flechas_qgis.py
Genera shapefiles de FLECHAS de sentido de flujo para ver en QGIS, a partir de
los links de output/shapefiles_v6 (que ya vienen orientados src->dst por
gravedad, igual que los [CONDUITS] del INP de SWMM).

Salida (output/shapefiles_v6):
  red_pluvial_flechas.shp   puntos con campo `angle` (rumbo, grados horarios
  red_residual_flechas.shp  desde el norte) para rotar el marcador en QGIS.

Ademas copia los estilos .qml de styles/qgis/ junto a cada shapefile para que
QGIS los aplique automaticamente al cargar la capa (sidecar .qml):
  red_*_links.qml    linea + flecha central sobre el propio tramo
  red_*_flechas.qml  triangulo rotado por el campo `angle`

Misma logica de espaciado que las flechas del visor 3D:
una flecha cada ~45 m, minimo una por tramo, tramos < 8 m sin flecha.

Uso:  python exportar_flechas_qgis.py
"""

import math
import shutil
import sys
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    sys.exit("pip install geopandas")

BASE      = Path(__file__).parent
SHP_DIR   = BASE / "output/shapefiles_v7"
STYLE_DIR = BASE / "styles/qgis"

SPACING_M = 45.0   # separacion entre flechas a lo largo del tramo
MIN_LEN_M = 8.0    # tramos mas cortos no reciben flecha

REDES = ["red_pluvial", "red_residual"]


def build_arrow_points(links_gdf, red_name):
    """Puntos de flecha sobre cada tramo con rumbo (grados horarios desde N)."""
    rows = []
    for row in links_gdf.itertuples():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "LineString":
            continue
        L = geom.length
        if L < MIN_LEN_M:
            continue
        n_arrows = max(1, int(L // SPACING_M))
        for k in range(n_arrows):
            d = (k + 0.5) * (L / n_arrows)
            pos = geom.interpolate(d)
            # Tangente local: puntos a +-0.5 m sobre la linea
            p1 = geom.interpolate(max(0.0, d - 0.5))
            p2 = geom.interpolate(min(L, d + 0.5))
            dx, dy = p2.x - p1.x, p2.y - p1.y
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                continue
            angle = math.degrees(math.atan2(dx, dy)) % 360.0  # horario desde norte
            rows.append({
                "geometry": Point(pos.x, pos.y),
                "link_id":  str(row.id),
                "red":      red_name,
                "src_id":   str(row.src_id),
                "dst_id":   str(row.dst_id),
                "angle":    round(angle, 1),
            })
    return gpd.GeoDataFrame(rows, crs=links_gdf.crs)


def copy_styles():
    """Copia los .qml de styles/qgis/ junto a los shapefiles (sidecars)."""
    if not STYLE_DIR.exists():
        print(f"  AVISO: no existe {STYLE_DIR} (estilos no copiados)")
        return
    n = 0
    for qml in STYLE_DIR.glob("*.qml"):
        dst = SHP_DIR / qml.name
        shutil.copyfile(qml, dst)
        n += 1
    print(f"  Estilos copiados a {SHP_DIR.name}: {n} .qml")


def main():
    for red in REDES:
        links_path = SHP_DIR / f"{red}_links.shp"
        if not links_path.exists():
            print(f"  AVISO: no existe {links_path.name}")
            continue
        links = gpd.read_file(str(links_path))
        if links.crs and links.crs.to_epsg() != 5367:
            links = links.to_crs(epsg=5367)
        flechas = build_arrow_points(links, red.replace("red_", ""))
        out = SHP_DIR / f"{red}_flechas.shp"
        flechas.to_file(str(out), encoding="utf-8")
        print(f"  -> {out.name}  ({len(flechas)} flechas de {len(links)} tramos)")

    copy_styles()
    print("Listo.")


if __name__ == "__main__":
    main()
