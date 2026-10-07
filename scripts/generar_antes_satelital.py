#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Genera las vistas aéreas actuales ("antes") de sitios SbN seleccionados:
recorta tiles satelitales Esri World Imagery (z19) alrededor de cada sitio y
dibuja encima el polígono propuesto de la capa SIG.

Salida: Documentos Tesis/figuras_mapa3d/cap_536_antes_sbnXX.png
"""
from __future__ import annotations

import io
import math
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "Documentos Tesis" / "figuras_mapa3d"
SHP = (BASE / "Arcgis shapes y curvas" / "Shapes Tragantes y alcantarillado"
       / "Area para SBN" / "Area para SBN.shp")
Z = 19
TILE_URL = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}")
MARGEN_M = 45.0

SITIOS = {
    1: "#2f9e44",   # jardín de lluvia
    7: "#2f9e44",   # jardín de lluvia
    2: "#7048e8",   # pozo de infiltración
    8: "#7048e8",   # pozo de infiltración
    9: "#868e96",   # cemento permeable
    12: "#868e96",  # cemento permeable
    16: "#e8590c",  # zanja infiltrante
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def merc_pix(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 256 * (2 ** z)
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(x: int, y: int, z: int) -> Image.Image:
    req = urllib.request.Request(TILE_URL.format(z=z, y=y, x=x),
                                 headers={"User-Agent": "tesis-cariari-figuras"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def main() -> None:
    import geopandas as gpd

    sitios = gpd.read_file(SHP).to_crs(4326)
    for _, fila in sitios.iterrows():
        if fila["id"] != fila["id"] or int(fila["id"]) not in SITIOS:
            continue
        sid = int(fila["id"])
        color = SITIOS[sid]
        geom = fila.geometry
        lon0, lat0, lon1, lat1 = geom.bounds
        # margen en grados
        dlat = MARGEN_M / 110600.0
        dlon = MARGEN_M / (111320.0 * math.cos(math.radians((lat0 + lat1) / 2)))
        px0, py1 = merc_pix(lon0 - dlon, lat0 - dlat, Z)
        px1, py0 = merc_pix(lon1 + dlon, lat1 + dlat, Z)
        tx0, tx1 = int(px0 // 256), int(px1 // 256)
        ty0, ty1 = int(py0 // 256), int(py1 // 256)
        mosaico = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                mosaico.paste(fetch_tile(tx, ty, Z), ((tx - tx0) * 256, (ty - ty0) * 256))
        # recorte al bbox pedido
        cx0, cy0 = px0 - tx0 * 256, py0 - ty0 * 256
        cx1, cy1 = px1 - tx0 * 256, py1 - ty0 * 256
        recorte = mosaico.crop((int(cx0), int(cy0), int(cx1), int(cy1)))
        esc = 2 if max(recorte.size) < 700 else 1
        if esc > 1:
            recorte = recorte.resize((recorte.width * esc, recorte.height * esc), Image.LANCZOS)
        dr = ImageDraw.Draw(recorte)

        def a_pix(lon: float, lat: float) -> tuple[float, float]:
            gx, gy = merc_pix(lon, lat, Z)
            return (gx - px0) * esc, (gy - py0) * esc

        anillos = [geom.exterior] if geom.geom_type == "Polygon" else [g.exterior for g in geom.geoms]
        for anillo in anillos:
            pts = [a_pix(x, y) for x, y in anillo.coords]
            dr.line(pts, fill="white", width=9, joint="curve")  # halo para contraste
            dr.line(pts, fill=color, width=5, joint="curve")
        try:
            fuente = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            fuente = ImageFont.load_default()
        etiqueta = f"SBN-{sid:02d}"
        c = geom.centroid
        ex, ey = a_pix(c.x, c.y)
        caja = dr.textbbox((ex, ey), etiqueta, font=fuente, anchor="mm")
        dr.rectangle((caja[0] - 6, caja[1] - 3, caja[2] + 6, caja[3] + 3), fill=(255, 255, 255, 230))
        dr.text((ex, ey), etiqueta, fill=color, font=fuente, anchor="mm")
        out = OUT / f"cap_536_antes_sbn{sid:02d}.png"
        recorte.save(out)
        print("ok", out.name, recorte.size)


if __name__ == "__main__":
    main()
