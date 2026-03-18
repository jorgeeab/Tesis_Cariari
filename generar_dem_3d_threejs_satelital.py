#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de Modelo 3D Interactivo del Terreno - Área de Estudio Cariari, Costa Rica
======================================================================================
Genera un único archivo HTML autocontenido con Three.js r128 que muestra:
  - Malla 3D del terreno con textura satelital Esri (zoom 19)
  - Curvas de nivel IGN 1:25k descargadas via WFS
  - Rutas GPS medidas en campo
  - Zonas verdes desde KML local
  - Cortina translúcida del área de estudio (convex hull)
  - OrbitControls para navegar la escena

Uso:
    python generar_dem_3d_threejs_satelital.py

Salida:
    resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html
"""

import os
import sys
import math
import json
import base64
import time
import io
import re
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
import traceback
import struct
import zlib

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
GRID_RESOLUTION    = 220        # resolución de la malla DEM (NxN)
IDW_POWER          = 1.4        # potencia de la IDW
IDW_NEIGHBORS      = 20         # vecinos para IDW
SMOOTH_PASSES      = 1          # pasadas de suavizado 3×3
SAT_TEXTURE_ZOOM   = 19         # zoom de tile satelital (alto detalle)
TEXTURE_LONG_SIDE  = 3200       # px del lado largo de la textura final
Z_EXAG             = 1.0        # exageración vertical (1.0 = real)
FIELD_POINT_WEIGHT = 3          # peso de los puntos GPS vs curvas de nivel
MAX_CONTOUR_POINTS = 20_000     # límite de puntos de curvas por rendimiento
MARGIN_DEG         = 0.003      # margen en grados para el bounding box
TILE_RETRY         = 3          # reintentos de descarga de tiles

# URLs WFS – IGN 1:25 000
WFS_BASE = "https://geos.snitcr.go.cr/be/IGN_25/wfs?"
WFS_LAYERS = {
    "indice":        "IGN_25:RE_120101",
    "intermedia":    "IGN_25:RE_120102",
    "suplementaria": "IGN_25:RE_120103",
}
CURVA_COL_ELEV = "elevacion"   # nombre del campo de elevación en WFS

# Google Maps Hybrid (mejor cobertura para Costa Rica rural)
SAT_TMS = "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
SAT_SERVERS = [0, 1, 2, 3]  # round-robin entre servidores mt0..mt3

# OpenStreetMap / Overpass para red vial
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OSM_DRIVABLE_HIGHWAYS = {
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
    "secondary", "secondary_link", "tertiary", "tertiary_link",
    "residential", "unclassified", "living_street", "service", "road",
}

# Catastro SNIT (WMS raster overlay)
CATASTRO_WMS_URL   = "https://geos.snitcr.go.cr/be/TEST/wms"
CATASTRO_WMS_LAYER = "TEST:catastro_unificado_datosfincas"
CATASTRO_LONG_SIDE = 2048

# Colores de curvas en Three.js
CONTOUR_COLORS = {
    "indice":        "#FFD700",
    "intermedia":    "#FFFFFF",
    "suplementaria": "#A8FF60",
}

# Colores de rutas GPS
ROUTE_COLORS = ["#00BFFF", "#FF6347", "#39FF14", "#FFD700", "#FF69B4",
                "#00FA9A", "#FF4500", "#7B68EE", "#20B2AA", "#DC143C"]

# Directorio de salida
OUT_DIR   = Path("resultados_refinados")
VENDOR    = OUT_DIR / "_vendor_threejs"
CACHE_RAW = OUT_DIR / "_cache_curvas_raw"
CACHE_CLP = OUT_DIR / "_cache_curvas_clipped"
CACHE_OSM = OUT_DIR / "_cache_osm"
CACHE_CAT = OUT_DIR / "_cache_catastro"

# Archivos de datos adicionales (ríos y puntos de desfogue)
CAUCE_SHP        = Path("Arcgis shapes y curvas/Cauce.shp")
OSM_KML          = Path("Arcgis shapes y curvas/Open Street Map & Cariari, etc.kml")
ZONAS_VERDES_KML = Path("Arcgis shapes y curvas/Zonas Verdes.kml")
OUT_HTML         = OUT_DIR / "Refinado_30_ThreeJS_Cortina_Ajustada.html"

# ─────────────────────────────────────────────────────────────
#  IMPORTS OPCIONALES (con mensajes claros si faltan)
# ─────────────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠  requests no instalado – sin descarga de tiles/WFS. pip install requests")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠  Pillow no instalado – sin textura satelital. pip install Pillow")

try:
    import geopandas as gpd
    from shapely.geometry import MultiPoint, Point
    from shapely import wkt as shapely_wkt
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("⚠  geopandas/shapely no instalados. pip install geopandas shapely")

# ─────────────────────────────────────────────────────────────
#  UTILIDADES AUXILIARES
# ─────────────────────────────────────────────────────────────

def lonlat_to_tile_xy(lon_deg, lat_deg, zoom):
    """Web Mercator tile coordinates."""
    lat_r = math.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n)
    return x, y

def tile_to_lonlat(x, y, zoom):
    """Esquina NW de un tile en grados."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_r)
    return lon, lat

# ─────────────────────────────────────────────────────────────
#  1. AUTO-DETECCIÓN DEL ARCHIVO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def find_input_file():
    """Busca el archivo de entrada: KML con gx:Track preferido (tiene elevación), luego SHP."""
    script_dir = Path(__file__).parent

    # 1) KML conocido con datos de altitud (gx:Track de Map Plus)
    kml_specific = script_dir / "Arcgis shapes y curvas/25 1 26, etc.kml"
    if kml_specific.exists():
        return str(kml_specific), "kml"

    # 2) Cualquier KML recursivo (suelen tener elevación si vienen de GPS)
    kmls = list(script_dir.rglob("*.kml"))
    for kml in kmls:
        if "__pycache__" not in str(kml):
            return str(kml), "kml"

    # 3) SHP específico con altitudes
    shp_specific = script_dir / "Shapes/Datos de alturas/Other.shp"
    if shp_specific.exists():
        return str(shp_specific), "shp"

    # 4) Cualquier SHP recursivo (puede no tener Z)
    shps = list(script_dir.rglob("*.shp"))
    for shp in shps:
        if "__pycache__" not in str(shp) and "Curvas" not in str(shp) and "Cauce" not in str(shp):
            return str(shp), "shp"

    return None, None

# ─────────────────────────────────────────────────────────────
#  2. LECTURA DE RUTAS GPS
# ─────────────────────────────────────────────────────────────

def read_kml_tracks(kml_path):
    """
    Lee gx:Track de un KML de Map Plus y devuelve lista de rutas.
    Cada ruta es lista de (lon, lat, elev).
    """
    routes = []
    ns = {
        "kml": "http://www.opengis.net/kml/2.2",
        "gx":  "http://www.google.com/kml/ext/2.2",
    }

    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Eliminar prefijo de namespace del root si existe
    def strip_ns(tag):
        return tag.split('}')[-1] if '}' in tag else tag

    # Buscar todos los gx:Track (puede estar anidado en Folder/Placemark)
    for track in root.iter("{http://www.google.com/kml/ext/2.2}Track"):
        coords_elements = track.findall("{http://www.google.com/kml/ext/2.2}coord")
        pts = []
        for ce in coords_elements:
            parts = ce.text.strip().split()
            if len(parts) >= 3:
                try:
                    lon, lat, elev = float(parts[0]), float(parts[1]), float(parts[2])
                    pts.append((lon, lat, elev))
                except ValueError:
                    pass
        if len(pts) >= 2:
            routes.append(pts)

    print(f"  KML: {len(routes)} rutas leídas, {sum(len(r) for r in routes)} puntos total")
    return routes


def read_shp_routes(shp_path):
    """Lee LineString(Z) de un shapefile y devuelve lista de rutas."""
    if not HAS_GEO:
        print("  ERROR: geopandas requerido para leer SHP.")
        return []

    gdf = gpd.read_file(shp_path)
    # Reproyectar a WGS84 si es necesario
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    routes = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type in ("LineString", "MultiLineString"):
            lines = [geom] if geom.geom_type == "LineString" else geom.geoms
            for line in lines:
                try:
                    coords = list(line.coords)
                    if len(coords[0]) >= 3:
                        routes.append(coords)  # (x=lon, y=lat, z=elev)
                    else:
                        # Sin Z – se omite
                        pass
                except Exception:
                    pass

    print(f"  SHP: {len(routes)} rutas leídas, {sum(len(r) for r in routes)} puntos total")
    return routes

# ─────────────────────────────────────────────────────────────
#  3A. DESCARGA DE CURVAS IGN WFS
# ─────────────────────────────────────────────────────────────

def bbox_wfs_str(lon_min, lat_min, lon_max, lat_max):
    return f"{lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326"

def download_wfs_layer(layer_name, layer_key, lon_min, lat_min, lon_max, lat_max):
    """Descarga curvas de nivel de un layer WFS y devuelve lista de (elev, [(lon,lat),...])."""
    if not HAS_REQUESTS:
        return []

    cache_file = CACHE_RAW / f"{layer_key}.geojson"
    CACHE_RAW.mkdir(parents=True, exist_ok=True)

    if cache_file.exists():
        print(f"  Usando caché raw: {cache_file.name}")
        with open(cache_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = None
    else:
        print(f"  Descargando WFS: {layer_name} ...", end="", flush=True)
        bbox_str = bbox_wfs_str(lon_min, lat_min, lon_max, lat_max)
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeName": layer_name,
            "outputFormat": "JSON",
            "srsName": "EPSG:4326",
            "bbox": bbox_str,
            "count": "50000",
        }
        try:
            r = requests.get(WFS_BASE, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f" OK ({len(data.get('features', []))} features)")
        except Exception as e:
            print(f" ERROR: {e}")
            data = None

    if not data or "features" not in data:
        return []

    contours = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})

        # Elevación
        elev = props.get(CURVA_COL_ELEV) or props.get("ELEVACION") or props.get("elevation")
        try:
            elev = float(elev)
        except (TypeError, ValueError):
            continue

        # Geometría
        gtype = geom.get("type", "")
        coords_raw = geom.get("coordinates", [])

        if gtype == "LineString":
            pts = [(c[0], c[1]) for c in coords_raw if len(c) >= 2]
            if pts:
                contours.append((elev, pts))

        elif gtype == "MultiLineString":
            for seg in coords_raw:
                pts = [(c[0], c[1]) for c in seg if len(c) >= 2]
                if pts:
                    contours.append((elev, pts))

    print(f"    → {len(contours)} segmentos de curvas ({layer_key})")
    return contours


def get_all_contours(lon_min, lat_min, lon_max, lat_max):
    """Descarga y cachea todas las capas de curvas IGN."""
    CACHE_CLP.mkdir(parents=True, exist_ok=True)
    all_contours = {}

    for key, layer in WFS_LAYERS.items():
        cache_clp = CACHE_CLP / f"{key}_clipped.json"

        if cache_clp.exists():
            print(f"  Usando caché clipped: {cache_clp.name}")
            with open(cache_clp, encoding="utf-8") as f:
                try:
                    all_contours[key] = json.load(f)
                except Exception:
                    all_contours[key] = []
        else:
            data = download_wfs_layer(layer, key, lon_min, lat_min, lon_max, lat_max)
            # Recortar al bbox
            clipped = []
            for elev, pts in data:
                filtered = [(lon, lat) for lon, lat in pts
                            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max]
                if len(filtered) >= 2:
                    clipped.append([elev, filtered])
            all_contours[key] = clipped
            with open(cache_clp, "w", encoding="utf-8") as f:
                json.dump(clipped, f)
            print(f"    → {len(clipped)} segmentos clippados ({key})")

    return all_contours

# ─────────────────────────────────────────────────────────────
#  3B. CONSTRUCCIÓN DEL DEM CON IDW
# ─────────────────────────────────────────────────────────────

def idw_interpolation(known_pts, grid_lon, grid_lat, power=1.4, n_neighbors=20):
    """
    IDW sobre una grilla regular.
    known_pts: array (N,3) con (lon, lat, elev)
    grid_lon, grid_lat: arrays 2D de coordenadas de la grilla
    Devuelve: array 2D de elevaciones interpoladas.
    """
    pts = np.array(known_pts)  # (N, 3)
    g_lon = grid_lon.ravel()
    g_lat = grid_lat.ravel()
    result = np.zeros(len(g_lon))

    batch = 500  # procesar por lotes para no explotar RAM
    n = len(g_lon)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        qlon = g_lon[start:end]  # (B,)
        qlat = g_lat[start:end]  # (B,)

        # Distancias en grados (aproximación suficiente para área pequeña)
        dlon = pts[:, 0:1].T - qlon[:, None]  # (B, N)
        dlat = pts[:, 1:2].T - qlat[:, None]
        dist2 = dlon**2 + dlat**2               # (B, N)

        # Seleccionar n_neighbors más cercanos por fila
        if pts.shape[0] > n_neighbors:
            idx = np.argpartition(dist2, n_neighbors, axis=1)[:, :n_neighbors]
            d2  = np.take_along_axis(dist2, idx, axis=1)
            elev = pts[idx, 2]
        else:
            d2   = dist2
            elev = np.tile(pts[:, 2], (end - start, 1))

        # IDW
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / (d2 ** (power / 2.0) + 1e-20)
        result[start:end] = np.sum(w * elev, axis=1) / np.sum(w, axis=1)

    return result.reshape(grid_lon.shape)


def smooth_dem(dem, passes=1):
    """Suavizado por promedio 3×3."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = dem.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode='edge')
        windows = sliding_window_view(padded, (3, 3))
        out = windows.mean(axis=(-2, -1))
    return out


def build_dem(gps_routes, contours_dict, lon_min, lat_min, lon_max, lat_max):
    """
    Construye el DEM 220×220 combinando puntos GPS y curvas.
    Retorna: (dem_array 220×220, lons 1D, lats 1D)
    """
    known_pts = []

    # Puntos GPS con peso mayor
    total_gps = 0
    for route in gps_routes:
        for pt in route:
            lon, lat, elev = pt[0], pt[1], pt[2]
            for _ in range(FIELD_POINT_WEIGHT):
                known_pts.append((lon, lat, elev))
            total_gps += 1

    # Puntos de curvas de nivel
    total_contour = 0
    for key, segments in contours_dict.items():
        for seg in segments:
            elev = seg[0]
            pts  = seg[1]
            # Submuestrear si hay demasiados
            step = max(1, len(pts) // (MAX_CONTOUR_POINTS // max(1, sum(len(s[1]) for s in segments))))
            for i in range(0, len(pts), step):
                lon, lat = pts[i]
                known_pts.append((lon, lat, elev))
                total_contour += 1

    print(f"  Puntos GPS: {total_gps} (×{FIELD_POINT_WEIGHT} peso) | Puntos curvas: {total_contour}")
    print(f"  Total puntos de control: {len(known_pts)}")

    if not known_pts:
        raise ValueError("Sin puntos de control para interpolar el DEM.")

    # Grilla
    lons = np.linspace(lon_min, lon_max, GRID_RESOLUTION)
    lats = np.linspace(lat_min, lat_max, GRID_RESOLUTION)
    grid_lon, grid_lat = np.meshgrid(lons, lats)  # (ny, nx)

    print(f"  Interpolando IDW {GRID_RESOLUTION}×{GRID_RESOLUTION} ...")
    t0 = time.time()
    dem = idw_interpolation(known_pts, grid_lon, grid_lat, power=IDW_POWER, n_neighbors=IDW_NEIGHBORS)
    print(f"  IDW completado en {time.time()-t0:.1f}s")

    dem = smooth_dem(dem, SMOOTH_PASSES)
    print(f"  DEM: min={dem.min():.1f}m  max={dem.max():.1f}m")
    return dem, lons, lats

# ─────────────────────────────────────────────────────────────
#  4. TEXTURA SATELITAL
# ─────────────────────────────────────────────────────────────

CACHE_TILES = OUT_DIR / "_cache_tiles"

def download_tile(x, y, z, session, idx=0):
    """Descarga un tile Google Maps Hybrid con caché en disco y reintentos."""
    # Caché en disco
    cache_dir = CACHE_TILES / str(z) / str(y)
    cache_file = cache_dir / f"{x}.jpg"
    if cache_file.exists():
        return cache_file.read_bytes()

    server = SAT_SERVERS[idx % len(SAT_SERVERS)]
    url = SAT_TMS.format(s=server, z=z, y=y, x=x)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://maps.google.com/",
    }
    for attempt in range(TILE_RETRY):
        try:
            r = session.get(url, timeout=20, headers=headers)
            if r.status_code == 200 and len(r.content) > 500:  # evitar tiles vacíos
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(r.content)
                return r.content
        except Exception as e:
            if attempt == TILE_RETRY - 1:
                print(f"    Tile {z}/{y}/{x} falló: {e}")
            time.sleep(0.5)
    return None


def build_satellite_texture(lon_min, lat_min, lon_max, lat_max):
    """
    Extrae la textura del área de estudio recortando la imagen satelital local de alta resolución.
    Usamos rasterio para leer sólo la ventana necesaria del JPG masivo (EPSG:3395).
    """
    img_path = Path("Arcgis shapes y curvas/Imagen Satelital_TESIS.jpg")
    if not img_path.exists():
        print(f"  Imagen local no encontrada: {img_path}")
        return None

    try:
        import rasterio
        from rasterio.windows import from_bounds
        from pyproj import Transformer
        import numpy as np
    except ImportError:
        print("  Falta rasterio o pyproj. Ejecuta: pip install rasterio pyproj")
        return None

    print(f"  Transformando BBox a CRS de la imagen...")
    t = Transformer.from_crs(4326, 3395, always_xy=True)
    x_min, y_min = t.transform(lon_min, lat_min)
    x_max, y_max = t.transform(lon_max, lat_max)

    print(f"  Extrayendo textura de la imagen local (~400MB) ...")
    with rasterio.open(img_path) as src:
        # Calcular ventana que corresponde a nuestro BBox
        window = from_bounds(x_min, y_min, x_max, y_max, transform=src.transform)
        # Leer solo esa ventana
        data = src.read(window=window)
        if data.size == 0:
            print("  El área de estudio está fuera de los límites de la imagen satelital.")
            return None

    # rasterio lee (bandas, alto, ancho), PIL necesita (alto, ancho, bandas)
    if data.shape[0] >= 3:
        data = data[:3, :, :] # RGB solamente si tiene canal alpha

    img_arr = np.transpose(data, (1, 2, 0))
    img = Image.fromarray(img_arr)
    
    # Escalar al tamaño máximo para no sobrecargar el navegador
    cw, ch = img.size
    print(f"  Textura original extraída: {cw}×{ch} px")
    if cw >= ch:
        new_w = TEXTURE_LONG_SIDE
        new_h = int(ch * TEXTURE_LONG_SIDE / cw)
    else:
        new_h = TEXTURE_LONG_SIDE
        new_w = int(cw * TEXTURE_LONG_SIDE / ch)
    new_w = max(new_w, 1); new_h = max(new_h, 1)

    resized = img.resize((new_w, new_h), Image.LANCZOS)
    print(f"  Textura escalada final: {new_w}×{new_h} px")

    buf = io.BytesIO()
    resized.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────
#  5. CÁLCULO CONVEX HULL DEL ÁREA DE ESTUDIO
# ─────────────────────────────────────────────────────────────

def compute_convex_hull(gps_routes):
    """Devuelve lista de (lon, lat) del convex hull de todos los puntos GPS."""
    all_pts = []
    for route in gps_routes:
        for pt in route:
            all_pts.append((pt[0], pt[1]))

    if not all_pts:
        return []

    if HAS_GEO:
        mp = MultiPoint(all_pts)
        hull = mp.convex_hull
        if hull.geom_type == "Polygon":
            return list(hull.exterior.coords)
        elif hull.geom_type == "LineString":
            return list(hull.coords)
        else:
            return all_pts[:1]
    else:
        # Fallback: bounding box
        lons = [p[0] for p in all_pts]
        lats = [p[1] for p in all_pts]
        return [
            (min(lons), min(lats)), (max(lons), min(lats)),
            (max(lons), max(lats)), (min(lons), max(lats)),
            (min(lons), min(lats)),
        ]

# ─────────────────────────────────────────────────────────────
#  5B. DATOS ADICIONALES: RÍOS Y PUNTOS DE DESFOGUE
# ─────────────────────────────────────────────────────────────

def read_osm_kml_points(kml_path, lon_min, lat_min, lon_max, lat_max):
    """
    Lee puntos (Point) del KML de OpenStreetMap/Map Plus.
    Retorna lista de (lon, lat, elev, name).
    """
    pts = []
    if not kml_path.exists():
        print(f"  KML puntos no encontrado: {kml_path}")
        return pts

    tree = ET.parse(str(kml_path))
    root = tree.getroot()

    for placemark in root.iter("{http://www.opengis.net/kml/2.2}Placemark"):
        name_el = placemark.find("{http://www.opengis.net/kml/2.2}name")
        name = name_el.text.strip() if name_el is not None else "Punto"

        point_el = placemark.find("{http://www.opengis.net/kml/2.2}Point")
        if point_el is None:
            continue
        coords_el = point_el.find("{http://www.opengis.net/kml/2.2}coordinates")
        if coords_el is None:
            continue

        parts = coords_el.text.strip().split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                elev = float(parts[2]) if len(parts) >= 3 else 950.0
                if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                    pts.append((lon, lat, elev, name))
            except ValueError:
                pass

    print(f"  Puntos de desfogue/muestreo: {len(pts)} dentro del área")
    return pts


def parse_kml_coordinates(coords_text):
    """Parsea el bloque <coordinates> de KML a lista de (lon, lat, elev)."""
    coords = []
    if not coords_text:
        return coords

    for token in re.split(r"\s+", coords_text.strip()):
        if not token:
            continue
        parts = token.split(",")
        if len(parts) < 2:
            continue
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            elev = float(parts[2]) if len(parts) >= 3 and parts[2] else 0.0
            coords.append((lon, lat, elev))
        except ValueError:
            continue

    if len(coords) >= 2 and coords[0][:2] == coords[-1][:2]:
        coords = coords[:-1]

    return coords


def read_kml_polygons(kml_path, lon_min, lat_min, lon_max, lat_max):
    """
    Lee polígonos de un KML y retorna:
      [{ "name": str, "outer": [(lon, lat), ...], "holes": [[(lon, lat), ...], ...] }]
    Solo conserva polígonos que intersectan el bbox del área.
    """
    polygons = []
    if not kml_path.exists():
        print(f"  KML de polígonos no encontrado: {kml_path}")
        return polygons

    ns = "{http://www.opengis.net/kml/2.2}"
    tree = ET.parse(str(kml_path))
    root = tree.getroot()

    for placemark in root.iter(f"{ns}Placemark"):
        name_el = placemark.find(f"{ns}name")
        base_name = name_el.text.strip() if name_el is not None and name_el.text else "Zona verde"

        for idx, poly_el in enumerate(placemark.findall(f".//{ns}Polygon"), start=1):
            outer_el = poly_el.find(f"{ns}outerBoundaryIs/{ns}LinearRing/{ns}coordinates")
            if outer_el is None or not outer_el.text:
                continue

            outer_coords = parse_kml_coordinates(outer_el.text)
            outer_ring = [(lon, lat) for lon, lat, _ in outer_coords]
            if len(outer_ring) < 3:
                continue

            lons = [pt[0] for pt in outer_ring]
            lats = [pt[1] for pt in outer_ring]
            intersects_bbox = not (
                max(lons) < lon_min or min(lons) > lon_max or
                max(lats) < lat_min or min(lats) > lat_max
            )
            if not intersects_bbox:
                continue

            holes = []
            for inner_el in poly_el.findall(f"{ns}innerBoundaryIs/{ns}LinearRing/{ns}coordinates"):
                if inner_el is None or not inner_el.text:
                    continue
                inner_coords = parse_kml_coordinates(inner_el.text)
                inner_ring = [(lon, lat) for lon, lat, _ in inner_coords]
                if len(inner_ring) >= 3:
                    holes.append(inner_ring)

            name = base_name if idx == 1 else f"{base_name} {idx}"
            polygons.append({
                "name": name,
                "outer": outer_ring,
                "holes": holes,
            })

    print(f"  Zonas verdes: {len(polygons)} poligonos dentro del area")
    return polygons


def read_cauce_rivers(shp_path, lon_min, lat_min, lon_max, lat_max):
    """
    Lee ríos/cauces de Cauce.shp, recortados al bbox.
    Retorna lista de listas de (lon, lat).
    """
    if not shp_path.exists():
        print(f"  Cauce.shp no encontrado: {shp_path}")
        return []
    if not HAS_GEO:
        print("  geopandas no disponible para leer ríos.")
        return []

    print("  Leyendo Cauce.shp ...", end="", flush=True)
    try:
        # Leer con bbox para no cargar todo el archivo (~75MB)
        from shapely.geometry import box
        bbox_poly = box(lon_min, lat_min, lon_max, lat_max)
        gdf = gpd.read_file(str(shp_path), bbox=(lon_min, lat_min, lon_max, lat_max))

        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        rivers = []
        for geom in gdf.geometry:
            if geom is None:
                continue
            lines = []
            if geom.geom_type == "LineString":
                lines = [geom]
            elif geom.geom_type == "MultiLineString":
                lines = list(geom.geoms)

            for line in lines:
                coords = [(c[0], c[1]) for c in line.coords
                          if lon_min <= c[0] <= lon_max and lat_min <= c[1] <= lat_max]
                if len(coords) >= 2:
                    rivers.append(coords)

        print(f" {len(rivers)} segmentos de ríos")
        return rivers
    except Exception as e:
        print(f" Error: {e}")
        return []


def rivers_to_threejs(rivers, lon_center, lat_center, dem, lons, lats):
    """Convierte ríos a segmentos 3D, elevando al DEM."""
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0
    lon_min_d, lon_max_d = lons[0], lons[-1]
    lat_min_d, lat_max_d = lats[0], lats[-1]
    ny, nx = dem.shape

    def sample_dem(lon, lat):
        fi = (lon - lon_min_d) / (lon_max_d - lon_min_d) * (nx - 1)
        fj = (lat - lat_min_d) / (lat_max_d - lat_min_d) * (ny - 1)
        i = int(np.clip(round(fi), 0, nx - 1))
        j = int(np.clip(round(fj), 0, ny - 1))
        return float(dem[j, i]) * Z_EXAG

    segs_3d = []
    skip = max(1, sum(len(r) for r in rivers) // 10000)
    for seg in rivers:
        pts_3d = []
        for k, (lon, lat) in enumerate(seg):
            if k % skip != 0 and k != len(seg) - 1:
                continue
            x_m = (lon - lon_center) * mpd_lon
            y_m = (lat - lat_center) * mpd_lat
            z_m = sample_dem(lon, lat) + 0.8  # 0.8 m sobre el terreno
            pts_3d.append([round(x_m, 2), round(z_m, 2), round(-y_m, 2)])
        if len(pts_3d) >= 2:
            segs_3d.append(pts_3d)

    return segs_3d


def drainage_points_to_threejs(pts, lon_center, lat_center, dem, lons, lats):
    """Convierte puntos de desfogue a esferas 3D."""
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0
    lon_min_d, lon_max_d = lons[0], lons[-1]
    lat_min_d, lat_max_d = lats[0], lats[-1]
    ny, nx = dem.shape

    def sample_dem(lon, lat):
        fi = (lon - lon_min_d) / (lon_max_d - lon_min_d) * (nx - 1)
        fj = (lat - lat_min_d) / (lat_max_d - lat_min_d) * (ny - 1)
        i = int(np.clip(round(fi), 0, nx - 1))
        j = int(np.clip(round(fj), 0, ny - 1))
        return float(dem[j, i]) * Z_EXAG

    out = []
    for (lon, lat, elev, name) in pts:
        x_m = (lon - lon_center) * mpd_lon
        y_m = (lat - lat_center) * mpd_lat
        z_m = max(elev * Z_EXAG, sample_dem(lon, lat)) + 2.0
        out.append({
            "x": round(x_m, 2),
            "y": round(z_m, 2),
            "z": round(-y_m, 2),
            "name": name,
        })
    return out


def green_zones_to_threejs(polygons, lon_center, lat_center):
    """Convierte polígonos lon/lat a anillos 2D en metros (plano XZ de Three.js)."""
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0

    def ring_to_xz(ring):
        out = []
        for lon, lat in ring:
            x_m = (lon - lon_center) * mpd_lon
            y_m = (lat - lat_center) * mpd_lat
            out.append([round(x_m, 2), round(-y_m, 2)])
        return out

    out = []
    for poly in polygons:
        outer = ring_to_xz(poly["outer"])
        holes = [ring_to_xz(ring) for ring in poly.get("holes", []) if len(ring) >= 3]
        if len(outer) >= 3:
            out.append({
                "name": poly.get("name", "Zona verde"),
                "outer": outer,
                "holes": holes,
            })
    return out


def download_osm_streets(lon_min, lat_min, lon_max, lat_max):
    """
    Descarga calles/carreteras de OSM vía Overpass para el bbox dado.
    Retorna lista de:
      [{ "name": str, "highway": str, "coords": [(lon, lat), ...] }]
    """
    if not HAS_REQUESTS:
        print("  requests no disponible para descargar calles OSM.")
        return []

    CACHE_OSM.mkdir(parents=True, exist_ok=True)
    bbox_key = f"{lon_min:.5f},{lat_min:.5f},{lon_max:.5f},{lat_max:.5f}"
    cache_hash = hashlib.sha1(bbox_key.encode("utf-8")).hexdigest()[:12]
    cache_file = CACHE_OSM / f"osm_streets_{cache_hash}.json"

    if cache_file.exists():
        print(f"  Usando caché OSM: {cache_file.name}")
        with open(cache_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                data = None
    else:
        bbox = f"{lat_min},{lon_min},{lat_max},{lon_max}"
        query = (
            "[out:json][timeout:25];"
            "("
            f'way["highway"]({bbox});'
            ");"
            "out geom;"
        )
        print("  Descargando calles desde OSM/Overpass ...", end="", flush=True)
        try:
            r = requests.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=60)
            r.raise_for_status()
            data = r.json()
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(" OK")
        except Exception as e:
            print(f" Error: {e}")
            return []

    elements = (data or {}).get("elements", [])
    streets = []
    for el in elements:
        if el.get("type") != "way":
            continue

        tags = el.get("tags", {})
        highway = tags.get("highway", "")
        if highway not in OSM_DRIVABLE_HIGHWAYS:
            continue

        coords = []
        for pt in el.get("geometry", []):
            lon = pt.get("lon")
            lat = pt.get("lat")
            if lon is None or lat is None:
                continue
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                coords.append((lon, lat))

        if len(coords) >= 2:
            streets.append({
                "name": tags.get("name", ""),
                "highway": highway,
                "coords": coords,
            })

    print(f"  Calles OSM: {len(streets)} segmentos dentro del área")
    return streets


def streets_to_threejs(streets, lon_center, lat_center, dem, lons, lats):
    """Convierte calles OSM a líneas 3D apoyadas sobre el DEM."""
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0
    lon_min_d, lon_max_d = lons[0], lons[-1]
    lat_min_d, lat_max_d = lats[0], lats[-1]
    ny, nx = dem.shape

    def sample_dem(lon, lat):
        fi = (lon - lon_min_d) / (lon_max_d - lon_min_d) * (nx - 1)
        fj = (lat - lat_min_d) / (lat_max_d - lat_min_d) * (ny - 1)
        i = int(np.clip(round(fi), 0, nx - 1))
        j = int(np.clip(round(fj), 0, ny - 1))
        return float(dem[j, i]) * Z_EXAG

    skip = max(1, sum(len(s["coords"]) for s in streets) // 18000)
    out = []
    for street in streets:
        pts_3d = []
        for k, (lon, lat) in enumerate(street["coords"]):
            if k % skip != 0 and k != len(street["coords"]) - 1:
                continue
            x_m = (lon - lon_center) * mpd_lon
            y_m = (lat - lat_center) * mpd_lat
            z_m = sample_dem(lon, lat) + 0.35
            pts_3d.append([round(x_m, 2), round(z_m, 2), round(-y_m, 2)])

        if len(pts_3d) >= 2:
            out.append({
                "name": street.get("name", ""),
                "highway": street.get("highway", ""),
                "points": pts_3d,
            })

    return out


def download_catastro_overlay(lon_min, lat_min, lon_max, lat_max):
    """
    Descarga el catastro SNIT como overlay PNG transparente por WMS.
    Retorna bytes PNG o None.
    """
    if not HAS_REQUESTS:
        print("  requests no disponible para descargar catastro.")
        return None

    CACHE_CAT.mkdir(parents=True, exist_ok=True)
    bbox_key = f"{lon_min:.5f},{lat_min:.5f},{lon_max:.5f},{lat_max:.5f}"
    cache_hash = hashlib.sha1(bbox_key.encode("utf-8")).hexdigest()[:12]
    cache_file = CACHE_CAT / f"catastro_{cache_hash}.png"

    if cache_file.exists():
        print(f"  Usando caché catastro: {cache_file.name}")
        return cache_file.read_bytes()

    lon_span = max(lon_max - lon_min, 1e-9)
    lat_span = max(lat_max - lat_min, 1e-9)
    if lon_span >= lat_span:
        width = CATASTRO_LONG_SIDE
        height = max(1, int(round(CATASTRO_LONG_SIDE * lat_span / lon_span)))
    else:
        height = CATASTRO_LONG_SIDE
        width = max(1, int(round(CATASTRO_LONG_SIDE * lon_span / lat_span)))

    params = {
        "service": "WMS",
        "version": "1.1.1",
        "request": "GetMap",
        "layers": CATASTRO_WMS_LAYER,
        "styles": "",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "width": str(width),
        "height": str(height),
        "srs": "EPSG:4326",
        "format": "image/png",
        "transparent": "true",
    }

    print("  Descargando overlay catastral SNIT ...", end="", flush=True)
    try:
        r = requests.get(CATASTRO_WMS_URL, params=params, timeout=90)
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "image/png" not in ctype or len(r.content) < 100:
            print(" Error: respuesta no PNG válida")
            return None
        cache_file.write_bytes(r.content)
        print(" OK")
        return r.content
    except Exception as e:
        print(f" Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  6. GENERACIÓN DE DATOS 3D PARA THREE.JS
# ─────────────────────────────────────────────────────────────

def geo_to_meters(lon, lat, elev, lon_center, lat_center):
    """Convierte (lon, lat, elev) a metros relativos al centro."""
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0
    x_m = (lon - lon_center) * mpd_lon
    y_m = (lat - lat_center) * mpd_lat
    z_m = elev * Z_EXAG
    return x_m, y_m, z_m


def prepare_dem_for_threejs(dem, lons, lats, lon_center, lat_center):
    """
    Convierte el DEM a formato para THREE.PlaneGeometry.
    Three.js PlaneGeometry con rotation.x = -PI/2:
      - eje X: longitud (E-W)
      - eje Y: elevación
      - eje Z: -latitud (S→N en el mundo 3D es -Y real)

    PlaneGeometry espera los vértices en orden [col0..colN] por fila,
    con la primera fila en el "tope" (lat_max) y la última en el "fondo" (lat_min).
    Esto significa que el DEM se pasa con np.flipud.

    Devuelve: dict con geometry_data, width_m, height_m, nx, ny
    """
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0

    width_m  = (lons[-1] - lons[0]) * mpd_lon
    height_m = (lats[-1] - lats[0]) * mpd_lat

    nx = len(lons)
    ny = len(lats)

    z_min = float(dem.min())
    z_max = float(dem.max())

    # Flip vertical: Three.js PlaneGeometry empieza desde la fila superior (lat_max)
    dem_flipped = np.flipud(dem) * Z_EXAG

    # Aplanar row-major (el orden que espera Three.js: ix varía más rápido)
    heights = dem_flipped.ravel().tolist()

    return {
        "width_m":  round(width_m, 2),
        "height_m": round(height_m, 2),
        "nx":       nx,
        "ny":       ny,
        "z_min":    round(z_min * Z_EXAG, 2),
        "z_max":    round(z_max * Z_EXAG, 2),
        "heights":  heights,
    }


def routes_to_threejs(gps_routes, lon_center, lat_center):
    """Convierte rutas GPS a lista de segmentos para Three.js Line."""
    out = []
    for i, route in enumerate(gps_routes):
        pts_3d = []
        for pt in route:
            lon, lat, elev = pt[0], pt[1], pt[2]
            x, y, z = geo_to_meters(lon, lat, elev, lon_center, lat_center)
            # Three.js: (x, elevation, -y) con rotation.x=-PI/2
            pts_3d.append([round(x, 2), round(z, 2), round(-y, 2)])
        out.append({
            "color":  ROUTE_COLORS[i % len(ROUTE_COLORS)],
            "points": pts_3d,
        })
    return out


def contours_to_threejs(contours_dict, lon_center, lat_center, dem, lons, lats):
    """
    Convierte curvas de nivel a Three.js, elevando cada punto según el DEM
    para que queden justo sobre la superficie del terreno.
    """
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0
    lon_min, lon_max = lons[0], lons[-1]
    lat_min, lat_max = lats[0], lats[-1]
    dn_lon = lon_max - lon_min
    dn_lat = lat_max - lat_min
    ny, nx = dem.shape

    def sample_dem(lon, lat):
        fi = (lon - lon_min) / dn_lon * (nx - 1)
        fj = (lat - lat_min) / dn_lat * (ny - 1)
        i = int(np.clip(round(fi), 0, nx - 1))
        j = int(np.clip(round(fj), 0, ny - 1))
        return float(dem[j, i]) * Z_EXAG

    out = {}
    for key, segments in contours_dict.items():
        segs_3d = []
        total_pts = sum(len(s[1]) for s in segments)
        # Submuestrear si hay demasiados puntos
        skip = max(1, total_pts // 8000)
        for seg in segments:
            elev_curve = seg[0] * Z_EXAG
            pts = seg[1]
            pts_3d = []
            for k, (lon, lat) in enumerate(pts):
                if k % skip != 0 and k != len(pts) - 1:
                    continue
                x_m = (lon - lon_center) * mpd_lon
                y_m = (lat - lon_center) * mpd_lat  # aproximación
                y_m = (lat - lat_center) * mpd_lat
                # Elevación: usar la del DEM (que está interpolado) + pequeño offset visual
                z_surf = sample_dem(lon, lat)
                z_use  = max(elev_curve, z_surf) + 0.5  # 0.5 m sobre el terreno
                pts_3d.append([round(x_m, 2), round(z_use, 2), round(-y_m, 2)])
            if len(pts_3d) >= 2:
                segs_3d.append(pts_3d)
        out[key] = {
            "color":    CONTOUR_COLORS[key],
            "segments": segs_3d,
        }
    return out


def hull_to_curtain_threejs(hull_pts, lon_center, lat_center, z_min, z_max):
    """
    Construye la cortina 3D (curtain) del hull del área de estudio.
    Devuelve: {"vertices": [...], "indices": [...], "top_line": [...]}
    """
    if not hull_pts:
        return None

    verts   = []
    indices = []
    top_line = []
    mpd_lon = 111320.0 * math.cos(math.radians(lat_center))
    mpd_lat = 110540.0

    z_bot = (z_min - 10) * Z_EXAG
    z_top = (z_max + 10) * Z_EXAG

    for i, (lon, lat) in enumerate(hull_pts):
        x = round((lon - lon_center) * mpd_lon, 2)
        y = round((lat - lat_center) * mpd_lat, 2)
        # Two verts per hull point: bottom and top
        verts += [x, z_bot, -y,  x, z_top, -y]  # idx: 2i, 2i+1
        top_line.append([x, z_top, -y])

        if i < len(hull_pts) - 1:
            i0 = 2 * i
            i1 = 2 * i + 1
            i2 = 2 * (i + 1)
            i3 = 2 * (i + 1) + 1
            indices += [i0, i2, i1,  i2, i3, i1]

    return {
        "vertices": verts,
        "indices":  indices,
        "top_line": top_line,
    }

# ─────────────────────────────────────────────────────────────
#  7. DESCARGA DE LIBRERÍAS THREE.JS
# ─────────────────────────────────────────────────────────────

THREE_URLS = {
    "three.r128.min.js":    "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
    "OrbitControls.r128.js":"https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
}

def download_vendors():
    """Descarga las librerías Three.js si no existen localmente."""
    VENDOR.mkdir(parents=True, exist_ok=True)

    vendor_content = {}
    for fname, url in THREE_URLS.items():
        local = VENDOR / fname
        if local.exists():
            print(f"  Usando vendor local: {fname}")
            with open(local, encoding="utf-8", errors="replace") as f:
                vendor_content[fname] = f.read()
        elif HAS_REQUESTS:
            print(f"  Descargando {fname} ...", end="", flush=True)
            try:
                import requests as rq
                r = rq.get(url, timeout=30)
                r.raise_for_status()
                with open(local, "w", encoding="utf-8", errors="replace") as f:
                    f.write(r.text)
                vendor_content[fname] = r.text
                print(" OK")
            except Exception as e:
                print(f" FALLBACK CDN ({e})")
                vendor_content[fname] = f"/* Could not download {fname}: {e} */"
        else:
            vendor_content[fname] = f"/* requests not available, cannot download {fname} */"

    return vendor_content

# ─────────────────────────────────────────────────────────────
#  8. GENERACIÓN DEL HTML
# ─────────────────────────────────────────────────────────────

def make_html(dem_data, contours_3d, curtain, rivers_3d, drainage_pts_3d, green_zones_2d, streets_3d, tex_b64, catastro_b64, vendor_js, lon_center, lat_center):
    """
    Genera el HTML autocontenido con Three.js.
    - Curvas de nivel: visibles en 3D (sobre la superficie del terreno)
    - Ríos (Cauce.shp): líneas azul-cian
    - Calles OSM: líneas sobre la superficie para apoyar análisis de escorrentía
    - Catastro SNIT: overlay raster transparente sobre el terreno
    - Puntos de desfogue (OSM KML): esferas naranja con etiqueta
    - Zonas verdes (KML): polígonos semitransparentes sobre el terreno
    - Rutas GPS: NO visibles (usadas solo para DEM)
    """
    three_js      = vendor_js.get("three.r128.min.js", "")
    orbit_js      = vendor_js.get("OrbitControls.r128.js", "")

    has_texture   = bool(tex_b64)
    tex_data_uri  = f"data:image/png;base64,{tex_b64}" if has_texture else ""
    has_catastro  = bool(catastro_b64)
    catastro_uri  = f"data:image/png;base64,{catastro_b64}" if has_catastro else ""

    # Serializar datos para JS
    # Curvas de nivel: se muestran en 3D | Rutas GPS: solo para DEM, no se muestran
    heights_json   = json.dumps(dem_data["heights"])
    contours_json  = json.dumps(contours_3d)
    rivers_json    = json.dumps(rivers_3d)
    drainage_json  = json.dumps(drainage_pts_3d)
    green_json     = json.dumps(green_zones_2d)
    streets_json   = json.dumps(streets_3d)
    curtain_json   = json.dumps(curtain) if curtain else "null"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Modelo 3D Terreno – Área de Estudio Cariari, Costa Rica</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:#0a0a14; overflow:hidden; font-family:'Segoe UI',Arial,sans-serif; }}
  #canvas-container {{ width:100vw; height:100vh; }}
  #info {{
    position:absolute; top:14px; left:14px;
    background:rgba(10,10,24,0.82);
    color:#e8f4ff;
    padding:12px 16px;
    border-radius:10px;
    border:1px solid rgba(100,180,255,0.25);
    font-size:13px;
    line-height:1.7;
    max-width:260px;
    backdrop-filter:blur(6px);
    pointer-events:none;
  }}
  #info h2 {{ font-size:15px; color:#7ecfff; margin-bottom:6px; }}
  #info .stat {{ color:#aed6f1; }}
  #legend {{
    position:absolute; bottom:14px; right:14px;
    background:rgba(10,10,24,0.82);
    color:#e8f4ff;
    padding:10px 14px;
    border-radius:10px;
    border:1px solid rgba(100,180,255,0.25);
    font-size:12px;
    line-height:1.8;
    backdrop-filter:blur(6px);
  }}
  .leg-item {{ display:flex; align-items:center; gap:8px; }}
  .leg-swatch {{ width:24px; height:4px; border-radius:2px; }}
  #controls-hint {{
    position:absolute; bottom:14px; left:14px;
    background:rgba(10,10,24,0.70);
    color:#8ab4cc;
    padding:8px 12px;
    border-radius:8px;
    font-size:11px;
    line-height:1.6;
  }}
</style>
</head>
<body>
<div id="canvas-container"></div>

<div id="info">
  <h2>🏔 Cariari, Costa Rica</h2>
  <div class="stat">Lat: {lat_center:.4f}° | Lon: {lon_center:.4f}°</div>
  <div class="stat">Malla DEM: {dem_data['nx']}×{dem_data['ny']}</div>
  <div class="stat">Elev. mín: {dem_data['z_min']:.0f} m</div>
  <div class="stat">Elev. máx: {dem_data['z_max']:.0f} m</div>
  <div class="stat">Ancho área: {dem_data['width_m']:.0f} m</div>
  <div class="stat">Alto área: {dem_data['height_m']:.0f} m</div>
</div>

<div id="legend">
  <div class="leg-item"><span class="leg-swatch" style="background:#FFD700"></span>Curvas índice</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#FFFFFF"></span>Curvas intermedias</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#A8FF60"></span>Curvas suplementarias</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#ff0000;opacity:0.7;height:10px;border-radius:3px"></span>Área de estudio</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#4fc3f7;height:4px"></span>Ríos / Cauces</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#ff4dd2;height:4px"></span>Calles / red vial OSM</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#ffffff;height:8px;border:1px solid #4a4a4a"></span>Catastro SNIT</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#53b66b;height:10px;border-radius:3px;opacity:0.8"></span>Zonas verdes</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#FF8C00;width:12px;height:12px;border-radius:50%"></span>Puntos de desfogue</div>
  <div class="leg-item" style="color:#6a8ca0;font-size:10px;">🗺 Rutas GPS: solo en DEM</div>
</div>

<div id="controls-hint">
  🖱 Arrastrar: Orbitar &nbsp;|&nbsp; Scroll: Zoom &nbsp;|&nbsp; Clic derecho: Desplazar
</div>

<!-- Three.js r128 embebido -->
<script>
{three_js}
</script>
<script>
{orbit_js}
</script>

<script>
// ═══════════════════════════════════════════════════════════════
//  DATOS DEL TERRENO
// ═══════════════════════════════════════════════════════════════
const DEM = {{
  widthM:  {dem_data['width_m']},
  heightM: {dem_data['height_m']},
  nx:      {dem_data['nx']},
  ny:      {dem_data['ny']},
  zMin:    {dem_data['z_min']},
  zMax:    {dem_data['z_max']},
  heights: {heights_json}
}};

const CONTOURS       = {contours_json};
const RIVERS         = {rivers_json};
const DRAINAGE_PTS   = {drainage_json};
const GREEN_ZONES    = {green_json};
const STREETS        = {streets_json};
const CURTAIN        = {curtain_json};
const HAS_TEX        = {'true' if has_texture else 'false'};
const TEX_URI        = "{tex_data_uri if has_texture else ''}";
const HAS_CATASTRO   = {'true' if has_catastro else 'false'};
const CATASTRO_URI   = "{catastro_uri if has_catastro else ''}";

// ═══════════════════════════════════════════════════════════════
//  SETUP THREE.JS
// ═══════════════════════════════════════════════════════════════
const container = document.getElementById('canvas-container');

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type    = THREE.PCFSoftShadowMap;
container.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x0a0a14);
scene.fog = new THREE.FogExp2(0x0a0a14, 0.00008);

// Cámara
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 1, 80000);
const zCenter = (DEM.zMin + DEM.zMax) / 2;
camera.position.set(0, zCenter + 800, DEM.heightM * 0.9);
camera.lookAt(0, zCenter, 0);

// OrbitControls
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping    = true;
controls.dampingFactor    = 0.08;
controls.screenSpacePanning = false;
controls.minDistance      = 50;
controls.maxDistance      = 15000;
controls.maxPolarAngle    = Math.PI / 2 * (89.1 / 90);
controls.target.set(0, zCenter, 0);
controls.update();

// ═══════════════════════════════════════════════════════════════
//  LUCES
// ═══════════════════════════════════════════════════════════════
scene.add(new THREE.AmbientLight(0xffffff, 0.92));

const dirLight = new THREE.DirectionalLight(0xfff5e0, 0.35);
dirLight.position.set(DEM.widthM * 0.6, DEM.zMax * 1.5, DEM.heightM * 0.4);
dirLight.castShadow = true;
scene.add(dirLight);

scene.add(new THREE.HemisphereLight(0x87ceeb, 0x3a3a2a, 0.20));

// ═══════════════════════════════════════════════════════════════
//  UTILIDADES DE MUESTREO
// ═══════════════════════════════════════════════════════════════
function clamp(value, min, max) {{
  return Math.max(min, Math.min(max, value));
}}

function sampleTerrainHeight(x, z) {{
  const col = clamp((x + DEM.widthM / 2) / DEM.widthM * (DEM.nx - 1), 0, DEM.nx - 1);
  const row = clamp((DEM.heightM / 2 + z) / DEM.heightM * (DEM.ny - 1), 0, DEM.ny - 1);

  const i0 = Math.floor(col);
  const j0 = Math.floor(row);
  const i1 = Math.min(i0 + 1, DEM.nx - 1);
  const j1 = Math.min(j0 + 1, DEM.ny - 1);
  const tx = col - i0;
  const ty = row - j0;

  const idx00 = j0 * DEM.nx + i0;
  const idx10 = j0 * DEM.nx + i1;
  const idx01 = j1 * DEM.nx + i0;
  const idx11 = j1 * DEM.nx + i1;

  const h00 = DEM.heights[idx00];
  const h10 = DEM.heights[idx10];
  const h01 = DEM.heights[idx01];
  const h11 = DEM.heights[idx11];

  const h0 = h00 + (h10 - h00) * tx;
  const h1 = h01 + (h11 - h01) * tx;
  return h0 + (h1 - h0) * ty;
}}

function buildClosedPath(points, PathCtor) {{
  if (!points || points.length < 3) return null;
  const path = new PathCtor();
  path.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {{
    path.lineTo(points[i][0], points[i][1]);
  }}
  path.closePath();
  return path;
}}

// ═══════════════════════════════════════════════════════════════
//  TERRENO
// ═══════════════════════════════════════════════════════════════
(function buildTerrain() {{
  const geo = new THREE.PlaneGeometry(
    DEM.widthM, DEM.heightM,
    DEM.nx - 1, DEM.ny - 1
  );

  // Asignar alturas – el plano tiene vértices en fila/columna
  // PlaneGeometry está en el plano XY (Z=0 inicialmente)
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {{
    pos.setZ(i, DEM.heights[i]);
  }}
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  let mat;
  if (HAS_TEX && TEX_URI) {{
    const loader = new THREE.TextureLoader();
    const tex = loader.load(TEX_URI);
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    mat = new THREE.MeshLambertMaterial({{
      map: tex,
      side: THREE.FrontSide,
    }});
  }} else {{
    // Color por elevación como fallback
    mat = new THREE.MeshLambertMaterial({{
      color: 0x4a7c3f, wireframe: false,
    }});
    // Colorear vértices por elevación
    const colors = [];
    const pos2 = geo.attributes.position;
    for (let i = 0; i < pos2.count; i++) {{
      const h = (pos2.getZ(i) - DEM.zMin) / (DEM.zMax - DEM.zMin);
      const c = new THREE.Color();
      c.setHSL(0.33 - h * 0.33, 0.7, 0.2 + h * 0.5);
      colors.push(c.r, c.g, c.b);
    }}
    geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(colors), 3));
    mat.vertexColors = true;
  }}

  const terrain = new THREE.Mesh(geo, mat);
  terrain.rotation.x = -Math.PI / 2;
  terrain.receiveShadow = true;
  scene.add(terrain);
}})();

// ═══════════════════════════════════════════════════════════════
//  OVERLAY CATASTRAL (SNIT WMS)
// ═══════════════════════════════════════════════════════════════
(function buildCatastroOverlay() {{
  if (!HAS_CATASTRO || !CATASTRO_URI) return;

  const geo = new THREE.PlaneGeometry(
    DEM.widthM, DEM.heightM,
    DEM.nx - 1, DEM.ny - 1
  );
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {{
    pos.setZ(i, DEM.heights[i] + 1.6);
  }}
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  const loader = new THREE.TextureLoader();
  loader.load(CATASTRO_URI, (tex) => {{
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;

    const mat = new THREE.MeshBasicMaterial({{
      map: tex,
      transparent: true,
      opacity: 1.0,
      alphaTest: 0.01,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
    }});

    const overlay = new THREE.Mesh(geo, mat);
    overlay.rotation.x = -Math.PI / 2;
    overlay.renderOrder = 1;
    scene.add(overlay);
  }});
}})();

// ═══════════════════════════════════════════════════════════════
//  GRID HELPER
// ═══════════════════════════════════════════════════════════════
const gridSize = Math.max(DEM.widthM, DEM.heightM) * 1.2;
const grid = new THREE.GridHelper(gridSize, 20, 0x223344, 0x112233);
grid.position.y = DEM.zMin - 12;
scene.add(grid);

// ═══════════════════════════════════════════════════════════════
//  CORTINA DEL ÁREA DE ESTUDIO
// ═══════════════════════════════════════════════════════════════
if (CURTAIN) {{
  const verts = new Float32Array(CURTAIN.vertices);
  const idx   = new Uint32Array(CURTAIN.indices);

  const cGeo = new THREE.BufferGeometry();
  cGeo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  cGeo.setIndex(new THREE.BufferAttribute(idx, 1));
  cGeo.computeVertexNormals();

  const cMat = new THREE.MeshBasicMaterial({{
    color: 0xff0000,
    opacity: 0.35,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  }});
  scene.add(new THREE.Mesh(cGeo, cMat));

  // Línea sólida en el tope
  const topPts = CURTAIN.top_line.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const topGeo = new THREE.BufferGeometry().setFromPoints(topPts);
  const topMat = new THREE.LineBasicMaterial({{ color: 0xff2222, linewidth: 2 }});
  scene.add(new THREE.Line(topGeo, topMat));
}}

// ═══════════════════════════════════════════════════════════════
//  CURVAS DE NIVEL
// ═══════════════════════════════════════════════════════════════
for (const [key, data] of Object.entries(CONTOURS)) {{
  const color = parseInt(data.color.replace('#',''), 16);
  for (const seg of data.segments) {{
    const pts = seg.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({{ color: color, linewidth: 1.5, opacity: 0.90, transparent: true }});
    scene.add(new THREE.Line(geo, mat));
  }}
}}

// ═══════════════════════════════════════════════════════════════
//  RÍOS / CAUCES (Cauce.shp)
// ═══════════════════════════════════════════════════════════════
for (const seg of RIVERS) {{
  const pts = seg.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({{ color: 0x4fc3f7, linewidth: 2, opacity: 0.95, transparent: true }});
  scene.add(new THREE.Line(geo, mat));
}}

// ═══════════════════════════════════════════════════════════════
//  CALLES / RED VIAL (OSM)
// ═══════════════════════════════════════════════════════════════
for (const street of STREETS) {{
  const pts = street.points.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({{
    color: 0xff4dd2,
    linewidth: 2,
    opacity: 0.92,
    transparent: true
  }});
  scene.add(new THREE.Line(geo, mat));
}}

// ═══════════════════════════════════════════════════════════════
//  ZONAS VERDES (KML)
// ═══════════════════════════════════════════════════════════════
if (GREEN_ZONES.length) {{
  const greenFillMat = new THREE.MeshLambertMaterial({{
    color: 0x53b66b,
    opacity: 0.52,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
  }});
  const greenEdgeMat = new THREE.LineBasicMaterial({{
    color: 0x1f6d33,
    opacity: 0.95,
    transparent: true,
  }});

  for (const zone of GREEN_ZONES) {{
    const shape = buildClosedPath(zone.outer, THREE.Shape);
    if (!shape) continue;

    for (const hole of zone.holes || []) {{
      const holePath = buildClosedPath(hole, THREE.Path);
      if (holePath) shape.holes.push(holePath);
    }}

    const geo = new THREE.ShapeGeometry(shape);
    const pos = geo.attributes.position;
    for (let i = 0; i < pos.count; i++) {{
      const x = pos.getX(i);
      const z = pos.getY(i);
      const y = sampleTerrainHeight(x, z) + 0.9;
      pos.setXYZ(i, x, y, z);
    }}
    pos.needsUpdate = true;
    geo.computeVertexNormals();
    scene.add(new THREE.Mesh(geo, greenFillMat));

    for (const ring of [zone.outer, ...(zone.holes || [])]) {{
      if (!ring || ring.length < 2) continue;
      const pts = ring.map(([x, z]) => new THREE.Vector3(x, sampleTerrainHeight(x, z) + 1.15, z));
      pts.push(pts[0].clone());
      const edgeGeo = new THREE.BufferGeometry().setFromPoints(pts);
      scene.add(new THREE.Line(edgeGeo, greenEdgeMat));
    }}
  }}
}}

// ═══════════════════════════════════════════════════════════════
//  PUNTOS DE DESFOGUE (OSM KML)
// ═══════════════════════════════════════════════════════════════
const drainGeo = new THREE.SphereGeometry(8, 8, 6);
const drainMat = new THREE.MeshLambertMaterial({{ color: 0xFF8C00, emissive: 0x552200 }});
for (const pt of DRAINAGE_PTS) {{
  const sphere = new THREE.Mesh(drainGeo, drainMat);
  sphere.position.set(pt.x, pt.y, pt.z);
  scene.add(sphere);

  // Palo vertical para visibilidad
  const pillarGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(pt.x, pt.y, pt.z),
    new THREE.Vector3(pt.x, pt.y + 25, pt.z),
  ]);
  const pillarMat = new THREE.LineBasicMaterial({{ color: 0xFF8C00 }});
  scene.add(new THREE.Line(pillarGeo, pillarMat));
}}

// ═══════════════════════════════════════════════════════════════
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ═══════════════════════════════════════════════════════════════
//  LOOP
// ═══════════════════════════════════════════════════════════════
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""
    return html

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generador 3D Terreno – Cariari, Costa Rica")
    print("=" * 60)

    # Crear directorios
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Encontrar archivo de entrada ──────────────────────
    print("\n[1] Buscando archivo de entrada ...")
    input_path, input_type = find_input_file()

    if input_path is None:
        print("  ERROR: No se encontró ningún archivo KML o SHP.")
        sys.exit(1)
    print(f"  → {input_path} ({input_type.upper()})")

    # ── 2. Leer rutas GPS ────────────────────────────────────
    print("\n[2] Leyendo rutas GPS ...")
    if input_type == "kml":
        gps_routes = read_kml_tracks(input_path)
    else:
        gps_routes = read_shp_routes(input_path)

    if not gps_routes:
        print("  ERROR: No se leyeron rutas GPS.")
        sys.exit(1)

    # Bounding box
    all_lons = [pt[0] for r in gps_routes for pt in r]
    all_lats = [pt[1] for r in gps_routes for pt in r]
    lon_min = min(all_lons) - MARGIN_DEG
    lon_max = max(all_lons) + MARGIN_DEG
    lat_min = min(all_lats) - MARGIN_DEG
    lat_max = max(all_lats) + MARGIN_DEG
    lon_center = (lon_min + lon_max) / 2
    lat_center = (lat_min + lat_max) / 2

    print(f"  BBox: lon [{lon_min:.5f}, {lon_max:.5f}] | lat [{lat_min:.5f}, {lat_max:.5f}]")
    print(f"  Centro: lon={lon_center:.5f}, lat={lat_center:.5f}")

    # ── 3. Curvas de nivel WFS ───────────────────────────────
    print("\n[3] Descargando curvas de nivel IGN WFS ...")
    contours_dict = {}
    if HAS_REQUESTS:
        try:
            contours_dict = get_all_contours(lon_min, lat_min, lon_max, lat_max)
        except Exception as e:
            print(f"  Advertencia: Error en WFS: {e}")
    else:
        print("  Saltando WFS (requests no disponible).")

    # ── 4. Construir DEM ────────────────────────────────────
    print("\n[4] Construyendo DEM con IDW ...")
    dem, lons, lats = build_dem(gps_routes, contours_dict, lon_min, lat_min, lon_max, lat_max)

    # ── 5. Textura satelital ────────────────────────────────
    print("\n[5] Descargando textura satelital ...")
    tex_bytes = None
    if HAS_PIL and HAS_REQUESTS:
        try:
            tex_bytes = build_satellite_texture(lon_min, lat_min, lon_max, lat_max)
        except Exception as e:
            print(f"  Error textura: {e}")
    tex_b64 = base64.b64encode(tex_bytes).decode() if tex_bytes else ""

    print("\n[5b] Descargando overlay catastral ...")
    catastro_bytes = None
    if HAS_REQUESTS:
        try:
            catastro_bytes = download_catastro_overlay(lon_min, lat_min, lon_max, lat_max)
        except Exception as e:
            print(f"  Error catastro: {e}")
    catastro_b64 = base64.b64encode(catastro_bytes).decode() if catastro_bytes else ""

    # ── 6. Preparar datos 3D ────────────────────────────────
    print("\n[6] Preparando datos 3D ...")
    dem_data    = prepare_dem_for_threejs(dem, lons, lats, lon_center, lat_center)
    # Curvas → visibles en 3D | Rutas GPS → solo DEM
    contours_3d = contours_to_threejs(contours_dict, lon_center, lat_center, dem, lons, lats)

    # Convex hull / cortina
    hull_pts = compute_convex_hull(gps_routes)
    all_elevs = [pt[2] for r in gps_routes for pt in r]
    z_min = min(all_elevs);  z_max = max(all_elevs)
    curtain = hull_to_curtain_threejs(hull_pts, lon_center, lat_center, z_min, z_max)

    # Ríos (Cauce.shp)
    print("\n[6b] Leyendo ríos (Cauce.shp) ...")
    raw_rivers = read_cauce_rivers(CAUCE_SHP, lon_min, lat_min, lon_max, lat_max)
    rivers_3d  = rivers_to_threejs(raw_rivers, lon_center, lat_center, dem, lons, lats)

    # Puntos de desfogue (OSM KML)
    print("\n[6c] Leyendo puntos de desfogue (OSM KML) ...")
    raw_drain_pts  = read_osm_kml_points(OSM_KML, lon_min, lat_min, lon_max, lat_max)
    drainage_pts_3d = drainage_points_to_threejs(raw_drain_pts, lon_center, lat_center, dem, lons, lats)

    # Zonas verdes (KML)
    print("\n[6d] Leyendo zonas verdes (KML) ...")
    raw_green_zones = read_kml_polygons(ZONAS_VERDES_KML, lon_min, lat_min, lon_max, lat_max)
    green_zones_2d = green_zones_to_threejs(raw_green_zones, lon_center, lat_center)

    # Calles / red vial OSM
    print("\n[6e] Leyendo calles / red vial (OSM) ...")
    raw_streets = download_osm_streets(lon_min, lat_min, lon_max, lat_max)
    streets_3d = streets_to_threejs(raw_streets, lon_center, lat_center, dem, lons, lats)

    # ── 7. Descargar vendors Three.js ───────────────────────
    print("\n[7] Preparando librerías Three.js ...")
    vendor_js = download_vendors()

    # ── 8. Generar HTML ─────────────────────────────────────
    print("\n[8] Generando HTML ...")
    html_content = make_html(
        dem_data, contours_3d, curtain, rivers_3d, drainage_pts_3d, green_zones_2d, streets_3d,
        tex_b64, catastro_b64, vendor_js, lon_center, lat_center
    )

    html_size = len(html_content.encode("utf-8"))
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n{'='*60}")
    print(f"  ✅ HTML generado: {OUT_HTML}")
    print(f"  Tamaño: {html_size / 1_048_576:.1f} MB")
    print(f"{'='*60}")
    print(f"\n  Abre este archivo en cualquier navegador moderno.")


if __name__ == "__main__":
    main()
