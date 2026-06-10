"""
Exporta la red hidráulica combinada inferida como shapefile para edición en QGIS.

Genera dos archivos en output/shapefiles/:
  - red_hidraulica_links.shp   → líneas de tubería (LineString) con atributos
  - red_hidraulica_nodos.shp   → nodos (tragantes, CRP, CRN, desfogues) como puntos

Uso:
    python exportar_red_hidraulica_shp.py
"""

import sys
import math
from pathlib import Path

# ── Dependencias ────────────────────────────────────────────────────────────
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
except ImportError:
    sys.exit("ERROR: instale geopandas: pip install geopandas")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: instale numpy: pip install numpy")

# ── Importar funciones del módulo principal ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
try:
    from generar_dem_3d_threejs_satelital import (
        read_hydraulic_point_layer,
        read_desfogues_points,
        download_osm_streets,
        build_hydraulic_network,
        lonlat_distance_m,
        TRAGANTES_SHP,
        CRP_SHP,
        CRN_SHP,
        DESFOGUES_SHP,
        DEFAULT_STRUCTURE_DEPTH_M,
        STRUCTURE_MIN_ELEV,
        STRUCTURE_MAX_ELEV,
        NETWORK_MAX_LINK_M,
        NETWORK_MIN_DROP_M,
    )
    print("✓ Módulo principal importado correctamente.")
except ImportError as e:
    sys.exit(f"ERROR importando módulo principal: {e}")

OUT_DIR = Path("output/shapefiles")
MARGIN_DEG = 0.002  # margen extra alrededor de los puntos


def compute_bbox_from_shps(*shp_paths):
    """Calcula bbox unificado de varios shapefiles, reproyectando a WGS84."""
    all_lons, all_lats = [], []
    for shp in shp_paths:
        p = Path(shp)
        if not p.exists():
            continue
        try:
            gdf = gpd.read_file(str(p))
            if gdf.empty:
                continue
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs(epsg=4326)
            b = gdf.total_bounds  # [minx, miny, maxx, maxy]
            if not any(math.isnan(v) for v in b):
                all_lons += [b[0], b[2]]
                all_lats += [b[1], b[3]]
        except Exception:
            pass
    if not all_lons:
        raise RuntimeError("No se pudieron leer los shapefiles para determinar el área.")
    return (
        min(all_lons) - MARGIN_DEG,
        min(all_lats) - MARGIN_DEG,
        max(all_lons) + MARGIN_DEG,
        max(all_lats) + MARGIN_DEG,
    )

# ── Clasificación simplificada (sin DEM) ────────────────────────────────────

def build_direct_network(structures, outlets):
    """
    Red hidráulica con conexiones directas (sin guía de calles).
    Conecta cada nodo hacia el vecino más cercano aguas abajo (mayor caída hidráulica)
    dentro de NETWORK_MAX_LINK_M metros.
    """
    all_nodes = list(structures) + list(outlets)
    lat_ref = np.mean([n["lat"] for n in all_nodes]) if all_nodes else 10.0
    links = []
    seen_pairs = set()

    for src in sorted(all_nodes, key=lambda n: n["hydraulic_elev"], reverse=True):
        if src.get("kind") == "desfogue":
            continue  # los desfogues son sinks, no fuentes

        if src["kind"] == "tragante":
            candidates = [n for n in all_nodes if n["kind"] in ("crp", "crn", "desfogue")]
        else:
            candidates = [n for n in all_nodes if n["kind"] in ("crp", "crn", "desfogue")]

        best = None
        for dst in candidates:
            if dst["id"] == src["id"]:
                continue
            dst_elev = dst.get("hydraulic_elev")
            if dst_elev is None or dst_elev > src["hydraulic_elev"] - NETWORK_MIN_DROP_M:
                continue
            pair_key = (src["id"], dst["id"])
            if pair_key in seen_pairs:
                continue
            dist = lonlat_distance_m(src["lon"], src["lat"], dst["lon"], dst["lat"], lat_ref=lat_ref)
            if dist > NETWORK_MAX_LINK_M:
                continue

            penalty = 0.0
            if src["kind"] == "tragante" and dst.get("kind") == "desfogue":
                penalty += 60.0
            if dst.get("kind") == "desfogue":
                penalty += 10.0
            if dst.get("kind") == "crp":
                penalty -= 8.0
            drop = max(src["hydraulic_elev"] - dst_elev, 0.0)
            score = dist + penalty - min(drop * 5.0, 34.0)
            if best is None or score < best["score"]:
                best = {"dst": dst, "score": score, "dist": dist}

        if best is None:
            continue

        dst = best["dst"]
        seen_pairs.add((src["id"], dst["id"]))
        links.append({
            "id":                   f"link_{len(links) + 1}",
            "source_id":            src["id"],
            "target_id":            dst["id"],
            "source_kind":          src["kind"],
            "target_kind":          dst.get("kind", "desfogue"),
            "drop_m":               round(src["hydraulic_elev"] - dst["hydraulic_elev"], 2),
            "distance_m":           round(best["dist"], 1),
            "source_elev":          src.get("hydraulic_elev"),
            "target_elev":          dst.get("hydraulic_elev"),
            "source_depth_m":       src.get("depth_used", DEFAULT_STRUCTURE_DEPTH_M.get(src["kind"], 1.20)),
            "target_depth_m":       dst.get("depth_used", DEFAULT_STRUCTURE_DEPTH_M.get(dst.get("kind", "desfogue"), 1.20)),
            "source_depth_source":  src.get("depth_source", ""),
            "target_depth_source":  dst.get("depth_source", ""),
            "points":               [(src["lon"], src["lat"]), (dst["lon"], dst["lat"])],
        })

    print(f"  Red directa (sin calles): {len(links)} enlaces")
    return links


def classify_without_dem(structures):
    """
    Asigna hydraulic_elev usando los datos de campo del shapefile.
    Sin DEM, usa: bottom = surface - depth si están disponibles, si no, surface o estimado.
    """
    result = []
    for row in structures:
        raw_s = row.get("surface_raw")
        raw_f = row.get("depth_raw")
        kind  = row["kind"]
        default_depth = DEFAULT_STRUCTURE_DEPTH_M.get(kind, 1.20)

        depth_ok = raw_f is not None and 0.0 <= raw_f <= 8.0
        depth_used = raw_f if depth_ok else default_depth

        # Cota superficial
        surface_elev = None
        if raw_s is not None and STRUCTURE_MIN_ELEV <= raw_s <= STRUCTURE_MAX_ELEV:
            surface_elev = raw_s

        # Cota de fondo (hydraulic_elev para enrutamiento)
        if surface_elev is not None and depth_ok:
            bottom_elev = surface_elev - raw_f
        elif surface_elev is not None:
            bottom_elev = surface_elev - default_depth
        else:
            # Sin datos de cota: no se puede enrutar
            continue

        result.append({
            **row,
            "surface_elev":   surface_elev,
            "bottom_elev":    bottom_elev,
            "depth_used":     depth_used,
            "depth_source":   "campo" if depth_ok else "estimado",
            "hydraulic_elev": bottom_elev,
        })
    return result


def classify_desfogues_without_dem(outlets, floor_elev):
    """
    Prepara desfogues (puntos de salida) sin DEM.
    Si no tienen cota propia, usa floor_elev (mínimo de estructuras) menos 1 m
    para garantizar que sean el punto más bajo de la red.
    """
    result = []
    depth = DEFAULT_STRUCTURE_DEPTH_M.get("desfogue", 0.60)
    for outlet in outlets:
        surface = outlet.get("surface_raw") or outlet.get("terrain_elev")
        if surface is None:
            # Sin cota propia: asignar un nivel que sea el más bajo de la red
            surface = floor_elev - 1.0
        hydraulic_elev = surface - depth
        result.append({
            **outlet,
            "surface_elev":   surface,
            "bottom_elev":    hydraulic_elev,
            "depth_used":     depth,
            "depth_source":   "estimado",
            "hydraulic_elev": hydraulic_elev,
        })
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0. Detectar bbox automáticamente desde los shapefiles
    print("\n[0] Detectando área de estudio desde shapefiles ...")
    LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = compute_bbox_from_shps(
        TRAGANTES_SHP, CRP_SHP, CRN_SHP, DESFOGUES_SHP
    )
    print(f"  BBox: lon [{LON_MIN:.5f}, {LON_MAX:.5f}] | lat [{LAT_MIN:.5f}, {LAT_MAX:.5f}]")

    # 1. Leer estructuras puntuales
    print("\n[1] Leyendo estructuras hidráulicas ...")
    raw = []
    raw.extend(read_hydraulic_point_layer(TRAGANTES_SHP, "tragante", "S",          "F",        LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
    raw.extend(read_hydraulic_point_layer(CRP_SHP,       "crp",     "S (altura)", "F (fondo)", LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
    raw.extend(read_hydraulic_point_layer(CRN_SHP,       "crn",     "S(altura)",  "F(fondo)",  LON_MIN, LAT_MIN, LON_MAX, LAT_MAX))
    print(f"  Estructuras leídas: {len(raw)}  (tragantes + CRP + CRN)")

    # 2. Leer desfogues
    print("\n[2] Leyendo desfogues ...")
    raw_outlets = read_desfogues_points(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    print(f"  Desfogues: {len(raw_outlets)}")

    # 3. Clasificar sin DEM
    print("\n[3] Clasificando elevaciones (sin DEM) ...")
    structures = classify_without_dem(raw)

    # Cota mínima de estructuras para asignar a desfogues sin cota propia
    elevs = [s["hydraulic_elev"] for s in structures if s.get("hydraulic_elev") is not None]
    floor_elev = min(elevs) if elevs else 0.0

    outlets = classify_desfogues_without_dem(raw_outlets, floor_elev)
    print(f"  Estructuras con cota válida: {len(structures)}")
    print(f"  Desfogues con cota válida:   {len(outlets)}")
    if floor_elev != 0.0:
        print(f"  Cota mínima de estructuras: {floor_elev:.2f} msnm (base para desfogues sin cota)")

    # 4. Descargar calles OSM para guiar el trazado
    print("\n[4] Descargando calles OSM ...")
    streets = download_osm_streets(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    print(f"  Calles OSM: {len(streets)} segmentos")

    # 5. Construir red hidráulica
    print("\n[5] Construyendo red hidráulica combinada ...")
    links = build_hydraulic_network(structures, outlets, streets)
    print(f"  Red guiada por calles: {len(links)} tramos")

    if not links:
        print("  Sin calles disponibles → usando conexiones directas como fallback ...")
        links = build_direct_network(structures, outlets)

    print(f"  Red final: {len(links)} tramos")

    if not links:
        print("\nADVERTENCIA: no se generaron tramos. Verifique las cotas en los SHP.")
        return

    # 6. Exportar LINKS como shapefile de líneas
    print("\n[6] Exportando shapefile de tramos ...")
    features_links = []
    for lk in links:
        pts = lk.get("points", [])
        if len(pts) < 2:
            continue
        geom = LineString([(p[0], p[1]) for p in pts])
        features_links.append({
            "geometry":    geom,
            "id":          lk["id"],
            "src_id":      lk["source_id"],
            "dst_id":      lk["target_id"],
            "src_kind":    lk["source_kind"],
            "dst_kind":    lk["target_kind"],
            "drop_m":      lk["drop_m"],
            "dist_m":      lk["distance_m"],
            "src_elev":    round(lk["source_elev"], 3) if lk["source_elev"] is not None else None,
            "dst_elev":    round(lk["target_elev"], 3) if lk["target_elev"] is not None else None,
            "src_prof":    round(lk["source_depth_m"], 2),
            "dst_prof":    round(lk["target_depth_m"], 2),
            "src_pfnte":   lk["source_depth_source"],
            "dst_pfnte":   lk["target_depth_source"],
        })

    gdf_links = gpd.GeoDataFrame(features_links, crs="EPSG:4326").to_crs(epsg=5367)
    out_links = OUT_DIR / "red_hidraulica_links.shp"
    gdf_links.to_file(str(out_links), encoding="utf-8")
    print(f"  → {out_links}  ({len(gdf_links)} tramos)  [CRTM05 / EPSG:5367]")

    # 7. Exportar NODOS como shapefile de puntos
    print("\n[7] Exportando shapefile de nodos ...")
    features_nodos = []
    for s in structures:
        features_nodos.append({
            "geometry":   Point(s["lon"], s["lat"]),
            "id":         s["id"],
            "kind":       s["kind"],
            "sup_elev":   round(s["surface_elev"], 3) if s.get("surface_elev") is not None else None,
            "fon_elev":   round(s["bottom_elev"],  3) if s.get("bottom_elev")  is not None else None,
            "prof_m":     round(s["depth_used"],   2),
            "prof_src":   s["depth_source"],
        })
    for o in outlets:
        features_nodos.append({
            "geometry":   Point(o["lon"], o["lat"]),
            "id":         o["id"],
            "kind":       "desfogue",
            "sup_elev":   round(o["surface_elev"], 3) if o.get("surface_elev") is not None else None,
            "fon_elev":   round(o["bottom_elev"],  3) if o.get("bottom_elev")  is not None else None,
            "prof_m":     round(o["depth_used"],   2),
            "prof_src":   o["depth_source"],
        })

    gdf_nodos = gpd.GeoDataFrame(features_nodos, crs="EPSG:4326").to_crs(epsg=5367)
    out_nodos = OUT_DIR / "red_hidraulica_nodos.shp"
    gdf_nodos.to_file(str(out_nodos), encoding="utf-8")
    print(f"  → {out_nodos}  ({len(gdf_nodos)} nodos)  [CRTM05 / EPSG:5367]")

    print("\n✓ Exportación completa.")
    print(f"  Carpeta de salida: {OUT_DIR.resolve()}")
    print("\n  Para abrir en QGIS:")
    print(f"    Capa > Añadir capa > Añadir capa vectorial")
    print(f"    → {out_links.resolve()}")
    print(f"    → {out_nodos.resolve()}")
    print("\n  Campos en red_hidraulica_links.shp:")
    print("    dist_m    → longitud del tramo (metros)")
    print("    drop_m    → caída hidráulica (metros)")
    print("    src_elev  → cota fondo aguas arriba (msnm)")
    print("    dst_elev  → cota fondo aguas abajo  (msnm)")
    print("    src_prof  → profundidad en nodo origen (m)")
    print("    dst_prof  → profundidad en nodo destino (m)")
    print("    src_pfnte → fuente de prof. origen (campo/estimado)")
    print("    dst_pfnte → fuente de prof. destino (campo/estimado)")


if __name__ == "__main__":
    main()
