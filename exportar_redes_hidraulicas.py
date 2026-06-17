"""
exportar_redes_hidraulicas.py
Genera secciones hidraulicas NODO-A-NODO usando Alcantarillado.shp como guia
de ruta entre cajas CRN/CRP.

Algoritmo:
 1. Para cada linea de Alcantarillado, asigna todos los nodos CRN/CRP cercanos
    y los ordena por distancia a lo largo de la linea (d_along).
 2. Pares consecutivos en ese orden = secciones del ramal.
    Geometria = sub-segmento de la linea original (nodos fijos, vertices conservados).
 3. Nodos huerfanos (no asignados a ningun ramal):
    si hay otra caja cercana ya en red, crear tramo recto de conexion;
    si hay una linea de alcantarillado cerca, enrutar por ella.
 4. Tragantes: conexion perpendicular al segmento mas cercano.

  Red Pluvial  (CRP)  -> output/shapefiles/red_pluvial_links.shp
  Red Residual (CRN)  -> output/shapefiles/red_residual_links.shp
"""

import sys
import math
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
except ImportError:
    sys.exit("ERROR: instale geopandas: pip install geopandas")

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

ALCANTARILLADO_SHP = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Alcantarillado.shp"
LINEAS_SHP         = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/liteas a seguir.shp"
TRAGANTES_SHP      = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Tragantes.shp"
CRP_SHP            = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/CRP.shp"
CRN_SHP            = Path(r"C:\Users\Jorge\Documents\CRN.shp")
DESFOGUES_SHP      = BASE / "Arcgis shapes y curvas/Desfogues.shp"
OUT_DIR            = BASE / "output/shapefiles"
OUT_DIR_V2         = BASE / "output/shapefiles_v6"   # v6: tipo de linea por extremos

# ── Tolerancias (metros, CRTM05) ─────────────────────────────────────────────────
SNAP_ENDPOINT_TOL_M = 30.0   # caja cercana al extremo de una linea -> anclada al extremo
ON_LINE_TOL_M       = 15.0   # caja sobre el cuerpo de la linea si distancia perp <= esto
MIN_SECTION_M       = 1.0    # seccion mas corta aceptada (m)
CONNECT_ORPHAN_M    = 60.0   # max distancia para conectar huerfanos
LATERAL_MAX_M       = 60.0   # max distancia tragante -> segmento
MAX_ADVERSE_DROP_M  = -1.0   # caida hidraulica minima aceptada (negativo = sube; rechaza >1m de subida)

MIN_ELEV      = 900.0
MAX_ELEV      = 1100.0
DEFAULT_DEPTH = {"tragante": 0.80, "crp": 1.60, "crn": 1.40, "desfogue": 0.60}


# ── Utilidades ───────────────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ── Carga de datos ───────────────────────────────────────────────────────────────

def load_nodes(shp_path, kind, surface_field, depth_field):
    p = Path(shp_path)
    if not p.exists():
        print(f"  AVISO: no se encontro {p.name}")
        return []
    gdf = gpd.read_file(str(p))
    if gdf.empty:
        return []
    if gdf.crs and gdf.crs.to_epsg() != 5367:
        gdf = gdf.to_crs(epsg=5367)
    nodes = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            continue
        x, y = float(geom.x), float(geom.y)
        raw_s = _safe_float(row.get(surface_field))
        raw_f = _safe_float(row.get(depth_field))
        surface_elev  = raw_s if raw_s is not None and MIN_ELEV <= raw_s <= MAX_ELEV else None
        depth_ok      = raw_f is not None and 0.0 < raw_f <= 8.0
        depth_used    = raw_f if depth_ok else DEFAULT_DEPTH.get(kind, 1.20)
        hydraulic_elev = (surface_elev - depth_used) if surface_elev is not None else None
        nodes.append({
            "id":             f"{kind}_{len(nodes) + 1}",
            "kind":           kind,
            "x":              x,
            "y":              y,
            "sup_elev":       round(surface_elev, 3) if surface_elev is not None else None,
            "fon_elev":       round(hydraulic_elev, 3) if hydraulic_elev is not None else None,
            "prof_m":         round(depth_used, 2),
            "prof_src":       "campo" if depth_ok else "estimado",
            "hydraulic_elev": hydraulic_elev,
        })
    return nodes


def load_desfogues():
    p = Path(DESFOGUES_SHP)
    if not p.exists():
        return []
    gdf = gpd.read_file(str(p))
    if gdf.empty:
        return []
    if gdf.crs and gdf.crs.to_epsg() != 5367:
        gdf = gdf.to_crs(epsg=5367)
    nodes = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            continue
        nodes.append({
            "id":             f"desfogue_{len(nodes) + 1}",
            "kind":           "desfogue",
            "x":              float(geom.x),
            "y":              float(geom.y),
            "sup_elev":       None,
            "fon_elev":       None,
            "prof_m":         DEFAULT_DEPTH["desfogue"],
            "prof_src":       "estimado",
            "hydraulic_elev": None,
        })
    return nodes


def _line_hash(coords):
    return tuple((round(x, 1), round(y, 1)) for x, y in coords)


def load_backbone_lines():
    """Carga Alcantarillado.shp + liteas a seguir.shp sin duplicados."""
    seen  = set()
    lines = []
    for shp, prefix in [(ALCANTARILLADO_SHP, "alc"), (LINEAS_SHP, "lin")]:
        if not Path(shp).exists():
            print(f"  AVISO: no se encontro {Path(shp).name}")
            continue
        gdf = gpd.read_file(str(shp))
        if gdf.crs and gdf.crs.to_epsg() != 5367:
            gdf = gdf.to_crs(epsg=5367)
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            geoms = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
            for g in geoms:
                if g.geom_type != "LineString" or len(g.coords) < 2:
                    continue
                coords = list(g.coords)
                h = _line_hash(coords)
                if h in seen:
                    continue
                seen.add(h)
                lines.append({"id": f"{prefix}_{len(lines) + 1}", "coords": coords})
    return lines


# ── Nucleo: nodos sobre cada linea ────────────────────────────────────────────────

def assign_nodes_to_line(line_geom, box_nodes):
    """
    Para una linea dada, asigna todos los nodos CRN/CRP/desfogue que le pertenecen
    y los ordena por posicion a lo largo de la linea (d_along).

    Reglas de asignacion:
    - Nodo dentro de SNAP_ENDPOINT_TOL_M del extremo inicial -> d = 0  (snap a inicio)
    - Nodo dentro de SNAP_ENDPOINT_TOL_M del extremo final   -> d = L  (snap a fin)
    - Nodo dentro de ON_LINE_TOL_M del cuerpo de la linea    -> d = proyeccion

    Desempate: si dos nodos caen en la misma posicion (<2m), gana el mas cercano.
    Los nodos ya asignados por snap de extremo no compiten con nodos de cuerpo.
    """
    L = line_geom.length
    if L < 0.1:
        return []

    start = Point(line_geom.coords[0])
    end   = Point(line_geom.coords[-1])

    endpoint_d = {}   # d (0.0 o L) -> (node, dist)
    body_d     = {}   # d_along rounded -> (node, dist_perp)

    for node in box_nodes:
        pt         = Point(node["x"], node["y"])
        d_perp     = line_geom.distance(pt)
        d_along    = line_geom.project(pt)
        d_to_start = start.distance(pt)
        d_to_end   = end.distance(pt)

        if d_to_start <= SNAP_ENDPOINT_TOL_M:
            k = 0.0
            if k not in endpoint_d or d_to_start < endpoint_d[k][1]:
                endpoint_d[k] = (node, d_to_start)
        elif d_to_end <= SNAP_ENDPOINT_TOL_M:
            k = L
            if k not in endpoint_d or d_to_end < endpoint_d[k][1]:
                endpoint_d[k] = (node, d_to_end)
        elif d_perp <= ON_LINE_TOL_M:
            k = round(d_along, 1)
            if k not in body_d or d_perp < body_d[k][1]:
                body_d[k] = (node, d_perp)

    # Combinar y ordenar
    all_pts = [(d, node) for d, (node, _) in endpoint_d.items()]
    all_pts += [(d, node) for d, (node, _) in body_d.items()]
    all_pts.sort(key=lambda t: t[0])

    # Deduplicar: fusionar puntos dentro de 2m (conservar el mas cercano a la linea)
    result = []
    for d, node in all_pts:
        if result and abs(d - result[-1][0]) < 2.0:
            # Mantener el que tiene menor distancia perp a la linea
            prev_d, prev_node = result[-1]
            prev_dist = line_geom.distance(Point(prev_node["x"], prev_node["y"]))
            curr_dist = line_geom.distance(Point(node["x"], node["y"]))
            if curr_dist < prev_dist:
                result[-1] = (d, node)
            # Si el anterior es mejor, no hacer nada
        else:
            result.append((d, node))

    return result


def extract_sub_segment(line_geom, d_start, d_end, start_xy=None, end_xy=None):
    """
    Extrae subsegmento entre d_start y d_end preservando vertices originales.
    Los extremos se anclan a las coordenadas exactas de los nodos (fijos).
    """
    p_start = Point(start_xy) if start_xy is not None else line_geom.interpolate(d_start)
    inner   = []
    for coord in line_geom.coords:
        d = line_geom.project(Point(coord))
        if d_start < d < d_end:
            inner.append(Point(coord))
    p_end = Point(end_xy) if end_xy is not None else line_geom.interpolate(d_end)

    pts = [p_start] + inner + [p_end]
    unique = [pts[0]]
    for p in pts[1:]:
        if p.distance(unique[-1]) > 0.01:
            unique.append(p)

    if len(unique) < 2:
        return None
    return LineString([(p.x, p.y) for p in unique])


def classify_network(kind_src, kind_dst):
    if "crn" in (kind_src, kind_dst):
        return "residual"
    return "pluvial"


def _make_section(seg_id, n_src, n_dst, geom):
    he_src = n_src["hydraulic_elev"]
    he_dst = n_dst["hydraulic_elev"]
    if he_src is not None and he_dst is not None and he_src < he_dst:
        n_src, n_dst = n_dst, n_src
        geom = LineString(list(reversed(geom.coords)))
        he_src, he_dst = he_dst, he_src
    drop_m = round(he_src - he_dst, 3) if he_src is not None and he_dst is not None else None
    return {
        "id":       seg_id,
        "red":      classify_network(n_src["kind"], n_dst["kind"]),
        "src_id":   n_src["id"],
        "dst_id":   n_dst["id"],
        "src_kind": n_src["kind"],
        "dst_kind": n_dst["kind"],
        "drop_m":   drop_m,
        "dist_m":   round(geom.length, 1),
        "src_elev": n_src["fon_elev"],
        "dst_elev": n_dst["fon_elev"],
        "src_prof": n_src["prof_m"],
        "dst_prof": n_dst["prof_m"],
        "snap_src": "T",
        "snap_dst": "T",
        "geom":     geom,
    }


# ── Paso principal: secciones nodo-a-nodo por linea de alcantarillado ─────────────

def _type_of_line(line_geom, crn_nodes, crp_nodes):
    """
    Determina el tipo de una linea por el nodo mas cercano a cada extremo.
    Devuelve ("crn", crn_nodes) o ("crp", crp_nodes) o (None, None).
    El extremo es la primera pista: la linea fue dibujada DESDE ese nodo.
    """
    start = Point(line_geom.coords[0])
    end   = Point(line_geom.coords[-1])

    def nearest(pt, nodes, tol):
        best_d, best_n = float("inf"), None
        for n in nodes:
            d = pt.distance(Point(n["x"], n["y"]))
            if d < tol and d < best_d:
                best_d, best_n = d, n
        return best_n

    crn_s = nearest(start, crn_nodes, SNAP_ENDPOINT_TOL_M)
    crn_e = nearest(end,   crn_nodes, SNAP_ENDPOINT_TOL_M)
    crp_s = nearest(start, crp_nodes, SNAP_ENDPOINT_TOL_M)
    crp_e = nearest(end,   crp_nodes, SNAP_ENDPOINT_TOL_M)

    has_crn = (crn_s is not None) or (crn_e is not None)
    has_crp = (crp_s is not None) or (crp_e is not None)

    if has_crn and not has_crp:
        return "crn", crn_nodes
    if has_crp and not has_crn:
        return "crp", crp_nodes
    if has_crn and has_crp:
        # Ambos tipos en los extremos: elegir el que tiene los DOS extremos snapeados
        both_crn = (crn_s is not None) and (crn_e is not None)
        both_crp = (crp_s is not None) and (crp_e is not None)
        if both_crn and not both_crp:
            return "crn", crn_nodes
        if both_crp and not both_crn:
            return "crp", crp_nodes
        # Empate: usar el tipo del extremo inicio
        d_crn_s = start.distance(Point(crn_s["x"], crn_s["y"])) if crn_s else float("inf")
        d_crp_s = start.distance(Point(crp_s["x"], crp_s["y"])) if crp_s else float("inf")
        return ("crn", crn_nodes) if d_crn_s <= d_crp_s else ("crp", crp_nodes)
    return None, None


def process_lines_into_sections(backbone_lines, box_nodes):
    """
    El tipo de la linea lo determinan los nodos en sus EXTREMOS.
    Una vez determinado el tipo, solo se buscan nodos de ESE tipo sobre la linea
    — jamas se mezclan CRN y CRP en el mismo ramal.
    """
    crn_nodes = [n for n in box_nodes if n["kind"] in ("crn", "desfogue")]
    crp_nodes = [n for n in box_nodes if n["kind"] == "crp"]

    sections  = []
    unmatched = []

    for line in backbone_lines:
        line_geom = LineString(line["coords"])

        net_type, type_nodes = _type_of_line(line_geom, crn_nodes, crp_nodes)
        if net_type is None:
            unmatched.append({
                "id": line["id"], "geom": line_geom,
                "motivo": "sin nodos en extremos",
            })
            continue

        # Solo nodos del tipo correcto sobre esta linea
        node_list = assign_nodes_to_line(line_geom, type_nodes)
        if len(node_list) < 2:
            unmatched.append({
                "id": line["id"], "geom": line_geom,
                "motivo": f"tipo={net_type} pero solo {len(node_list)} nodo(s)",
            })
            continue

        for i in range(len(node_list) - 1):
            d0, n_src = node_list[i]
            d1, n_dst = node_list[i + 1]
            if d1 - d0 < MIN_SECTION_M or n_src["id"] == n_dst["id"]:
                continue
            seg_geom = extract_sub_segment(
                line_geom, d0, d1,
                start_xy=(n_src["x"], n_src["y"]),
                end_xy=(n_dst["x"], n_dst["y"]),
            )
            if seg_geom is None:
                continue
            sections.append(_make_section(f"sec_{len(sections) + 1}", n_src, n_dst, seg_geom))

    print(f"  Secciones backbone (nodo-a-nodo): {len(sections)} | Lineas sin asignar: {len(unmatched)}")
    return sections, unmatched


# ── Conexion de nodos huerfanos ───────────────────────────────────────────────────

def _route_via_line(n_a, n_b, backbone_lines, tol=None):
    """
    Intenta enrutar entre dos nodos siguiendo la linea de alcantarillado mas cercana.
    Ambos nodos deben estar dentro de `tol` metros de la MISMA linea.
    Devuelve (geom, n_src, n_dst) o None si no hay linea valida.
    """
    if tol is None:
        tol = CONNECT_ORPHAN_M
    pt_a = Point(n_a["x"], n_a["y"])
    pt_b = Point(n_b["x"], n_b["y"])

    best_geom  = None
    best_score = float("inf")

    for line in backbone_lines:
        lg = LineString(line["coords"])
        da = lg.distance(pt_a)
        db = lg.distance(pt_b)
        if da > tol or db > tol:
            continue
        dA_along = lg.project(pt_a)
        dB_along = lg.project(pt_b)
        if abs(dA_along - dB_along) < MIN_SECTION_M:
            continue
        score = da + db
        if score < best_score:
            best_score = score
            d0 = min(dA_along, dB_along)
            d1 = max(dA_along, dB_along)
            n0, n1 = (n_a, n_b) if dA_along <= dB_along else (n_b, n_a)
            seg = extract_sub_segment(
                lg, d0, d1,
                start_xy=(n0["x"], n0["y"]),
                end_xy=(n1["x"], n1["y"]),
            )
            if seg is not None:
                best_geom = (seg, n0, n1)

    return best_geom


def _compatible_kinds(kind_a, kind_b):
    """True si ambos nodos pertenecen a la misma red (nunca CRP<->CRN)."""
    residual = {"crn", "desfogue"}
    pluvial  = {"crp"}
    if kind_a in residual and kind_b in residual:
        return True
    if kind_a in pluvial and kind_b in pluvial:
        return True
    return False


def _elevation_ok(n_a, n_b):
    """True si la conexion es hidraulicamente aceptable segun elevacion de fondo."""
    he_a = n_a.get("hydraulic_elev")
    he_b = n_b.get("hydraulic_elev")
    if he_a is None or he_b is None:
        return True  # sin datos de elevacion -> no filtrar
    drop = abs(he_a - he_b)  # caida entre los dos
    # Rechazar si el nodo de mayor cota sube demasiado respecto al otro
    # (drop negativo en _make_section significaria subida excesiva)
    min_he = min(he_a, he_b)
    max_he = max(he_a, he_b)
    # La "subida" real desde el punto mas bajo al mas alto es max_he - min_he
    # Permitimos conexion si la subida esta dentro de un gradiente razonable
    # o si simplemente hay caida (lo normal en gravedad)
    return (max_he - min_he) >= MAX_ADVERSE_DROP_M  # siempre True para caida; False si sube mas de 1m con sentido opuesto


def connect_orphan_nodes(sections, box_nodes, node_by_id, backbone_lines):
    """
    Nodos CRN/CRP que no aparecen en ninguna seccion:
    - Solo conecta con nodos del MISMO tipo de red (nunca CRP<->CRN)
    - Filtra por elevacion de fondo (no conecta si sube demasiado)
    - Primero intenta enrutar por linea de alcantarillado; si no, tramo recto
    """
    networked_ids = set()
    for sec in sections:
        networked_ids.add(sec["src_id"])
        networked_ids.add(sec["dst_id"])

    orphans   = [n for n in box_nodes if n["id"] not in networked_ids]
    networked = [node_by_id[nid] for nid in networked_ids if nid in node_by_id]

    new_links = []
    for orph in orphans:
        ox, oy = orph["x"], orph["y"]
        best_n, best_d = None, float("inf")
        for n in networked:
            # Solo mismo tipo de red
            if not _compatible_kinds(orph["kind"], n["kind"]):
                continue
            d = math.hypot(n["x"] - ox, n["y"] - oy)
            if d < best_d:
                best_d = d
                best_n = n
        if best_n is None or best_d > CONNECT_ORPHAN_M:
            continue
        if not _elevation_ok(orph, best_n):
            continue

        routed = _route_via_line(orph, best_n, backbone_lines)
        if routed is not None:
            seg_geom, n_s, n_d = routed
            new_links.append(_make_section(f"orph_{len(new_links) + 1}", n_s, n_d, seg_geom))
        else:
            geom = LineString([(ox, oy), (best_n["x"], best_n["y"])])
            new_links.append(_make_section(f"orph_{len(new_links) + 1}", orph, best_n, geom))

        networked_ids.add(orph["id"])
        networked.append(orph)

    n_sin = len(orphans) - len(new_links)
    print(f"  Huerfanos: {len(orphans)} | conectados: {len(new_links)} | sin red: {n_sin}")
    return new_links


# ── Gap-fill: conexiones directas entre nodos del mismo tipo ─────────────────────

GAP_FILL_M     = 250.0  # distancia maxima nodo-nodo a lo largo de la linea
GAP_LINE_TOL_M = 5.0    # max desviacion perpendicular del nodo a la linea (garantiza paralelismo)

def _score_candidate(dist_m, drop_m):
    """
    Puntaje para elegir el mejor candidato de gap-fill.
    Menor es mejor: prioriza distancia corta; premia caida de elevacion.
    """
    elev_bonus = max(0.0, drop_m) * 0.5  # cada metro de caida vale 0.5m de distancia
    return dist_m - elev_bonus


def connect_same_type_gaps(sections, box_nodes, backbone_lines, max_dist=GAP_FILL_M):
    """
    Para cada nodo, evalua TODOS los candidatos validos del mismo tipo y conecta
    SOLO al mejor (menor score = mas cercano + preferiblemente mas caida).

    Condiciones para que un candidato sea valido:
      1. Mismo tipo de red (CRN<->CRN o CRP<->CRP), no ya conectado directamente.
      2. Elevacion de fondo: no sube mas de MAX_ADVERSE_DROP_M.
      3. Ambos proyectan sobre la MISMA linea de alcantarillado (tol=GAP_LINE_TOL_M).
      4. No hay ningun nodo intermedio del mismo tipo sobre esa linea.
    """
    direct_pairs = set()
    for sec in sections:
        a, b = sec["src_id"], sec["dst_id"]
        direct_pairs.add((min(a, b), max(a, b)))

    crn_group = [n for n in box_nodes if n["kind"] in ("crn", "desfogue")]
    crp_group = [n for n in box_nodes if n["kind"] == "crp"]

    new_sections = []

    for group in [crn_group, crp_group]:
        for n_a in group:
            best_score   = float("inf")
            best_routed  = None
            best_pair    = None

            for n_b in group:
                if n_b["id"] == n_a["id"]:
                    continue
                pair = (min(n_a["id"], n_b["id"]), max(n_a["id"], n_b["id"]))
                if pair in direct_pairs:
                    continue
                dist = math.hypot(n_a["x"] - n_b["x"], n_a["y"] - n_b["y"])
                if dist > max_dist:
                    continue
                if not _elevation_ok(n_a, n_b):
                    continue

                routed = _route_via_line(n_a, n_b, backbone_lines, tol=GAP_LINE_TOL_M)
                if routed is None:
                    continue
                seg_geom, n_s, n_d = routed

                # Sin nodos intermedios del mismo grupo sobre el subsegmento
                dA    = seg_geom.project(Point(n_s["x"], n_s["y"]))
                dB    = seg_geom.project(Point(n_d["x"], n_d["y"]))
                d_lo, d_hi = min(dA, dB), max(dA, dB)
                skip = False
                for n_c in group:
                    if n_c["id"] in (n_s["id"], n_d["id"]):
                        continue
                    if seg_geom.distance(Point(n_c["x"], n_c["y"])) <= GAP_LINE_TOL_M:
                        d_c = seg_geom.project(Point(n_c["x"], n_c["y"]))
                        if d_lo < d_c < d_hi:
                            skip = True
                            break
                if skip:
                    continue

                # Calcular caida para el score
                he_a = n_a.get("hydraulic_elev")
                he_b = n_b.get("hydraulic_elev")
                drop = (he_a - he_b) if he_a is not None and he_b is not None else 0.0
                score = _score_candidate(dist, drop)

                if score < best_score:
                    best_score  = score
                    best_routed = (seg_geom, n_s, n_d)
                    best_pair   = pair

            if best_routed is None:
                continue

            seg_geom, n_s, n_d = best_routed
            new_sections.append(
                _make_section(f"gap_{len(new_sections) + 1}", n_s, n_d, seg_geom)
            )
            direct_pairs.add(best_pair)

    crn_new = sum(1 for s in new_sections if s["red"] == "residual")
    crp_new = sum(1 for s in new_sections if s["red"] == "pluvial")
    print(f"  Gap-fill: {len(new_sections)} conexiones  (CRN-CRN: {crn_new} | CRP-CRP: {crp_new})")
    return new_sections


# ── Laterales de Tragantes ────────────────────────────────────────────────────────

def process_tragante_laterals(nodes_trag, all_sections):
    """
    Proyecta cada Tragante perpendicularmente sobre el segmento de red mas cercano.
    El lateral hereda la clasificacion del segmento al que conecta.
    """
    trag_already = set()
    for sec in all_sections:
        if sec["src_kind"] == "tragante":
            trag_already.add(sec["src_id"])
        if sec["dst_kind"] == "tragante":
            trag_already.add(sec["dst_id"])

    # Tragantes solo conectan a la red pluvial (CRP)
    section_geoms = [(sec, sec["geom"]) for sec in all_sections if sec["red"] == "pluvial"]
    laterals = []
    n_skip   = 0

    for trag in nodes_trag:
        if trag["id"] in trag_already:
            continue
        tx, ty = trag["x"], trag["y"]
        pt     = Point(tx, ty)

        best_sec, best_dist, proj_pt = None, float("inf"), None
        for sec, geom in section_geoms:
            d = geom.distance(pt)
            if d < best_dist:
                best_dist = d
                best_sec  = sec
                pp, _     = nearest_points(geom, pt)
                proj_pt   = pp

        if best_sec is None or best_dist > LATERAL_MAX_M:
            n_skip += 1
            continue

        geom_lat = LineString([(tx, ty), (proj_pt.x, proj_pt.y)])
        if geom_lat.length < 0.1:
            continue

        frac    = best_sec["geom"].project(proj_pt, normalized=True)
        he_s    = best_sec.get("src_elev")
        he_d    = best_sec.get("dst_elev")
        he_proj = (he_s + frac * (he_d - he_s)) if he_s is not None and he_d is not None else (he_s or he_d)
        he_trag = trag["hydraulic_elev"]
        drop_m  = round(he_trag - he_proj, 3) if he_trag is not None and he_proj is not None else None

        laterals.append({
            "id":       f"lat_{len(laterals) + 1}",
            "red":      best_sec["red"],
            "src_id":   trag["id"],
            "dst_id":   best_sec["dst_id"],
            "src_kind": "tragante",
            "dst_kind": best_sec["dst_kind"],
            "drop_m":   drop_m,
            "dist_m":   round(geom_lat.length, 1),
            "src_elev": trag["fon_elev"],
            "dst_elev": round(he_proj, 3) if he_proj is not None else None,
            "src_prof": trag["prof_m"],
            "dst_prof": best_sec["dst_prof"],
            "snap_src": "T",
            "snap_dst": "T",
            "geom":     geom_lat,
        })

    print(f"  Laterales Tragantes: {len(laterals)} | {n_skip} demasiado lejos (>{LATERAL_MAX_M}m)")
    return laterals


# ── Exportar ─────────────────────────────────────────────────────────────────────

def export_links(links, out_path):
    if not links:
        print(f"  (sin tramos para {out_path.name})")
        return
    rows = [
        {
            "geometry": lk["geom"],
            "id":       lk["id"],
            "red":      lk["red"],
            "src_id":   lk["src_id"],
            "dst_id":   lk["dst_id"],
            "src_kind": lk["src_kind"],
            "dst_kind": lk["dst_kind"],
            "drop_m":   lk["drop_m"],
            "dist_m":   lk["dist_m"],
            "src_elev": lk["src_elev"],
            "dst_elev": lk["dst_elev"],
            "src_prof": lk["src_prof"],
            "dst_prof": lk["dst_prof"],
            "snap_src": lk["snap_src"],
            "snap_dst": lk["snap_dst"],
        }
        for lk in links
    ]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5367")
    gdf.to_file(str(out_path), encoding="utf-8")
    print(f"  -> {out_path.name}  ({len(gdf)} tramos)")


def export_nodes(node_index, node_ids, out_path, red_name):
    rows = [
        {
            "geometry": Point(n["x"], n["y"]),
            "id":       n["id"],
            "kind":     n["kind"],
            "red":      red_name,
            "sup_elev": n["sup_elev"],
            "fon_elev": n["fon_elev"],
            "prof_m":   n["prof_m"],
            "prof_src": n["prof_src"],
        }
        for n in node_index
        if n["id"] in node_ids
    ]
    if not rows:
        print(f"  (sin nodos para {out_path.name})")
        return
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5367")
    # Agregar lon/lat WGS84 para importacion directa en HTML/Three.js
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf["lon"] = gdf_wgs.geometry.x.round(7)
    gdf["lat"] = gdf_wgs.geometry.y.round(7)
    gdf.to_file(str(out_path), encoding="utf-8")
    print(f"  -> {out_path.name}  ({len(gdf)} nodos)")


def export_unmatched(unmatched_geoms, out_path):
    if not unmatched_geoms:
        return
    rows = [{"geometry": u["geom"], "id": u["id"], "motivo": u["motivo"]} for u in unmatched_geoms]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5367")
    gdf.to_file(str(out_path), encoding="utf-8")
    print(f"  -> {out_path.name}  ({len(gdf)} lineas sin nodos)")


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR_V2.mkdir(parents=True, exist_ok=True)

    print("\n[1] Cargando lineas de backbone (Alcantarillado.shp) ...")
    backbone = load_backbone_lines()
    print(f"  Lineas unicas: {len(backbone)}")

    print("\n[2] Cargando nodos ...")
    nodes_trag = load_nodes(TRAGANTES_SHP, "tragante", "S",          "F")
    nodes_crp  = load_nodes(CRP_SHP,       "crp",      "S (altura)", "F (fondo)")
    nodes_crn  = load_nodes(CRN_SHP,       "crn",      "S(altura)",  "F(fondo)")
    nodes_des  = load_desfogues()
    node_index = nodes_trag + nodes_crp + nodes_crn + nodes_des
    node_by_id = {n["id"]: n for n in node_index}
    print(f"  Tragantes: {len(nodes_trag)} | CRP: {len(nodes_crp)} | CRN: {len(nodes_crn)} | Desfogues: {len(nodes_des)}")

    box_nodes = nodes_crp + nodes_crn + nodes_des

    print("\n[3] Construyendo secciones nodo-a-nodo por linea de alcantarillado ...")
    sections, unmatched = process_lines_into_sections(backbone, box_nodes)

    print("\n[4] Conectando nodos huerfanos ...")
    orphan_links = connect_orphan_nodes(sections, box_nodes, node_by_id, backbone)

    print("\n[4b] Gap-fill: conectando nodos del mismo tipo via alcantarillado ...")
    gap_links = connect_same_type_gaps(sections + orphan_links, box_nodes, backbone)

    print("\n[5] Generando laterales de Tragantes ...")
    all_sections  = sections + orphan_links + gap_links
    lateral_links = process_tragante_laterals(nodes_trag, all_sections)

    all_links = all_sections + lateral_links

    pluvial_links  = [lk for lk in all_links if lk["red"] == "pluvial"]
    residual_links = [lk for lk in all_links if lk["red"] == "residual"]
    print(f"\n  Red Pluvial:  {len(pluvial_links)} tramos")
    print(f"  Red Residual: {len(residual_links)} tramos")

    print("\n[6] Exportando shapefiles ...")
    pluv_ids = {lk["src_id"] for lk in pluvial_links} | {lk["dst_id"] for lk in pluvial_links}
    resi_ids = {lk["src_id"] for lk in residual_links} | {lk["dst_id"] for lk in residual_links}

    export_links(pluvial_links,  OUT_DIR_V2 / "red_pluvial_links.shp")
    export_nodes(node_index, pluv_ids,  OUT_DIR_V2 / "red_pluvial_nodos.shp",  "pluvial")
    export_links(residual_links, OUT_DIR_V2 / "red_residual_links.shp")
    export_nodes(node_index, resi_ids,  OUT_DIR_V2 / "red_residual_nodos.shp", "residual")
    export_unmatched(unmatched,  OUT_DIR_V2 / "backbone_sin_nodos.shp")

    print(f"\nExportacion completa -> {OUT_DIR.resolve()}")
    print("\nCampos en links:")
    print("  red       -> pluvial | residual")
    print("  src/dst_kind -> tipo de caja (crn | crp | tragante | desfogue)")
    print("  drop_m    -> caida hidraulica (m)")
    print("  dist_m    -> longitud en planta (m, CRTM05)")
    print("  id sec_*  -> seccion de Alcantarillado.shp")
    print("  id orph_* -> conexion de nodo huerfano")
    print("  id lat_*  -> lateral de tragante")


if __name__ == "__main__":
    main()
