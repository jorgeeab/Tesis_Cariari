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
    from shapely.ops import nearest_points, substring
except ImportError:
    sys.exit("ERROR: instale geopandas: pip install geopandas")

# ── Paths ────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

ALCANTARILLADO_SHP = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Alcantarillado.shp"
LINEAS_SHP         = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/liteas a seguir.shp"
# jul 2026: capas reorganizadas por el usuario — CRP ampliado a 277 puntos,
# tragantes a 254; los nombres viejos (CRP.shp, CRN_CAPA.shp, Tragantes.shp) ya no existen
TRAGANTES_SHP      = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Tragantes (2).shp"
CRP_SHP            = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/CRP actualizado/CRP actualizado.shp"
CRN_SHP            = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/CRN (2).shp"
DESFOGUES_SHP      = BASE / "Arcgis shapes y curvas/Desfogues.shp"
# Conexiones trazadas a mano por el usuario sobre la capa de links exportada
# (solo se usan las filas SIN src_id: las dibujadas, no las re-exportadas)
LINEAS_USUARIO_SHP = BASE / "Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Pluvial actualizado/Pluvial actualizado.shp"
OUT_DIR            = BASE / "output/shapefiles"
OUT_DIR_V2         = BASE / "output/shapefiles_v7"   # v7: CRP ampliado + conexiones manuales

# ── Tolerancias (metros, CRTM05) ─────────────────────────────────────────────────
SNAP_ENDPOINT_TOL_M = 30.0   # caja cercana al extremo de una linea -> anclada al extremo
ON_LINE_TOL_M       = 15.0   # caja sobre el cuerpo de la linea si distancia perp <= esto
MIN_SECTION_M       = 1.0    # seccion mas corta aceptada (m)
CONNECT_ORPHAN_M    = 60.0   # max distancia para conectar huerfanos
LATERAL_MAX_M       = 60.0   # max distancia tragante -> segmento
MAX_ADVERSE_DROP_M  = -1.0   # caida hidraulica minima aceptada (negativo = sube; rechaza >1m de subida)
# Un tramo que pasa a menos de esto de una caja de su PROPIA red sin
# conectarla, se parte en esa caja para integrarla (el tubo real atraviesa
# la caja; dejarlo de largo duplica el corredor y salta la caja). Pasa con
# secciones cuya caja intermedia fue reclamada por otra linea guia, y con
# rutas de gap-fill (la guia puede correr hasta ~6 m de la caja).
SPLIT_NODE_TOL_M    = 6.5
SPLIT_END_MARGIN_M  = 8.0    # cajas tan cerca del extremo no cuentan como "de paso"

MIN_ELEV      = 900.0
MAX_ELEV      = 1100.0
DEFAULT_DEPTH = {"tragante": 0.80, "crp": 1.60, "crn": 1.40, "desfogue": 0.60}

# Desfogues: sin datos de campo. Se les asigna una cota MAS BAJA que la de los
# nodos vecinos para que la orientacion por gravedad siempre fluya hacia ellos.
DESFOGUE_DROP_M    = 1.0    # metros por debajo del nodo mas bajo del vecindario
DESFOGUE_RADIO_M   = 120.0  # radio de busqueda de nodos de referencia
DESFOGUE_CONNECT_M = 150.0  # radio de conexion de desfogues huerfanos (son
                            # los puntos de descarga: mejor un tramo largo
                            # estimado que dejarlos fuera de la red)

# Sentidos confirmados en campo/plano por el usuario, aunque las cotas de
# fondo digitalizadas indiquen lo contrario (se respetan tal cual en [5c]).
# Plano colector troncal Los Arcos-Cariari: el pluvial sigue la calle al sur.
FORCED_FLOW = [
    ("crp_7", "crp_9"),
    ("crp_9", "crp_10"),
    ("crp_75", "crp_76"),   # el pluvial sigue la calle al NO (plano, 2026-07-14)
    ("crp_52", "crp_53"),   # la calle baja al sur 52->53->58->desfogue_6; la
                            # cota F/967 de crp_53 es un outlier (vecinos ~957-961)
    ("crp_19", "crp_20"),   # fondos casi planos (5 cm); el usuario confirma 19->20
    ("crp_4", "crp_1"),     # el agua va al este hacia desfogue_7 (usuario);
                            # el F/3.40 de crp_4 parece doble fondo
    ("crp_x112", "crp_x110"),  # superficies identicas (955.36): fijar bajada
    # Corredor 41->42: los fondos reales caen 960.33 -> 960.05; las cotas
    # estimadas de los clones (superficie con loma en medio) partirian el
    # flujo en dos — el tubo real va enterrado bajo la loma.
    ("crp_41", "crp_x40"),
    ("crp_x40", "crp_x41"),
    ("crp_x41", "crp_x42"),
    ("crp_x42", "crp_42"),
    # La bajada 41 -> 40 -> x38 -> 27 (validada por el usuario hasta x38/27):
    # el F/948.88 de crp_40 queda por DEBAJO del fondo estimado de crp_x38
    # (prof 1.6 m, seguro mas somera que la real) — parece caida/doble fondo.
    ("crp_40", "crp_x38"),
]

# Cajas pluviales que existen en campo pero no estan en CRP actualizado:
# se clonan de la posicion de la caja CRN del mismo pozo/esquina.
# El id resultante es crp_x<numero> (p.ej. crn_115 -> crp_x115).
CRP_DESDE_CRN = ["crn_115", "crn_40", "crn_41", "crn_42", "crn_38",
                 "crn_160", "crn_103",
                 # bajada del hotel (crp_144) al desfogue_4 siguiendo los CRN
                 "crn_114", "crn_113", "crn_112", "crn_110", "crn_111",
                 "crn_109", "crn_108", "crn_107", "crn_106",
                 # C.R. SELLADA S/963.17 sobre la calle MUNI (plano): existe
                 # como caja pluvial pero solo estaba digitalizada como CRN.
                 # Clonarla hace que crp_117 baje por la calle (117->x148->109)
                 # en vez de la diagonal recta (usuario, 2026-07-16).
                 "crn_148"]

# Conexiones pluviales decididas por el usuario. Se enrutan por las lineas
# guia si es posible; si no, tramo recto. El sentido lo dan las reglas.
CONECTAR_DIRECTO = [
    # Colector del rio: crp_x115 baja por la calle y ENTRA a crp_x106, que
    # descarga al desfogue_4. (Antes iba directo a crp_75 pasando a 3 m del
    # x106 sin conectarlo: corredor duplicado — usuario, 2026-07-15.)
    ("crp_x115", "crp_x106"),
    ("crp_77", "crp_57"),      # el 77 drena via 57 (plano, 2026-07-14)
    ("crp_56", "desfogue_5"),  # seccion larga que los limites automaticos no tomaban
    ("crp_49", "crp_54"),      # seccion faltante (usuario, 2026-07-14): destapa
                               # el pozo de crp_49 (10 entradas sin salida)
    # Corredor crp_41 -> crp_42 por el camino real (via cajas junto a
    # crn_40/41/42; el directo de 155 m iba por la calle equivocada)
    ("crp_41", "crp_x40"),
    ("crp_x40", "crp_x41"),
    ("crp_x41", "crp_x42"),
    ("crp_x42", "crp_42"),
    # crp_43 sube a crp_38 por la calle de las "2 TUBERIAS paralelas de
    # A.N. y Pluvial" del plano. La rotonda izquierda (crp_x38) NO va por
    # la servidumbre: drena al oeste hacia crp_27 (colector Ø600).
    ("crp_43", "crp_38"),
    ("crp_19", "crp_20"),      # seccion real (usuario); el 20-21 era invento
    ("crp_81", "crp_79"),      # seccion faltante (usuario): destapa crp_81
    # Servidumbre por las canchas de golf (plano): el pluvial baja de la
    # zona PTAR/plaza por crp_59 hacia la calle de abajo
    ("crp_60", "crp_59"),
    ("crp_59", "crp_x160"),
    ("crp_x160", "crp_x103"),
    ("crp_51", "desfogue_6"),  # descarga de crp_51 (usuario): destapa el pozo
    ("crp_35", "crp_34"),      # seccion faltante (usuario): destapa crp_35
    ("crp_84", "crp_83"),      # seccion faltante (usuario)
    # Bajada del hotel Cariari: crp_144 sigue los CRN hasta desfogue_4
    ("crp_144", "crp_x114"),
    ("crp_x114", "crp_x113"),
    ("crp_x113", "crp_x112"),
    ("crp_x112", "crp_x110"),
    ("crp_x110", "crp_x111"),
    ("crp_x111", "crp_x109"),
    ("crp_x109", "crp_x108"),
    ("crp_x108", "crp_x107"),
    ("crp_x107", "crp_x106"),
    ("crp_x106", "desfogue_4"),
    ("crp_114", "crp_113"),    # seccion faltante (usuario): destapa crp_114
    ("crp_125", "crp_118"),    # el 125 va al 118, no al 126 (usuario)
    # Cadena real del cluster crp_109: 109 -> 107 -> 104 -> 106 (usuario
    # veto 109->104 y 107->106; el 107 continua al 104, a 15 m y mas bajo)
    ("crp_107", "crp_104"),
    # Descargas al rio aprobadas 2026-07-15: reutiliza D9/D6 y conecta cada
    # subred sin salida a su desfogue nuevo sobre el cauce.
    ("crp_134", "desfogue_9"),   # reutiliza: D9 a 76 m
    # desfogue_6 lo alimenta crp_58 (justo encima, a 77 m), no crp_52
    # (crp_52 baja por la calle 52->53->58; usuario, 2026-07-16)
    ("crp_58", "desfogue_6"),
    ("crp_10", "desfogue_10"),   # troncal Los Arcos (128 nodos) al rio oeste
    ("crp_101", "desfogue_11"),
    ("crp_110", "desfogue_12"),
    ("crp_137", "desfogue_13"),
    ("crp_131", "desfogue_14"),
    ("crp_90", "desfogue_15"),
    ("crp_88", "desfogue_16"),
]

# Pares que NUNCA deben conectarse (el gap-fill los inventaba).
# crp_x115 baja al crp_x106 (y de ahi al desfogue_4); crp_75 drena aparte
# hacia crp_76: sin atajos automaticos a los desfogues del rio.
NO_CONECTAR = [
    ("crp_75", "desfogue_4"),
    ("crp_x115", "crp_75"),     # duplicaba el corredor hacia x106 (usuario, 2026-07-15)
    ("crp_x115", "desfogue_2"),
    ("crp_x115", "desfogue_4"),
    ("crp_77", "desfogue_5"),   # el 77 NO descarga directo: va via crp_57
    ("crp_146", "crp_42"),      # ese tramo no existe en campo (usuario)
    ("crp_41", "crp_42"),       # el corredor 41-42 va via crp_x40/x41/x42
    ("crp_42", "crp_136"),      # tampoco existe en campo (usuario, 2026-07-14)
    ("crp_20", "crp_21"),       # no existe en campo: el 20 recibe del 19
    ("crp_43", "crp_x38"),      # la diagonal entre rotondas cruzaba la manzana
    ("crp_x38", "crp_38"),      # la servidumbre no lleva pluvial (usuario)
    ("crp_x103", "desfogue_5"), # su agua va por la calle: crp_77 -> crp_57
    ("crp_63", "crp_67"),       # no existe en campo (usuario, 2026-07-14)
    ("crp_74", "crp_71"),       # diagonal cruzando la manzana (usuario)
    ("crp_x106", "crp_75"),     # su agua va al desfogue_4 (a 6 m), no a crp_75
    ("crp_109", "crp_106"),     # no existe en campo (usuario, 2026-07-14)
    ("crp_125", "crp_126"),     # la diagonal no existe: va via crp_118 (usuario)
    ("crp_109", "crp_104"),     # gap saltaba 92 m; el 109 va via crp_107 (usuario, 2026-07-16)
    ("crp_107", "crp_106"),     # gap saltaba crp_104; el 107 va via 104 (usuario, 2026-07-16)
    ("crp_58", "crp_57"),       # gap con caida de 21 m: el 58 descarga a desfogue_6, no al 57 (usuario, 2026-07-16)
]

# Tragantes que NO deben generar lateral automatico (el mas cercano les
# quedaba cruzando casas u otras barreras; confirmar destino real en campo).
TRAGANTES_SIN_LATERAL = {
    "tragante_39",   # su lateral de 56 m cruzaba la manzana hasta crp_17
}

# Desplazamientos de desfogues (dx, dy en metros CRTM05: +x=este, +y=norte).
# Desfogues.shp trae los puntos levantados junto a la calle; el usuario los
# corre para que la descarga caiga EN el rio. No se toca el shapefile fuente.
MOVER_DESFOGUES = {
    "desfogue_4": (-20.0, -20.0),   # 20 m al oeste y 20 m al sur (usuario, 2026-07-15)
}

# Desfogues adicionales a los 9 de Desfogues.shp (aprobados por el usuario,
# 2026-07-15): puntos EXACTOS sobre el cauce (Cauce.shp) para descargar las
# subredes sin salida grandes. Ids explicitos para que no se renumeren.
DESFOGUES_EXTRA = [
    ("desfogue_10", 481254.5, 1102417.1),   # rio oeste, recibe crp_10 (143 m)
    ("desfogue_11", 482652.4, 1102186.8),   # quebrada sur, recibe crp_101 (33 m)
    ("desfogue_12", 482917.3, 1102312.3),   # quebrada SE, recibe crp_110 (62 m)
    ("desfogue_13", 483148.0, 1102688.9),   # Bermudez SE, recibe crp_137 (22 m)
    ("desfogue_14", 483214.0, 1102572.0),   # Bermudez SE, recibe crp_131 (78 m)
    ("desfogue_15", 482439.7, 1102118.3),   # quebrada sur, recibe crp_90 (57 m)
    ("desfogue_16", 482121.5, 1102059.8),   # quebrada sur, recibe crp_88 (90 m)
]
_NO_CONECTAR_SET = {tuple(sorted(p)) for p in NO_CONECTAR}


def _par_prohibido(id_a, id_b):
    return tuple(sorted((id_a, id_b))) in _NO_CONECTAR_SET


# ── Utilidades ───────────────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ── Carga de datos ───────────────────────────────────────────────────────────────

DEDUP_TOL_M = 0.1   # puntos a <10 cm son la misma caja digitalizada dos veces


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
    n_dup = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Point":
            continue
        x, y = float(geom.x), float(geom.y)
        if any(math.hypot(n["x"] - x, n["y"] - y) < DEDUP_TOL_M for n in nodes):
            n_dup += 1
            continue  # duplicado exacto: conservar el primero (ids estables)
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
    if n_dup:
        print(f"  {p.stem}: {n_dup} puntos duplicados descartados (mismo sitio)")
    return nodes


def clonar_crp_desde_crn(nodes_crp, nodes_crn):
    """Crea cajas CRP en la posicion de cajas CRN conocidas (CRP_DESDE_CRN).
    La caja pluvial real comparte esquina con la residual; se toma su cota de
    superficie y profundidad tipica de CRP (estimada). Se desplaza ~2 m para
    que ambos puntos se distingan en el mapa."""
    crn_por_id = {n["id"]: n for n in nodes_crn}
    for cid in CRP_DESDE_CRN:
        base = crn_por_id.get(cid)
        if base is None:
            print(f"  AVISO: CRP_DESDE_CRN no encontro {cid}")
            continue
        sup = base["sup_elev"]
        prof = DEFAULT_DEPTH["crp"]
        nodes_crp.append({
            "id":             f"crp_x{cid.split('_')[1]}",
            "kind":           "crp",
            "x":              base["x"] + 2.0,
            "y":              base["y"] - 2.0,
            "sup_elev":       sup,
            "fon_elev":       round(sup - prof, 3) if sup is not None else None,
            "prof_m":         prof,
            "prof_src":       "estimado",
            "hydraulic_elev": (sup - prof) if sup is not None else None,
        })
        print(f"  Clonado {cid} -> crp_x{cid.split('_')[1]} (S={sup}, prof estimada {prof})")


def connect_directos(sections, node_by_id, backbone_lines):
    """Crea las conexiones de CONECTAR_DIRECTO que aun no existan."""
    existentes = {tuple(sorted((s["src_id"], s["dst_id"]))) for s in sections}
    nuevos = []
    for a_id, b_id in CONECTAR_DIRECTO:
        if tuple(sorted((a_id, b_id))) in existentes:
            continue
        na, nb = node_by_id.get(a_id), node_by_id.get(b_id)
        if na is None or nb is None:
            print(f"  AVISO: CONECTAR_DIRECTO no encontro {a_id} o {b_id}")
            continue
        d_recta = math.hypot(na["x"] - nb["x"], na["y"] - nb["y"])
        routed = _route_via_line(na, nb, backbone_lines, tol=60.0)
        # Guarda anti-rodeo: si la ruta por la guia es mucho mas larga que
        # la distancia recta, la guia era otra calle — mejor tramo recto.
        if routed is not None and routed[0].length <= max(d_recta * 3.0, d_recta + 50.0):
            geom, n_s, n_d = routed
            nuevos.append(_make_section(f"dir_{len(nuevos) + 1}", n_s, n_d, geom))
        else:
            geom = LineString([(na["x"], na["y"]), (nb["x"], nb["y"])])
            nuevos.append(_make_section(f"dir_{len(nuevos) + 1}", na, nb, geom))
    print(f"  Conexiones directas: {len(nuevos)}")
    return nuevos


def integrar_tramos_sobre_cajas(links, box_nodes, node_by_id, prefijos,
                                tol=SPLIT_NODE_TOL_M):
    """Parte los tramos que PASAN sobre una caja de su propia red sin
    conectarla y la integra a la red. Ocurre cuando la caja intermedia fue
    reclamada por otra linea guia (la seccion la salta) o cuando un directo
    largo se dibujo antes de que existiera la caja. Las piezas que dupliquen
    un tramo existente o esten en NO_CONECTAR se descartan."""
    kind_por_red = {"pluvial": "crp", "residual": "crn"}
    existentes = {tuple(sorted((lk["src_id"], lk["dst_id"]))) for lk in links}
    out, divididos = [], 0
    for lk in links:
        if not lk["id"].startswith(prefijos):
            out.append(lk)
            continue
        geom, kind = lk["geom"], kind_por_red.get(lk["red"])
        sobre = []
        for n in box_nodes:
            if n["kind"] != kind or n["id"] in (lk["src_id"], lk["dst_id"]):
                continue
            p = Point(n["x"], n["y"])
            if geom.distance(p) > tol:
                continue
            t = geom.project(p)
            if SPLIT_END_MARGIN_M < t < geom.length - SPLIT_END_MARGIN_M:
                sobre.append((t, n))
        if not sobre:
            out.append(lk)
            continue
        sobre.sort(key=lambda tn: tn[0])
        filtrados = []
        for t, n in sobre:
            if not filtrados or t - filtrados[-1][0] >= 2.0:
                filtrados.append((t, n))
        print(f"  {lk['id']} ({lk['src_id']} -> {lk['dst_id']}) pasa sobre "
              + ", ".join(n["id"] for _, n in filtrados) + ": se parte")
        divididos += 1
        path = ([(0.0, node_by_id[lk["src_id"]])] + filtrados
                + [(geom.length, node_by_id[lk["dst_id"]])])
        for i in range(len(path) - 1):
            (t_a, n_a), (t_b, n_b) = path[i], path[i + 1]
            par = tuple(sorted((n_a["id"], n_b["id"])))
            if _par_prohibido(n_a["id"], n_b["id"]):
                print(f"    pieza {n_a['id']} -> {n_b['id']}: descartada (NO_CONECTAR)")
                continue
            if par in existentes:
                print(f"    pieza {n_a['id']} -> {n_b['id']}: descartada (ya existe)")
                continue
            sub = substring(geom, t_a, t_b)
            coords = list(sub.coords)
            coords[0]  = (n_a["x"], n_a["y"])
            coords[-1] = (n_b["x"], n_b["y"])
            pieza = LineString(coords)
            if pieza.length < MIN_SECTION_M:
                continue
            out.append(_make_section(f"{lk['id']}s{i + 1}", n_a, n_b, pieza))
            existentes.add(par)
    if divididos:
        print(f"  Tramos partidos sobre cajas propias: {divididos}")
    else:
        print("  Ningun tramo pasa sobre cajas propias.")
    return out


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
    for n in nodes:
        dx, dy = MOVER_DESFOGUES.get(n["id"], (0.0, 0.0))
        if dx or dy:
            n["x"] += dx
            n["y"] += dy
            print(f"  {n['id']} desplazado ({dx:+.0f}, {dy:+.0f}) m")
    for did, x, y in DESFOGUES_EXTRA:
        nodes.append({
            "id":             did,
            "kind":           "desfogue",
            "x":              x,
            "y":              y,
            "sup_elev":       None,
            "fon_elev":       None,
            "prof_m":         DEFAULT_DEPTH["desfogue"],
            "prof_src":       "estimado",
            "hydraulic_elev": None,
        })
    return nodes


def assign_desfogue_elevations(desfogues, ref_nodes):
    """
    Asigna cota de fondo a los desfogues (Desfogues.shp no trae elevaciones):
    DESFOGUE_DROP_M por debajo del nodo MAS BAJO dentro de DESFOGUE_RADIO_M.
    Asi `_make_section` orienta por gravedad siempre HACIA el desfogue.
    """
    refs = [n for n in ref_nodes if n["hydraulic_elev"] is not None]
    for d in desfogues:
        near = [n for n in refs
                if math.hypot(n["x"] - d["x"], n["y"] - d["y"]) <= DESFOGUE_RADIO_M]
        if near:
            base = min(n["hydraulic_elev"] for n in near)
        elif refs:
            nearest = min(refs, key=lambda n: math.hypot(n["x"] - d["x"], n["y"] - d["y"]))
            base = nearest["hydraulic_elev"]
        else:
            base = 935.0
        he = round(base - DESFOGUE_DROP_M, 3)
        d["hydraulic_elev"] = he
        d["fon_elev"]       = he
        d["sup_elev"]       = round(he + d["prof_m"], 3)


def _line_hash(coords):
    return tuple((round(x, 1), round(y, 1)) for x, y in coords)


def load_backbone_lines():
    """Carga Alcantarillado.shp + liteas a seguir.shp + conexiones manuales
    del usuario (Pluvial actualizado, solo filas dibujadas) sin duplicados."""
    seen  = set()
    lines = []
    for shp, prefix in [(ALCANTARILLADO_SHP, "alc"), (LINEAS_SHP, "lin"),
                        (LINEAS_USUARIO_SHP, "usr")]:
        if not Path(shp).exists():
            print(f"  AVISO: no se encontro {Path(shp).name}")
            continue
        gdf = gpd.read_file(str(shp))
        if gdf.crs and gdf.crs.to_epsg() != 5367:
            gdf = gdf.to_crs(epsg=5367)
        if prefix == "usr" and "src_id" in gdf.columns:
            # Solo las lineas trazadas a mano (las re-exportadas traen src_id)
            gdf = gdf[gdf["src_id"].isna() | (gdf["src_id"].astype(str) == "")]
            print(f"  Conexiones manuales del usuario: {len(gdf)}")
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


def _dobla_atras(a, b, c, umbral_deg=100.0):
    """True si en b el trazo a->b->c 'se devuelve' (gira mas que umbral)."""
    v1x, v1y = b.x - a.x, b.y - a.y
    v2x, v2y = c.x - b.x, c.y - b.y
    l1 = math.hypot(v1x, v1y)
    l2 = math.hypot(v2x, v2y)
    if l1 < 0.01 or l2 < 0.01:
        return False
    cosang = (v1x * v2x + v1y * v2y) / (l1 * l2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang)))) > umbral_deg


def extract_sub_segment(line_geom, d_start, d_end, start_xy=None, end_xy=None):
    """
    Extrae subsegmento entre d_start y d_end preservando vertices originales.
    Los extremos se anclan a las coordenadas exactas de los nodos (fijos).

    Poda de ganchos: si la linea guia fue dibujada pasandose de la caja
    (tipico en extremos con snap), quedaban vertices sobrantes y el trazo
    se devolvia en curva antes de llegar al punto. Se eliminan los vertices
    junto a cada extremo mientras el giro ahi supere los 100 grados.
    """
    p_start = Point(start_xy) if start_xy is not None else line_geom.interpolate(d_start)
    inner   = []
    for coord in line_geom.coords:
        d = line_geom.project(Point(coord))
        if d_start < d < d_end:
            inner.append(Point(coord))
    p_end = Point(end_xy) if end_xy is not None else line_geom.interpolate(d_end)

    pts = [p_start] + inner + [p_end]

    while len(pts) > 2 and _dobla_atras(pts[0], pts[1], pts[2]):
        pts.pop(1)
    while len(pts) > 2 and _dobla_atras(pts[-3], pts[-2], pts[-1]):
        pts.pop(-2)

    # Vertices pegados al nodo (<5 m): la linea guia suele estar dibujada
    # hasta la caja VECINA (p.ej. la CRN de la esquina) y el trazo tocaba ese
    # punto ajeno antes de saltar a su caja. Entrar directo al nodo.
    while len(pts) > 2 and pts[1].distance(pts[0]) < 5.0:
        pts.pop(1)
    while len(pts) > 2 and pts[-2].distance(pts[-1]) < 5.0:
        pts.pop(-2)

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
    """El sentido queda src -> dst. Reglas de orientacion (en orden):
    1. Un desfogue solo puede RECIBIR flujo (siempre queda como dst).
    2. Si alguna caja tiene profundidad ESTIMADA (p.ej. "sellada"), su fondo
       calculado no es confiable: se orienta por la SUPERFICIE medida.
    3. Gravedad: src = mayor cota de fondo (hydraulic_elev).
    4. Sin datos suficientes: se conserva el orden de llegada (trazado)."""
    he_src = n_src["hydraulic_elev"]
    he_dst = n_dst["hydraulic_elev"]
    ori = "fon"
    if n_src["kind"] == "desfogue" and n_dst["kind"] != "desfogue":
        flip = True
    elif n_dst["kind"] == "desfogue":
        flip = False
    elif ((n_src["prof_src"] == "estimado" or n_dst["prof_src"] == "estimado")
          and n_src["sup_elev"] is not None and n_dst["sup_elev"] is not None):
        flip = n_src["sup_elev"] < n_dst["sup_elev"]
        ori = "sup"
    else:
        flip = he_src is not None and he_dst is not None and he_src < he_dst
    if flip:
        n_src, n_dst = n_dst, n_src
        geom = LineString(list(reversed(geom.coords)))
        he_src, he_dst = he_dst, he_src
    drop_m = round(he_src - he_dst, 3) if he_src is not None and he_dst is not None else None
    return {
        "_ori":     ori,
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


def _flip_link(lk):
    """Invierte el sentido de un tramo ya construido (src <-> dst)."""
    lk["src_id"],   lk["dst_id"]   = lk["dst_id"],   lk["src_id"]
    lk["src_kind"], lk["dst_kind"] = lk["dst_kind"], lk["src_kind"]
    lk["src_elev"], lk["dst_elev"] = lk["dst_elev"], lk["src_elev"]
    lk["src_prof"], lk["dst_prof"] = lk["dst_prof"], lk["src_prof"]
    lk["snap_src"], lk["snap_dst"] = lk["snap_dst"], lk["snap_src"]
    lk["geom"] = LineString(list(reversed(lk["geom"].coords)))
    if lk["src_elev"] is not None and lk["dst_elev"] is not None:
        lk["drop_m"] = round(lk["src_elev"] - lk["dst_elev"], 3)


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
    crn_puros = [n for n in box_nodes if n["kind"] == "crn"]
    crp_puros = [n for n in box_nodes if n["kind"] == "crp"]
    desfogues = [n for n in box_nodes if n["kind"] == "desfogue"]

    sections   = []
    unmatched  = []
    seen_pairs = set()   # evita tramos duplicados cuando dos guias cubren el mismo par

    for line in backbone_lines:
        line_geom = LineString(line["coords"])

        # El tipo lo deciden los nodos CRN/CRP puros; los desfogues son
        # neutrales (pueden rematar lineas de cualquiera de las dos redes).
        net_type, _ = _type_of_line(line_geom, crn_puros, crp_puros)
        if net_type is None:
            unmatched.append({
                "id": line["id"], "geom": line_geom,
                "motivo": "sin nodos en extremos",
            })
            continue

        # Solo nodos del tipo correcto sobre esta linea (+ desfogues)
        type_nodes = (crn_puros if net_type == "crn" else crp_puros) + desfogues
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
            if n_src["kind"] == "desfogue" and n_dst["kind"] == "desfogue":
                continue  # dos descargas no forman tramo
            if _par_prohibido(n_src["id"], n_dst["id"]):
                continue
            pair = (min(n_src["id"], n_dst["id"]), max(n_src["id"], n_dst["id"]))
            if pair in seen_pairs:
                continue
            seg_geom = extract_sub_segment(
                line_geom, d0, d1,
                start_xy=(n_src["x"], n_src["y"]),
                end_xy=(n_dst["x"], n_dst["y"]),
            )
            if seg_geom is None:
                continue
            seen_pairs.add(pair)
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
    """True si ambos nodos pueden pertenecer a la misma red (nunca CRP<->CRN).
    Los desfogues son puntos de descarga de CUALQUIERA de las dos redes,
    pero jamas se conectan entre si (ambos son terminales)."""
    if kind_a == "desfogue" and kind_b == "desfogue":
        return False
    residual = {"crn", "desfogue"}
    pluvial  = {"crp", "desfogue"}
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
            if _par_prohibido(orph["id"], n["id"]):
                continue
            d = math.hypot(n["x"] - ox, n["y"] - oy)
            if d < best_d:
                best_d = d
                best_n = n
        max_d = DESFOGUE_CONNECT_M if orph["kind"] == "desfogue" else CONNECT_ORPHAN_M
        if best_n is None or best_d > max_d:
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
    crp_group = [n for n in box_nodes if n["kind"] in ("crp", "desfogue")]

    new_sections = []

    for group in [crn_group, crp_group]:
        for n_a in group:
            best_score   = float("inf")
            best_routed  = None
            best_pair    = None

            for n_b in group:
                if n_b["id"] == n_a["id"]:
                    continue
                if not _compatible_kinds(n_a["kind"], n_b["kind"]):
                    continue
                if _par_prohibido(n_a["id"], n_b["id"]):
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
        if trag["id"] in trag_already or trag["id"] in TRAGANTES_SIN_LATERAL:
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


def export_puntos_ids(nodes, out_path):
    """Exporta TODOS los puntos de un tipo con su id asignado (esten o no
    conectados a la red), para identificarlos con etiquetas en QGIS."""
    rows = [
        {
            "geometry": Point(n["x"], n["y"]),
            "id":       n["id"],
            "kind":     n["kind"],
            "sup_elev": n["sup_elev"],
            "fon_elev": n["fon_elev"],
            "prof_m":   n["prof_m"],
            "prof_src": n["prof_src"],
        }
        for n in nodes
    ]
    if not rows:
        return
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:5367")
    gdf.to_file(str(out_path), encoding="utf-8")
    print(f"  -> {out_path.name}  ({len(gdf)} puntos con id)")


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
    clonar_crp_desde_crn(nodes_crp, nodes_crn)
    node_index = nodes_trag + nodes_crp + nodes_crn + nodes_des
    node_by_id = {n["id"]: n for n in node_index}
    print(f"  Tragantes: {len(nodes_trag)} | CRP: {len(nodes_crp)} | CRN: {len(nodes_crn)} | Desfogues: {len(nodes_des)}")

    print("\n[2b] Asignando cota a desfogues (minimo del vecindario - "
          f"{DESFOGUE_DROP_M} m) ...")
    assign_desfogue_elevations(nodes_des, nodes_crp + nodes_crn)
    for d in nodes_des:
        print(f"  {d['id']}: fondo {d['fon_elev']} m")

    box_nodes = nodes_crp + nodes_crn + nodes_des

    print("\n[3] Construyendo secciones nodo-a-nodo por linea de alcantarillado ...")
    sections, unmatched = process_lines_into_sections(backbone, box_nodes)

    print("\n[3b] Conexiones directas decididas por el usuario ...")
    sections += connect_directos(sections, node_by_id, backbone)

    print("\n[3c] Integrando cajas propias bajo secciones y tramos directos ...")
    sections = integrar_tramos_sobre_cajas(sections, box_nodes, node_by_id,
                                           ("sec_", "dir_"))

    print("\n[4] Conectando nodos huerfanos ...")
    orphan_links = connect_orphan_nodes(sections, box_nodes, node_by_id, backbone)

    print("\n[4b] Gap-fill: conectando nodos del mismo tipo via alcantarillado ...")
    gap_links = connect_same_type_gaps(sections + orphan_links, box_nodes, backbone)

    all_sections = sections + orphan_links + gap_links

    print("\n[4b1] Integrando cajas propias bajo tramos automaticos ...")
    all_sections = integrar_tramos_sobre_cajas(all_sections, box_nodes, node_by_id,
                                               ("gap_", "orph_"))

    # Triangulos: si A y B ya drenan al mismo punto (p.ej. ambos a un
    # desfogue) y ademas existe un tramo automatico A-B, ese tramo es
    # redundante y dibuja un "triangulo" sin sentido hidraulico. Se elimina
    # el tramo AUTO (gap_/orph_) de cada triangulo; los dibujados (sec_) y
    # los decididos por el usuario (dir_) nunca se tocan.
    print("\n[4b2] Eliminando tramos automaticos que forman triangulos ...")
    n_tri = 0
    while True:
        pares = {}
        for lk in all_sections:
            pares[frozenset((lk["src_id"], lk["dst_id"]))] = lk
        ady = {}
        for lk in all_sections:
            ady.setdefault(lk["src_id"], set()).add(lk["dst_id"])
            ady.setdefault(lk["dst_id"], set()).add(lk["src_id"])
        candidatos = []
        vistos = set()
        for a in ady:
            for b in ady[a]:
                for c in ady[a] & ady.get(b, set()):
                    tri = frozenset((a, b, c))
                    if len(tri) != 3 or tri in vistos:
                        continue
                    vistos.add(tri)
                    for par in [(a, b), (b, c), (a, c)]:
                        lk = pares.get(frozenset(par))
                        if lk is not None and lk["id"].split("_")[0] in ("gap", "orph"):
                            es_gap = lk["id"].startswith("gap")
                            drop = abs(lk["drop_m"]) if lk["drop_m"] is not None else 0.0
                            candidatos.append((0 if es_gap else 1, drop, lk))
        if not candidatos:
            break
        candidatos.sort(key=lambda t: (t[0], t[1]))
        quitar = candidatos[0][2]
        all_sections.remove(quitar)
        n_tri += 1
    print(f"  Tramos redundantes eliminados: {n_tri}")

    # Limpieza visual: las lineas guia suelen estar dibujadas pasando por la
    # caja de la OTRA red (CRP y CRN comparten esquina a 2-4 m); quitar los
    # vertices interiores que caen sobre esas cajas ajenas. Debe correr ANTES
    # de los laterales, para que las proyecciones de los tragantes caigan
    # sobre el trazo definitivo (no sobre el trazo previo a la limpieza).
    print("\n[4c] Limpiando vertices que caen sobre cajas de la otra red ...")
    crp_xy = [(n["x"], n["y"]) for n in node_index if n["kind"] == "crp"]
    crn_xy = [(n["x"], n["y"]) for n in node_index if n["kind"] == "crn"]

    def _limpia_vertices_ajenos(lk, ajenos):
        coords = list(lk["geom"].coords)
        if len(coords) <= 2:
            return 0
        keep, quitados = [coords[0]], 0
        for c in coords[1:-1]:
            if any(math.hypot(c[0] - x, c[1] - y) < 1.5 for x, y in ajenos):
                quitados += 1
                continue
            keep.append(c)
        keep.append(coords[-1])
        if quitados:
            lk["geom"] = LineString(keep)
            lk["dist_m"] = round(lk["geom"].length, 1)
        return quitados

    n_limpiados = sum(
        _limpia_vertices_ajenos(lk, crn_xy if lk["red"] == "pluvial" else crp_xy)
        for lk in all_sections
    )
    print(f"  Vertices sobre cajas ajenas eliminados: {n_limpiados}")

    print("\n[5] Generando laterales de Tragantes ...")
    lateral_links = process_tragante_laterals(nodes_trag, all_sections)

    all_links = all_sections + lateral_links

    # Propagar cotas de fondo y superficie estimadas (flujo por gravedad al 1% de pendiente)
    print("\n[5b] Propagando cotas estimadas por gravedad (1% pendiente) a nodos sin cota (desfogues, etc.) ...")
    for lk in all_links:
        src_elev = lk["src_elev"]
        dst_elev = lk["dst_elev"]
        dist = lk["dist_m"]
        
        if src_elev is not None and dst_elev is None:
            lk["dst_elev"] = round(src_elev - (dist * 0.01), 3)
        elif dst_elev is not None and src_elev is None:
            lk["src_elev"] = round(dst_elev + (dist * 0.01), 3)
        elif src_elev is None and dst_elev is None:
            lk["src_elev"] = 936.0
            lk["dst_elev"] = round(936.0 - (dist * 0.01), 3)

        if lk["drop_m"] is None:
            lk["drop_m"] = round(lk["src_elev"] - lk["dst_elev"], 3)

    # Actualizar cotas en el índice de nodos
    for lk in all_links:
        src_id = lk["src_id"]
        dst_id = lk["dst_id"]
        if src_id in node_by_id and node_by_id[src_id]["fon_elev"] is None:
            node_by_id[src_id]["fon_elev"] = lk["src_elev"]
            node_by_id[src_id]["sup_elev"] = round(lk["src_elev"] + node_by_id[src_id]["prof_m"], 3)
            node_by_id[src_id]["hydraulic_elev"] = lk["src_elev"]
        if dst_id in node_by_id and node_by_id[dst_id]["fon_elev"] is None:
            node_by_id[dst_id]["fon_elev"] = lk["dst_elev"]
            node_by_id[dst_id]["sup_elev"] = round(lk["dst_elev"] + node_by_id[dst_id]["prof_m"], 3)
            node_by_id[dst_id]["hydraulic_elev"] = lk["dst_elev"]

    # Un desfogue debe ser el punto MAS BAJO de sus tramos. La cota por
    # vecindario (120 m) puede quedar alta si el nodo que le aporta esta mas
    # lejos que el radio (p.ej. desfogue_10, con crp_10 a 143 m): se baja a
    # (minimo de los aportes - DESFOGUE_DROP_M) y se recalculan los drops.
    print("\n[5b2] Desfogues por debajo de sus aportes ...")
    aportes = {}
    for lk in all_links:
        if lk["dst_kind"] == "desfogue" and lk["src_elev"] is not None:
            aportes.setdefault(lk["dst_id"], []).append(lk["src_elev"])
    ajustados = 0
    for did, elevs in aportes.items():
        n_d = node_by_id[did]
        piso = round(min(elevs) - DESFOGUE_DROP_M, 3)
        if n_d["fon_elev"] is None or n_d["fon_elev"] > piso:
            print(f"  {did}: cota {n_d['fon_elev']} -> {piso}")
            n_d["fon_elev"] = piso
            n_d["hydraulic_elev"] = piso
            n_d["sup_elev"] = round(piso + n_d["prof_m"], 3)
            ajustados += 1
    if ajustados:
        for lk in all_links:
            if lk["dst_kind"] == "desfogue":
                lk["dst_elev"] = node_by_id[lk["dst_id"]]["fon_elev"]
                if lk["src_elev"] is not None:
                    lk["drop_m"] = round(lk["src_elev"] - lk["dst_elev"], 3)
    else:
        print("  Todos los desfogues ya estan por debajo de sus aportes.")

    # Verificacion final de sentido: con las cotas ya completas, ningun tramo
    # (salvo laterales de tragante, que son topologicos) debe "subir".
    # Los pares de FORCED_FLOW se imponen tal cual (conocimiento de campo).
    print("\n[5c] Verificacion final de sentido ...")
    forced = {tuple(p) for p in FORCED_FLOW}
    flipped = n_forced = 0
    for lk in all_links:
        par = (lk["src_id"], lk["dst_id"])
        if par in forced:
            n_forced += 1
            continue                # ya va en el sentido decretado
        if (par[1], par[0]) in forced:
            _flip_link(lk)
            flipped += 1
            n_forced += 1
            continue
        if lk["src_kind"] == "tragante" or lk["dst_kind"] == "desfogue":
            continue  # direccion topologica correcta por construccion
        if lk["src_kind"] == "desfogue":
            _flip_link(lk)          # un desfogue jamas origina flujo
            flipped += 1
            continue
        se, de = lk["src_elev"], lk["dst_elev"]
        if lk.get("_ori") == "sup":
            # Orientado por superficie (profundidad estimada): comparar sup
            ss = se + lk["src_prof"] if se is not None and lk["src_prof"] is not None else None
            ds = de + lk["dst_prof"] if de is not None and lk["dst_prof"] is not None else None
            if ss is not None and ds is not None and ss < ds:
                _flip_link(lk)
                flipped += 1
        elif se is not None and de is not None and se < de:
            _flip_link(lk)
            flipped += 1
    print(f"  Tramos re-orientados: {flipped} (sentidos forzados aplicados: {n_forced}/{len(forced)})")

    # Dos tramos troncales de la misma red corriendo paralelos (<3 m) por mas
    # de 30 m suelen ser el mismo tubo digitalizado dos veces. Solo avisa: la
    # correccion la decide el usuario (NO_CONECTAR / CONECTAR_DIRECTO).
    print("\n[5d] Control de corredores duplicados ...")
    troncales = [lk for lk in all_links if not lk["id"].startswith("lat_")]
    bounds = [lk["geom"].bounds for lk in troncales]
    avisos = 0
    for i, a in enumerate(troncales):
        ax0, ay0, ax1, ay1 = bounds[i]
        for j in range(i + 1, len(troncales)):
            b = troncales[j]
            if a["red"] != b["red"]:
                continue
            bx0, by0, bx1, by1 = bounds[j]
            if bx0 > ax1 + 3 or bx1 < ax0 - 3 or by0 > ay1 + 3 or by1 < ay0 - 3:
                continue
            ga, gb = a["geom"], b["geom"]
            comun = {a["src_id"], a["dst_id"]} & {b["src_id"], b["dst_id"]}
            for nid in comun:
                n = node_by_id[nid]
                mask = Point(n["x"], n["y"]).buffer(20.0)
                ga, gb = ga.difference(mask), gb.difference(mask)
            if ga.is_empty or gb.is_empty:
                continue
            solape = gb.intersection(ga.buffer(3.0)).length
            if solape > 30.0:
                print(f"  AVISO: {a['id']} ({a['src_id']} -> {a['dst_id']}) y {b['id']} "
                      f"({b['src_id']} -> {b['dst_id']}) comparten {solape:.0f} m de corredor")
                avisos += 1
    if not avisos:
        print("  Sin corredores duplicados.")

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

    # Capas de identificacion: todos los puntos con su id (para etiquetas QGIS)
    export_puntos_ids(nodes_crp,  OUT_DIR_V2 / "crp_puntos_ids.shp")
    export_puntos_ids(nodes_crn,  OUT_DIR_V2 / "crn_puntos_ids.shp")
    export_puntos_ids(nodes_trag, OUT_DIR_V2 / "tragantes_puntos_ids.shp")
    export_puntos_ids(nodes_des,  OUT_DIR_V2 / "desfogues_puntos_ids.shp")

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
