"""
exportar_redes_a_js.py
Convierte shapefiles v6 de red hidraulica al formato JS para el mapa 3D Three.js.
Genera RED_PLUVIAL_LINKS, RED_PLUVIAL_NODES, RED_RESIDUAL_LINKS, RED_RESIDUAL_NODES.
"""

import sys
import json
import math
from pathlib import Path

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError:
    sys.exit("pip install geopandas")

BASE       = Path(__file__).parent
SHP_DIR    = BASE / "output/shapefiles_v7"
HTML_PATH  = BASE / "resultados_refinados/Refinado_30_ThreeJS_Cortina_Ajustada.html"

# Centro de escena extraido del HTML (Lat/Lon en el panel de info)
LON_CENTER = -84.1614
LAT_CENTER =  9.9729
MPD_LON    = 111320.0 * math.cos(math.radians(LAT_CENTER))
MPD_LAT    = 110540.0
Z_EXAG     = 1.0

# Elevacion de terreno base para puntos sin elevacion de fondo
TERRAIN_BASE = 936.0


def crtm_to_scene(x_crtm, y_crtm, elev=None):
    """Convierte coords CRTM05 -> escena Three.js via WGS84."""
    # Reproyectar punto individual CRTM05 -> WGS84
    gdf = gpd.GeoDataFrame(geometry=[Point(x_crtm, y_crtm)], crs="EPSG:5367")
    gdf_wgs = gdf.to_crs(epsg=4326)
    lon = float(gdf_wgs.geometry.iloc[0].x)
    lat = float(gdf_wgs.geometry.iloc[0].y)
    sx = (lon - LON_CENTER) * MPD_LON
    sz = -(lat - LAT_CENTER) * MPD_LAT
    sy = (elev if elev is not None else TERRAIN_BASE) * Z_EXAG
    return round(sx, 2), round(sy, 2), round(sz, 2)


def build_node_js(shp_path):
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs.to_epsg() != 5367:
        gdf = gdf.to_crs(epsg=5367)
    gdf_wgs = gdf.to_crs(epsg=4326)

    nodes = []
    for i, row in gdf.iterrows():
        x, y = float(row.geometry.x), float(row.geometry.y)
        lon  = float(gdf_wgs.geometry.iloc[i].x)
        lat  = float(gdf_wgs.geometry.iloc[i].y)
        fon  = row.get("fon_elev")
        sup  = row.get("sup_elev")
        fon_f = float(fon) if fon is not None and str(fon) != 'None' else None
        sup_f = float(sup) if sup is not None and str(sup) != 'None' else None
        sy = (fon_f if fon_f is not None else TERRAIN_BASE) * Z_EXAG + 1.5
        sx = (lon - LON_CENTER) * MPD_LON
        sz = -(lat - LAT_CENTER) * MPD_LAT
        nodes.append({
            "id":      str(row.get("id", f"n{i}")),
            "kind":    str(row.get("kind", "")),
            "x":       round(sx, 2),
            "y":       round(sy, 2),
            "z":       round(sz, 2),
            "bottom":  round(fon_f, 2) if fon_f is not None else None,
            "surface": round(sup_f, 2) if sup_f is not None else None,
        })
    return nodes


def build_link_js(shp_path):
    gdf = gpd.read_file(str(shp_path))
    if gdf.crs.to_epsg() != 5367:
        gdf = gdf.to_crs(epsg=5367)
    gdf_wgs = gdf.to_crs(epsg=4326)

    links = []
    for i, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # fon_elev (fondo) + prof_m (profundidad) = sup_elev (superficie = terreno)
        # Usar sup_elev + 1.0 para que la tuberia sea visible sobre el terreno
        def to_f(v):
            return float(v) if v is not None and str(v) not in ('None', 'nan') else None

        src_fon  = to_f(row.get("src_elev"))
        dst_fon  = to_f(row.get("dst_elev"))
        src_prof = to_f(row.get("src_prof")) or 1.4
        dst_prof = to_f(row.get("dst_prof")) or 1.4

        # Superficie = fondo + profundidad; si no hay fondo usar TERRAIN_BASE
        src_sup = (src_fon + src_prof) if src_fon is not None else TERRAIN_BASE
        dst_sup = (dst_fon + dst_prof) if dst_fon is not None else TERRAIN_BASE

        coords = list(geom.coords)
        n_pts  = len(coords)

        pts_gdf = gpd.GeoDataFrame(geometry=[Point(c[0], c[1]) for c in coords], crs="EPSG:5367")
        pts_wgs = pts_gdf.to_crs(epsg=4326)

        pts_3d = []
        buried_pts_3d = []
        for j, pt_wgs in enumerate(pts_wgs.geometry):
            lon  = float(pt_wgs.x)
            lat  = float(pt_wgs.y)
            frac = j / max(1, n_pts - 1)
            # Interpolar superficie y fondo a lo largo del tramo
            sup_val = src_sup + frac * (dst_sup - src_sup)
            fon_val = (src_fon + frac * (dst_fon - src_fon)) if src_fon is not None and dst_fon is not None else (sup_val - (src_prof + frac * (dst_prof - src_prof)))
            
            sy = (sup_val + 0.55) * Z_EXAG
            sy_buried = fon_val * Z_EXAG
            
            sx = (lon - LON_CENTER) * MPD_LON
            sz = -(lat - LAT_CENTER) * MPD_LAT
            pts_3d.append([round(sx, 2), round(sy, 2), round(sz, 2)])
            buried_pts_3d.append([round(sx, 2), round(sy_buried, 2), round(sz, 2)])

        drop = row.get("drop_m")
        dist = row.get("dist_m")
        links.append({
            "id":       str(row.get("id", f"lk{i}")),
            "src_kind": str(row.get("src_kind", "")),
            "dst_kind": str(row.get("dst_kind", "")),
            "drop_m":   round(float(drop), 2) if to_f(drop) is not None else None,
            "dist_m":   round(float(dist), 1) if to_f(dist) is not None else None,
            "points":   pts_3d,
            "buried_points": buried_pts_3d,
        })
    return links


def main():
    print("[1] Convirtiendo tramos...")
    pluv_links = build_link_js(SHP_DIR / "red_pluvial_links.shp")
    resi_links = build_link_js(SHP_DIR / "red_residual_links.shp")
    print(f"  Pluvial: {len(pluv_links)} tramos | Residual: {len(resi_links)} tramos")

    # Solo tramos — los nodos ya están en el mapa como capas CRP/CRN
    js = (
        f"const RED_PLUVIAL_LINKS = {json.dumps(pluv_links, ensure_ascii=False)};\n"
        f"const RED_RESIDUAL_LINKS = {json.dumps(resi_links, ensure_ascii=False)};\n"
    )

    print("[3] Inyectando en HTML...")
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html = f.read()

    MARKER_DATA    = "/* === RED_HIDRAULICA_SEPARADA_DATA === */"
    MARKER_LAYERS  = "/* === RED_HIDRAULICA_SEPARADA_LAYERS === */"
    MARKER_PANEL   = "<!-- RED_HIDRAULICA_SEPARADA_PANEL -->"
    MARKER_LEGEND  = "<!-- RED_HIDRAULICA_SEPARADA_LEGEND -->"

    # ── Insertar datos JS ─────────────────────────────────────────────────────
    if MARKER_DATA in html:
        # Reemplazar bloque existente
        start = html.find(MARKER_DATA)
        end   = html.find(MARKER_DATA, start + 1) + len(MARKER_DATA)
        html  = html[:start] + MARKER_DATA + "\n" + js + "\n" + MARKER_DATA + html[end:]
    else:
        # Insertar antes del cierre del último <script> antes de </body>
        insert_at = html.rfind("</script>")
        html = (html[:insert_at]
                + f"\n{MARKER_DATA}\n{js}\n{MARKER_DATA}\n"
                + html[insert_at:])

    # ── Insertar código de capas ──────────────────────────────────────────────
    layers_js = _build_layers_js()
    if MARKER_LAYERS in html:
        start = html.find(MARKER_LAYERS)
        end   = html.find(MARKER_LAYERS, start + 1) + len(MARKER_LAYERS)
        html  = html[:start] + MARKER_LAYERS + "\n" + layers_js + "\n" + MARKER_LAYERS + html[end:]
    else:
        insert_at = html.rfind("</script>")
        html = (html[:insert_at]
                + f"\n{MARKER_LAYERS}\n{layers_js}\n{MARKER_LAYERS}\n"
                + html[insert_at:])

    # ── Panel de capas ────────────────────────────────────────────────────────
    # El bloque debe quedar DENTRO de #layer-panel (si queda tras </script>,
    # como lo emite la plantilla del generador, los checkboxes no se ven).
    # Estrategia: eliminar todo bloque existente y re-insertar en el ancla.
    MARKER_PANEL_END = "<!-- /RED_HIDRAULICA_SEPARADA_PANEL -->"
    panel_html = _build_panel_html()
    while MARKER_PANEL in html:
        start = html.find(MARKER_PANEL)
        end   = html.find(MARKER_PANEL_END, start)
        end   = end + len(MARKER_PANEL_END) if end >= 0 else start + len(MARKER_PANEL)
        html  = html[:start] + html[end:]
    panel_block = f"{MARKER_PANEL}\n{panel_html}\n  {MARKER_PANEL_END}"
    for anchor in ('<label class="layer-row"><input id="toggle-hydraulic-nodes"',
                   '<label class="layer-row"><input id="toggle-hydraulic-links"'):
        idx = html.find(anchor)
        if idx >= 0:
            end_label = html.find("</label>", idx) + len("</label>")
            html = html[:end_label] + f"\n  {panel_block}" + html[end_label:]
            break
    else:
        idx = html.find('<div id="layer-panel">')
        if idx >= 0:
            end_tag = html.find(">", idx) + 1
            html = html[:end_tag] + f"\n  {panel_block}" + html[end_tag:]
        else:
            print("  AVISO: no se encontro ancla para el panel de redes")

    # ── Leyenda ───────────────────────────────────────────────────────────────
    MARKER_LEGEND_END = "<!-- /RED_HIDRAULICA_SEPARADA_LEGEND -->"
    legend_html = _build_legend_html()
    if MARKER_LEGEND in html:
        start = html.find(MARKER_LEGEND)
        end   = html.find(MARKER_LEGEND_END, start) + len(MARKER_LEGEND_END)
        html  = html[:start] + MARKER_LEGEND + "\n" + legend_html + "\n  " + MARKER_LEGEND_END + html[end:]
    else:
        anchor = '<div id="legend"'
        idx = html.find(anchor)
        if idx >= 0:
            end_tag = html.find(">", idx) + 1
            html = html[:end_tag] + f"\n  {MARKER_LEGEND}\n{legend_html}\n  {MARKER_LEGEND_END}" + html[end_tag:]

    # ── Eventos ───────────────────────────────────────────────────────────────
    MARKER_EVENTS = "/* === RED_HIDRAULICA_SEPARADA_EVENTS === */"
    events_js = _build_events_js()
    if MARKER_EVENTS in html:
        start = html.find(MARKER_EVENTS)
        end   = html.find(MARKER_EVENTS, start + 1) + len(MARKER_EVENTS)
        html  = html[:start] + MARKER_EVENTS + "\n" + events_js + "\n" + MARKER_EVENTS + html[end:]
    else:
        insert_at = html.rfind("</script>")
        html = html[:insert_at] + f"\n{MARKER_EVENTS}\n{events_js}\n{MARKER_EVENTS}\n" + html[insert_at:]

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  -> {HTML_PATH.name} actualizado")
    print("Listo.")


def _build_layers_js():
    return """
// ── Redes Hidráulicas Separadas ───────────────────────────────────────────────
const pluvialLinkLayer   = new THREE.Group();
const residualLinkLayer  = new THREE.Group();
const pluvialArrowLayer  = new THREE.Group();
const residualArrowLayer = new THREE.Group();
scene.add(pluvialLinkLayer);
scene.add(residualLinkLayer);
scene.add(pluvialArrowLayer);
scene.add(residualArrowLayer);

const redesState = { pluvial: true, residual: true, arrows: true };
const REDES_ARROW_SPACING_M = 45;   // separacion entre flechas a lo largo del tramo
const REDES_ARROW_MIN_LEN_M = 8;    // tramos mas cortos no reciben flecha

function _redesLinkPoints(link, burial) {
  const buried = link.buried_points || link.points;
  return link.points.map((p, idx) => {
    const b = buried[idx] || p;
    return new THREE.Vector3(p[0], lerp(p[1], b[1], burial), p[2]);
  });
}

function _buildRedTubes(targetLayer, linksData, colorHex, burial) {
  clearGroup(targetLayer);
  if (!linksData || !linksData.length) return;
  const radius = 1.2;
  const col = new THREE.Color(colorHex);
  for (const link of linksData) {
    if (!link.points || link.points.length < 2) continue;
    const pts = _redesLinkPoints(link, burial);
    const curve = new THREE.CurvePath();
    for (let i = 1; i < pts.length; i++) curve.add(new THREE.LineCurve3(pts[i-1], pts[i]));
    const geo = new THREE.TubeGeometry(curve, Math.max(pts.length*6, 16), radius, 6, false);
    const mat = new THREE.MeshStandardMaterial({
      color: col, emissive: col, emissiveIntensity: 0.25,
      roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.90,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 6;
    targetLayer.add(mesh);
  }
}

// Flechas de sentido de flujo: conos instanciados orientados src -> dst
// (los tramos ya vienen orientados por gravedad: src = cota alta, dst = cota baja)
function _buildRedArrows(targetLayer, linksData, colorHex, burial) {
  clearGroup(targetLayer);
  if (!linksData || !linksData.length) return;
  const transforms = [];
  for (const link of linksData) {
    if (!link.points || link.points.length < 2) continue;
    const pts = _redesLinkPoints(link, burial);
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + pts[i].distanceTo(pts[i - 1]));
    const L = cum[cum.length - 1];
    if (L < REDES_ARROW_MIN_LEN_M) continue;
    const nArrows = Math.max(1, Math.floor(L / REDES_ARROW_SPACING_M));
    for (let k = 0; k < nArrows; k++) {
      const dTarget = (k + 0.5) * (L / nArrows);
      let i = 1;
      while (i < cum.length - 1 && cum[i] < dTarget) i++;
      const segLen = Math.max(cum[i] - cum[i - 1], 1e-6);
      const t = (dTarget - cum[i - 1]) / segLen;
      const pos = new THREE.Vector3().lerpVectors(pts[i - 1], pts[i], t);
      const dir = new THREE.Vector3().subVectors(pts[i], pts[i - 1]).normalize();
      transforms.push([pos, dir]);
    }
  }
  if (!transforms.length) return;
  const geo  = new THREE.ConeGeometry(2.6, 7.0, 8);
  const mat  = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.95 });
  const inst = new THREE.InstancedMesh(geo, mat, transforms.length);
  const up = new THREE.Vector3(0, 1, 0);
  const q  = new THREE.Quaternion();
  const m  = new THREE.Matrix4();
  const s  = new THREE.Vector3(1, 1, 1);
  transforms.forEach(([pos, dir], i) => {
    q.setFromUnitVectors(up, dir);
    m.compose(pos, q, s);
    inst.setMatrixAt(i, m);
  });
  inst.instanceMatrix.needsUpdate = true;
  inst.renderOrder = 7;
  targetLayer.add(inst);
}

function updateRedesVisibility() {
  pluvialLinkLayer.visible   = redesState.pluvial;
  residualLinkLayer.visible  = redesState.residual;
  pluvialArrowLayer.visible  = redesState.pluvial  && redesState.arrows;
  residualArrowLayer.visible = redesState.residual && redesState.arrows;
}

function buildRedPluvialLinks() {
  const burial = typeof viewerState !== 'undefined' ? clamp(viewerState.hydraulicLinkBurial, 0, 1) : 0.0;
  _buildRedTubes(pluvialLinkLayer, RED_PLUVIAL_LINKS, 0x29b6f6, burial);
  _buildRedArrows(pluvialArrowLayer, RED_PLUVIAL_LINKS, 0x81d4fa, burial);
  updateRedesVisibility();
}

function buildRedResidualLinks() {
  const burial = typeof viewerState !== 'undefined' ? clamp(viewerState.hydraulicLinkBurial, 0, 1) : 0.0;
  _buildRedTubes(residualLinkLayer, RED_RESIDUAL_LINKS, 0xff7043, burial);
  _buildRedArrows(residualArrowLayer, RED_RESIDUAL_LINKS, 0xffab91, burial);
  updateRedesVisibility();
}
"""


def _build_events_js():
    return """
// ── Inicializar redes hidraulicas separadas ───────────────────────────────────
buildRedPluvialLinks();
buildRedResidualLinks();

// Los checkboxes pueden estar despues de este script en el documento:
// enganchar listeners solo cuando el DOM este completo.
function wireRedesUI() {
  document.getElementById('toggle-red-pluvial-links')?.addEventListener('change', e => {
    redesState.pluvial = e.target.checked;
    updateRedesVisibility();
  });
  document.getElementById('toggle-red-residual-links')?.addEventListener('change', e => {
    redesState.residual = e.target.checked;
    updateRedesVisibility();
  });
  document.getElementById('toggle-red-flow-arrows')?.addEventListener('change', e => {
    redesState.arrows = e.target.checked;
    updateRedesVisibility();
  });
  // Reactivar reconstrucción al mover el control de entierro (burial slider)
  document.getElementById('hydraulic-link-burial')?.addEventListener('input', () => {
    buildRedPluvialLinks();
    buildRedResidualLinks();
  });
}
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', wireRedesUI);
} else {
  wireRedesUI();
}
"""


def _build_panel_html():
    return """  <div class="layer-subtitle">Redes Hidráulicas</div>
  <label class="layer-row"><input id="toggle-red-pluvial-links" type="checkbox" checked>
    <span style="color:#29b6f6">&#9646;</span> Red Pluvial (CRP)</label>
  <label class="layer-row"><input id="toggle-red-residual-links" type="checkbox" checked>
    <span style="color:#ff7043">&#9646;</span> Red Residual (CRN)</label>
  <label class="layer-row"><input id="toggle-red-flow-arrows" type="checkbox" checked>
    <span style="color:#b3e5fc">&#10148;</span> Flechas de flujo</label>"""


def _build_legend_html():
    return """  <div class="leg-item"><span class="leg-swatch" style="background:#29b6f6"></span>Red Pluvial (CRP)</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#ff7043"></span>Red Residual (CRN)</div>
  <div class="leg-item"><span style="color:#b3e5fc">&#10148;</span>&nbsp;Sentido del flujo (gravedad)</div>"""


if __name__ == "__main__":
    main()
