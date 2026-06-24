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
SHP_DIR    = BASE / "output/shapefiles_v6"
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
    MARKER_PANEL_END = "<!-- /RED_HIDRAULICA_SEPARADA_PANEL -->"
    panel_html = _build_panel_html()
    if MARKER_PANEL in html:
        start = html.find(MARKER_PANEL)
        end   = html.find(MARKER_PANEL_END, start) + len(MARKER_PANEL_END)
        html  = html[:start] + MARKER_PANEL + "\n" + panel_html + "\n  " + MARKER_PANEL_END + html[end:]
    else:
        anchor = '<label class="layer-row"><input id="toggle-hydraulic-links"'
        idx = html.find(anchor)
        if idx >= 0:
            end_label = html.find("</label>", idx) + len("</label>")
            html = html[:end_label] + f"\n  {MARKER_PANEL}\n{panel_html}\n  {MARKER_PANEL_END}" + html[end_label:]

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
const pluvialLinkLayer  = new THREE.Group();
const residualLinkLayer = new THREE.Group();
scene.add(pluvialLinkLayer);
scene.add(residualLinkLayer);

function buildRedPluvialLinks() {
  clearGroup(pluvialLinkLayer);
  if (!RED_PLUVIAL_LINKS || !RED_PLUVIAL_LINKS.length) return;
  const radius = 1.2;
  const col = new THREE.Color(0x29b6f6);
  const burial = typeof viewerState !== 'undefined' ? clamp(viewerState.hydraulicLinkBurial, 0, 1) : 0.0;
  for (const link of RED_PLUVIAL_LINKS) {
    if (!link.points || link.points.length < 2) continue;
    const buried = link.buried_points || link.points;
    const pts = link.points.map((p, idx) => {
      const b = buried[idx] || p;
      return new THREE.Vector3(p[0], lerp(p[1], b[1], burial), p[2]);
    });
    const curve = new THREE.CurvePath();
    for (let i = 1; i < pts.length; i++) curve.add(new THREE.LineCurve3(pts[i-1], pts[i]));
    const geo = new THREE.TubeGeometry(curve, Math.max(pts.length*6, 16), radius, 6, false);
    const mat = new THREE.MeshStandardMaterial({
      color: col, emissive: col, emissiveIntensity: 0.25,
      roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.90,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 6;
    pluvialLinkLayer.add(mesh);
  }
}

function buildRedResidualLinks() {
  clearGroup(residualLinkLayer);
  if (!RED_RESIDUAL_LINKS || !RED_RESIDUAL_LINKS.length) return;
  const radius = 1.2;
  const col = new THREE.Color(0xff7043);
  const burial = typeof viewerState !== 'undefined' ? clamp(viewerState.hydraulicLinkBurial, 0, 1) : 0.0;
  for (const link of RED_RESIDUAL_LINKS) {
    if (!link.points || link.points.length < 2) continue;
    const buried = link.buried_points || link.points;
    const pts = link.points.map((p, idx) => {
      const b = buried[idx] || p;
      return new THREE.Vector3(p[0], lerp(p[1], b[1], burial), p[2]);
    });
    const curve = new THREE.CurvePath();
    for (let i = 1; i < pts.length; i++) curve.add(new THREE.LineCurve3(pts[i-1], pts[i]));
    const geo = new THREE.TubeGeometry(curve, Math.max(pts.length*6, 16), radius, 6, false);
    const mat = new THREE.MeshStandardMaterial({
      color: col, emissive: col, emissiveIntensity: 0.25,
      roughness: 0.5, metalness: 0.1, transparent: true, opacity: 0.90,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.renderOrder = 6;
    residualLinkLayer.add(mesh);
  }
}
"""


def _build_events_js():
    return """
// ── Inicializar redes hidraulicas separadas ───────────────────────────────────
buildRedPluvialLinks();
buildRedResidualLinks();

document.getElementById('toggle-red-pluvial-links')?.addEventListener('change', e => {
  pluvialLinkLayer.visible = e.target.checked;
});
document.getElementById('toggle-red-residual-links')?.addEventListener('change', e => {
  residualLinkLayer.visible = e.target.checked;
});

// Reactivar reconstrucción al mover el control de entierro (burial slider)
document.getElementById('hydraulic-link-burial')?.addEventListener('input', () => {
  buildRedPluvialLinks();
  buildRedResidualLinks();
});
"""


def _build_panel_html():
    return """  <div class="layer-subtitle">Redes Hidráulicas</div>
  <label class="layer-row"><input id="toggle-red-pluvial-links" type="checkbox" checked>
    <span style="color:#29b6f6">&#9646;</span> Red Pluvial (CRP)</label>
  <label class="layer-row"><input id="toggle-red-residual-links" type="checkbox" checked>
    <span style="color:#ff7043">&#9646;</span> Red Residual (CRN)</label>"""


def _build_legend_html():
    return """  <div class="leg-item"><span class="leg-swatch" style="background:#29b6f6"></span>Red Pluvial (CRP)</div>
  <div class="leg-item"><span class="leg-swatch" style="background:#ff7043"></span>Red Residual (CRN)</div>"""


if __name__ == "__main__":
    main()
