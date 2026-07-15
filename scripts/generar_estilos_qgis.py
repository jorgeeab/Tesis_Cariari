"""Genera los estilos .qml de flechas de flujo para QGIS y verifica renderizando.

Se ejecuta con el Python de QGIS (no con el .venv):
  "D:\\Program Files\\QGIS 4.0.1\\bin\\python-qgis.bat" scripts/generar_estilos_qgis.py

Produce:
  styles/qgis/red_*_links.qml    linea + flecha central rotada con el tramo
  styles/qgis/red_*_flechas.qml  triangulo rotado por el campo `angle`
  (copias sidecar junto a los shapefiles en output/shapefiles_v6)
  output/qgis/flechas_preview.png  render de verificacion

Los .qml se generan con la API de QGIS para que el formato coincida
exactamente con la version instalada.
"""

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (  # noqa: E402
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsLineSymbol,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsMarkerLineSymbolLayer,
    QgsMarkerSymbol,
    QgsPalLayerSettings,
    QgsProperty,
    QgsRendererCategory,
    QgsSimpleMarkerSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsSymbolLayer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QSize, Qt  # noqa: E402
from qgis.PyQt.QtGui import QColor, QFont  # noqa: E402

BASE      = Path(__file__).resolve().parents[1]
SHP_DIR   = BASE / "output" / "shapefiles_v7"
STYLE_DIR = BASE / "styles" / "qgis"
OUT_PNG   = BASE / "output" / "qgis" / "flechas_preview.png"

# (nombre, color linea, color flecha)
REDES = [
    ("red_pluvial",  QColor(41, 182, 246), QColor(2, 119, 189)),    # celeste / azul
    ("red_residual", QColor(255, 112, 67), QColor(191, 54, 12)),    # naranja / ladrillo
]


def _triangle(size_mm, color, angle=0.0):
    sm = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Triangle, size_mm)
    sm.setColor(color)
    sm.setStrokeStyle(Qt.PenStyle.NoPen)
    sm.setAngle(angle)
    return sm


def style_links(layer, line_color, arrow_color):
    """Linea continua + flecha central rotada con la direccion del tramo."""
    sym = QgsLineSymbol.createSimple({
        "line_color": f"{line_color.red()},{line_color.green()},{line_color.blue()},255",
        "line_width": "0.66",
        "capstyle": "round",
        "joinstyle": "round",
    })
    ml = QgsMarkerLineSymbolLayer()
    ml.setRotateMarker(True)
    ml.setPlacements(Qgis.MarkerLinePlacements(Qgis.MarkerLinePlacement.CentralPoint))
    marker = QgsMarkerSymbol()
    # Triangulo apunta al norte con angulo 0; sobre una marker line rotada
    # se necesita +90 para que apunte en el sentido de digitalizacion (src->dst)
    marker.changeSymbolLayer(0, _triangle(3.2, arrow_color, angle=90.0))
    ml.setSubSymbol(marker)
    sym.appendSymbolLayer(ml)
    layer.setRenderer(QgsSingleSymbolRenderer(sym))


def style_flechas(layer, arrow_color):
    """Triangulo rotado por el campo `angle` (rumbo horario desde el norte)."""
    sm = _triangle(3.4, arrow_color)
    sm.setDataDefinedProperty(QgsSymbolLayer.Property.Angle, QgsProperty.fromField("angle"))
    marker = QgsMarkerSymbol()
    marker.changeSymbolLayer(0, sm)
    layer.setRenderer(QgsSingleSymbolRenderer(marker))


def _circulo(color, size_mm=2.0):
    sm = QgsSimpleMarkerSymbolLayer(Qgis.MarkerShape.Circle, size_mm)
    sm.setColor(color)
    sm.setStrokeColor(QColor(255, 255, 255))
    marker = QgsMarkerSymbol()
    marker.changeSymbolLayer(0, sm)
    return marker


def _etiqueta_id(layer, color):
    fmt = QgsTextFormat()
    fmt.setFont(QFont("Arial"))
    fmt.setSize(8.5)
    fmt.setColor(color.darker(120))
    buf = QgsTextBufferSettings()
    buf.setEnabled(True)
    buf.setSize(0.9)
    buf.setColor(QColor(255, 255, 255))
    fmt.setBuffer(buf)
    pal = QgsPalLayerSettings()
    pal.fieldName = "id"
    pal.setFormat(fmt)
    layer.setLabeling(QgsVectorLayerSimpleLabeling(pal))
    layer.setLabelsEnabled(True)


def style_puntos_ids(layer, color, size_mm=2.0):
    """Punto pequeño + etiqueta con el id (para identificar cajas en capturas)."""
    layer.setRenderer(QgsSingleSymbolRenderer(_circulo(color, size_mm)))
    _etiqueta_id(layer, color)


# Color por tipo de caja para las capas de nodos de red
KIND_COLORS = {
    "crp":      QColor(2, 119, 189),    # azul
    "crn":      QColor(191, 54, 12),    # ladrillo
    "tragante": QColor(46, 125, 50),    # verde
    "desfogue": QColor(106, 27, 154),   # violeta
}


def style_nodos_red(layer):
    """Nodos de red (crp/tragante/desfogue o crn): color por tipo + id."""
    cats = [QgsRendererCategory(kind, _circulo(col, 1.9), kind)
            for kind, col in KIND_COLORS.items()]
    layer.setRenderer(QgsCategorizedSymbolRenderer("kind", cats))
    _etiqueta_id(layer, QColor(60, 60, 60))


def save_style(layer, qml_path):
    msg, ok = layer.saveNamedStyle(str(qml_path))
    if not ok:
        raise RuntimeError(f"No se pudo guardar {qml_path.name}: {msg}")
    print(f"  -> {qml_path.relative_to(BASE)}")


def main():
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        render_layers = []
        for red, line_color, arrow_color in REDES:
            links = QgsVectorLayer(str(SHP_DIR / f"{red}_links.shp"), f"{red}_links", "ogr")
            flechas = QgsVectorLayer(str(SHP_DIR / f"{red}_flechas.shp"), f"{red}_flechas", "ogr")
            if not links.isValid() or not flechas.isValid():
                raise RuntimeError(f"Capa invalida para {red} (¿existe el shapefile?)")

            style_links(links, line_color, arrow_color)
            style_flechas(flechas, arrow_color)
            save_style(links, STYLE_DIR / f"{red}_links.qml")
            save_style(flechas, STYLE_DIR / f"{red}_flechas.qml")
            # Sidecars: QGIS los aplica automaticamente al cargar la capa
            for name in (f"{red}_links.qml", f"{red}_flechas.qml"):
                shutil.copyfile(STYLE_DIR / name, SHP_DIR / name)
            render_layers.extend([flechas, links])
            if red == "red_pluvial":
                pluv_links_layer = links

        # ── Capas de puntos con id etiquetado ────────────────────────────────
        for name, color in [("crp_puntos_ids",       KIND_COLORS["crp"]),
                            ("crn_puntos_ids",       KIND_COLORS["crn"]),
                            ("tragantes_puntos_ids", KIND_COLORS["tragante"]),
                            ("desfogues_puntos_ids", KIND_COLORS["desfogue"])]:
            shp = SHP_DIR / f"{name}.shp"
            if not shp.exists():
                print(f"  AVISO: no existe {shp.name}")
                continue
            capa = QgsVectorLayer(str(shp), name, "ogr")
            if not capa.isValid():
                continue
            style_puntos_ids(capa, color, size_mm=1.7 if "tragante" in name else 2.0)
            save_style(capa, STYLE_DIR / f"{name}.qml")
            shutil.copyfile(STYLE_DIR / f"{name}.qml", SHP_DIR / f"{name}.qml")
            render_layers.insert(0, capa)

        # ── Nodos de red: color por tipo de caja + etiqueta id ───────────────
        for name in ("red_pluvial_nodos", "red_residual_nodos"):
            shp = SHP_DIR / f"{name}.shp"
            if not shp.exists():
                continue
            capa = QgsVectorLayer(str(shp), name, "ogr")
            if not capa.isValid():
                continue
            style_nodos_red(capa)
            save_style(capa, STYLE_DIR / f"{name}.qml")
            shutil.copyfile(STYLE_DIR / f"{name}.qml", SHP_DIR / f"{name}.qml")

        # ── Render de verificacion: zoom al tramo pluvial mas largo ──────────
        longest = max(pluv_links_layer.getFeatures(), key=lambda f: f["dist_m"] or 0)
        extent = longest.geometry().boundingBox()
        extent.grow(80)

        settings = QgsMapSettings()
        settings.setLayers(render_layers)
        settings.setExtent(extent)
        settings.setOutputSize(QSize(1400, 900))
        settings.setBackgroundColor(QColor(248, 248, 245))
        settings.setOutputDpi(96)

        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        job.renderedImage().save(str(OUT_PNG))
        print(f"  -> {OUT_PNG.relative_to(BASE)} (verificacion)")
    finally:
        qgs.exitQgis()
    print("Listo.")


if __name__ == "__main__":
    sys.exit(main() or 0)
