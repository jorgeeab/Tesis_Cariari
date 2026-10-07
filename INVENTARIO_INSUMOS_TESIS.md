# Inventario de insumos de la tesis (Ciudad Cariari)

Inventario de los insumos originales necesarios para revisar y reproducir el
modelo pluvial y mejorar las figuras del TFG. Elaborado el 2026-10-06.

Convenciones:

- **Verificado**: la relación con la figura o la simulación se comprobó
  (rutas en el código, fechas de archivo, sumas MD5 o cifras citadas en el
  documento).
- **Probable**: hay indicios, pero no una comprobación directa.
- Las rutas son relativas a la raíz del repositorio.
- Ningún archivo fue renombrado, editado ni reconstruido. Los nombres con
  errores aparentes (`liteas a seguir`, `actaulizado`, `erath`) se conservan
  tal cual.

---

## 1. Insumos SIG originales: `Arcgis shapes y curvas/`

La carpeta se sube completa con su estructura, salvo las exclusiones de la
sección 4. Es idéntica al contenido de `Arcgis shapes y curvas.zip` (raíz del
proyecto, 745 MB) y a la copia en `E:\Tesis_Cariari`.

### 1.1 Capas que leen los scripts del modelo

Todas están en EPSG:5367 (CRTM05), salvo que se indique otra cosa.

| Ruta (dentro de `Arcgis shapes y curvas/`) | Contenido | Usada por | Relación |
|---|---|---|---|
| `Shapes Tragantes y alcantarillado/Alcantarillado.shp` (+ .shx .dbf .prj .cpg .qmd) | 43 líneas, trazado del alcantarillado | `exportar_redes_hidraulicas.py` (`ALCANTARILLADO_SHP`) | Verificado: red pluvial → `.inp` |
| `Shapes Tragantes y alcantarillado/liteas a seguir.shp` (+ .shx .dbf .prj .cpg) | 44 líneas guía de trazado | `exportar_redes_hidraulicas.py` (`LINEAS_SHP`) | Verificado |
| `Shapes Tragantes y alcantarillado/Tragantes (2).shp` (+ .shx .dbf .prj .cpg .sbn .sbx; metadatos en `Tragantes.shp (2).xml`) | 254 tragantes (puntos) | `exportar_redes_hidraulicas.py`, `generar_dem_3d_threejs_satelital.py`, `scripts/generar_microcuencas.py`, `scripts/generar_antes_satelital.py` | Verificado |
| `Shapes Tragantes y alcantarillado/CRP actualizado/CRP actualizado.shp` (+ .shx .dbf .prj .cpg .qmd) y su `.zip` | 277 cajas de registro pluvial | `exportar_redes_hidraulicas.py`, `generar_dem_3d_threejs_satelital.py` (`CRP_SHP`) | Verificado |
| `Shapes Tragantes y alcantarillado/CRP (2).shp` (+ .shx .dbf .prj .qmd; **sin .cpg**) | 277 cajas pluviales, versión anterior a "CRP actualizado" | Ningún script actual | Versión previa |
| `Shapes Tragantes y alcantarillado/CRN (2).shp` (+ .shx .dbf .prj .cpg .qmd) | 193 cajas de registro sanitario (red residual) | `exportar_redes_hidraulicas.py`, `generar_dem_3d_threejs_satelital.py` (`CRN_SHP`) | Verificado |
| `Shapes Tragantes y alcantarillado/Pluvial actualizado/Pluvial actualizado.shp` (+ .shx .dbf .prj .cpg .qmd) y su `.zip` | 321 tramos pluviales trazados a mano | `exportar_redes_hidraulicas.py` (`LINEAS_USUARIO_SHP`) | Verificado |
| `Desfogues.shp` (+ .shx .dbf .prj .cpg .sbn .sbx .shp.xml) | 9 desfogues al río | `exportar_redes_hidraulicas.py`, `generar_dem_3d_threejs_satelital.py`, `scripts/generar_figuras_escrito.py` | Verificado |
| `Cauce.shp` (+ .shx .prj .cpg; **el .dbf no se sube**, ver §4) | Ríos y cauces | `generar_dem_3d_threejs_satelital.py`, `scripts/generar_figuras_escrito.py`, `scripts/generar_figuras_536.py`, `scripts/generar_microcuencas.py` | Verificado; incompleto en GitHub |
| `Open Street Map & Cariari, etc.kml` | Calles y elementos OSM | `generar_dem_3d_threejs_satelital.py` (`OSM_KML`) | Verificado |
| `Zonas Verdes.kml` (y su versión comprimida `Zonas Verdes.kmz`, que no lee ningún script) | Zonas verdes | `generar_dem_3d_threejs_satelital.py` (`ZONAS_VERDES_KML`) | Verificado |
| `25 1 26, etc.kml` | Capa KML del 2026-03-03 | `generar_dem_3d_threejs_satelital.py` (`kml_specific`) | Verificado |
| `Imagen Satelital_TESIS.jpgw`, `.prj`, `.jpg.aux.xml` | Georreferencia de la imagen satelital | `generar_dem_3d_threejs_satelital.py` | La imagen `.jpg` no se sube (§4) |
| `G-08 SANITARIO OESTE.pdf` y `G-08 SANITARIO OESTE (1).pdf` (copia idéntica, mismo MD5) | Lámina G-08 del PMASB (CONCESA para la Municipalidad de Belén, enero 2011) | `scripts/generar_figuras_escrito.py` | Verificado: original de la **Figura 5** (§2) y de la Figura 28 |

### 1.2 Otros contenidos de la carpeta (no los lee ningún script actual)

| Ruta | Descripción |
|---|---|
| `Shapes Tragantes y alcantarillado/area de estudio.shp`, `Area de estudio actaulizado/` (+ `.zip`) | Polígono del área de estudio (dos versiones) |
| `Shapes Tragantes y alcantarillado/Area para SBN/` (+ `.zip`) | 16 polígonos de áreas candidatas a SbN (EPSG:4326) |
| `Shapes Tragantes y alcantarillado/Area de intervencion.zip`, `Tuberia Pluvial con Caudal.zip` | Capas comprimidas; no hay versión descomprimida en la carpeta |
| `Shapes Tragantes y alcantarillado/Shapes Tragantes y alcantarillado.rar` | Contiene solo `CRP (2).*` y `CRN (2).*`, iguales a los sueltos |
| `Shapes Tragantes y alcantarillado/Puntos CRP.qlr`, `Puntos CRP (1).qlr` | Definiciones de capa QGIS que apuntan a `Puntos CRP.shp`, que no existe (§5) |
| `Pluvial confirmado.gdb/`, `Alcantarillado no confirmado.gdb/` y sus `.lyr` | Geodatabases de ArcGIS del proyecto inicial (2024) |
| `Mapa cariari.mxd` | Proyecto ArcMap 10.5 del curso "Taller de tecnologías" (2024), con referencias a rutas de otro equipo (§5) |
| `Mapa cariari puntos de muestreo.pdf` / `.png` | Mapa de puntos de muestreo exportado |
| `Curvas 1 25000.*`, `Curvas intermedias 1 25.*` | Curvas de nivel IGN 1:25 000. **Ya venían sin `.shp`** en el original |
| `Curvas suplementarias 1 25.*` | Curvas suplementarias, completas |
| `Shapes y KML Rouse/` (`Carto 1 25/`, `Zone de estudio. .*`) | Cartografía base 1:25 000 (cobertura, cuerpos de agua, infraestructura, puntos) |

---

## 2. Originales de las figuras 4 y 5: `insumos_figuras/`

La numeración corresponde a la versión más reciente del documento
(`TFG 2026 AH-CB-JA EDECA Potada actualizada.docx`, 2026-08-05). En
`TFG 2026 AH-JA-CB FINAL.docx` las mismas figuras aparecen como 3 y 4. Las
imágenes se extrajeron del `.docx` sin modificarlo.

### Figura 4: construcciones antes y después del año 2000

| Ruta | Descripción | Estado |
|---|---|---|
| `insumos_figuras/figura_04_construcciones_antes_despues_2000/image4_extraida_de_TFG.png` | Imagen incrustada en el TFG (`word/media/image4.png`, 894 × 688 px, MD5 `521f603f…`). Mapa de la Unidad de Topografía, Municipalidad de Belén, escala 1:4 500, fechado 11/3/2021, con las capas *PermisosAntes2000*, *PermisosDespues2000* y *MapaPredial*. | **Falta el original.** La misma imagen de baja resolución ya aparecía en `Desktop/Bosques de Doña Rosa/Proyecto Ciudad Cariari.docx` (noviembre de 2024). No hay en este equipo ningún PDF, proyecto SIG ni capa de permisos. |

### Figura 5: plan maestro de recolección, tratamiento y disposición de aguas residuales

| Ruta | Descripción | Estado |
|---|---|---|
| `Arcgis shapes y curvas/G-08 SANITARIO OESTE.pdf` | Lámina G-08 "Sector Oeste" del PMASB, vectorial, 1 página | **Verificado visualmente:** la figura del TFG es un recorte de esta lámina (misma simbología, bloques, colectores CRC4/CRC5 y recuadro de ubicación) |
| `insumos_figuras/figura_05_plan_maestro_aguas_residuales/image5_extraida_de_TFG.png` | Imagen incrustada en el TFG (`word/media/image5.png`, 1580 × 700 px) | Referencia para reproducir el recorte |
| `insumos_figuras/figura_05_plan_maestro_aguas_residuales/Plan Maestro Alcantarillado Sanitario.pdf` | Presentación "PMAS, División Estratégica…" de la Municipalidad de Belén (30 págs., 2018). Copiada de `Desktop/Tesis_Belen/` (mismo MD5). | Respaldo del texto (Cuenca A: Cariari, La Asunción, Los Arcos). No contiene la lámina G-08. |

---

## 3. Simulación SWMM / PySWMM

### 3.1 Cadena de archivos

```
Arcgis shapes y curvas/  (§1.1)
  └─ exportar_redes_hidraulicas.py ──► output/shapefiles_v7/red_pluvial_{nodos,links}.shp
       └─ scripts/generar_inp_pluvial.py ──► models/cariari_pluvial.inp
generar_dem_3d_threejs_satelital.py ──► results/hydrology/surface_grid.json  (grid D8 del MDE)
datos/lluvia/era5_cariari_horario_2000_2026.csv  +  IDF Santa Lucía (constante en el script)
  └─ scripts/simular_evento_real.py  (swmm_integration/ + PySWMM)
       ├─► results/simulaciones/diseno_T10_santa_lucia/{hietograma.csv, resumen.json}
       └─► results/simulaciones/evento_real_2016-06-14/{hietograma.csv, resumen.json}
```

### 3.2 Archivos incluidos

| Ruta | Descripción | Fecha local | Relación con la tesis |
|---|---|---|---|
| `scripts/simular_evento_real.py` | Ejecuta ambos escenarios con PySWMM. Contiene la tabla IDF de la estación 84-111 Santa Lucía (IMN, Cuadro 17) y lee el CSV de ERA5. | 2026-07-18 16:44 | Verificado |
| `swmm_integration/` (ya estaba versionado) | Motor PySWMM, hidrogramas, grid y esquemas | 2026-07-07 | Usado por el script anterior |
| `scripts/generar_inp_pluvial.py` (ya estaba versionado) | Genera el `.inp` desde `output/shapefiles_v7` | 2026-07-11 | Verificado por rutas |
| `exportar_redes_hidraulicas.py` (**cambios locales que no estaban en git**) | Construye la red pluvial y residual desde los shapefiles originales | 2026-07-16 09:58 | Es la versión que generó `output/shapefiles_v7` (10:11:34) y el `.inp` (10:11:36). Se sube tal como estaba en el disco. |
| `output/shapefiles_v7/*` (cambios locales) | Red pluvial y residual derivada | 2026-07-16 10:11 | Entrada del `.inp` |
| `models/cariari_pluvial.inp` (cambios locales) | 404 nodos, 19 descargas, 418 conductos; DYNWAVE, CMS, ventana de 4 h | 2026-07-16 10:11 | Es el `.inp` indicado en ambos `resumen.json`. La versión anterior en git (commit `ebae015`) tenía 378/28/400. |
| `models/cariari_pluvial.rpt`, `models/cariari_pluvial.out` (cambios locales) | Reporte y binario SWMM 5.2.4 | 2026-07-18 17:17 | **Versión sin identificar** (ver §3.3) |
| `results/hydrology/surface_grid.json` (cambio local de 1 línea) | Grid hidrológico de superficie | 2026-07-18 14:04 | Entrada de la simulación (anterior a las corridas de 16:45 y 16:48) |
| `results/swmm/ui_scenario.json` (ya estaba versionado) | Parámetros de infiltración y captación | 2026-07-07 | Lo carga `simular_evento_real.py` |
| `results/simulaciones/diseno_T10_santa_lucia/hietograma.csv`, `resumen.json` | Tormenta de diseño T = 10 años, 60 min, bloques alternos: 82,9 mm, pico de 245 mm/h | 2026-07-18 16:45 | **Verificado:** 63 373 m³ inundados, 57 nodos y 127 259 m³ captados aparecen en las versiones FINAL, "Final para revision" y "EDECA Potada actualizada" |
| `results/simulaciones/evento_real_2016-06-14/hietograma.csv`, `resumen.json` | Evento ERA5 del 14-06-2016 (10:00 a 24:00): 120,2 mm, pico de 23,5 mm/h | 2026-07-18 16:48 | **Verificado:** 120,2 mm y 184 606 m³ aparecen en las mismas versiones |
| `datos/lluvia/era5_cariari_horario_2000_2026.csv` | Serie horaria ERA5 (Open-Meteo), celda 9.947 N, −84.173 O; incluye las 24 h del 2016-06-14 | 2026-07-18 16:35 | Verificado |
| `datos/lluvia/analisis_lluvia.md` | Fuentes, descarga y valores adoptados | 2026-07-18 | Documentación |
| `datos/lluvia/referencias/IMN_IDF_estaciones_mecanicas.pdf`, `IMN_IDF_estaciones_automaticas.pdf` | Curvas IDF del IMN (la de Santa Lucía está en el Cuadro 17 del PDF de estaciones mecánicas) | 2026-07-18 | Fuente de la tormenta de diseño |
| `datos/infiltracion/tabla_infiltracion.csv`, `.md` | Tabla de infiltración por tipo de superficie | 2026-07-18 | Probable (documentación de parámetros) |

### 3.3 Versiones y advertencias

- **`.rpt` y `.out`:** PySWMM abre el `.inp` en su ubicación y sobrescribe
  `models/cariari_pluvial.rpt` y `.out` en cada corrida. Los archivos actuales
  son del 2026-07-18 17:17, con un aporte externo de 43 462 m³. Ese valor no
  coincide con ninguno de los escenarios de la tesis (127 259 y 184 606 m³),
  así que corresponden a una corrida posterior que no se pudo identificar
  (probablemente desde el visor). **Los resultados de la tesis están en
  `results/simulaciones/*/resumen.json`, no en el `.rpt` ni en el `.out`.**
- La versión de `.inp`, `.rpt` y `.out` del commit `ebae015` (2026-07-14,
  corrida del 2026-07-10) es anterior y no corresponde a la tesis.
- `models/cariari_base.*` es el modelo mínimo de prueba del MVP. No es de la
  tesis.
- No hay registros de ejecución (`*.log`). Las salidas de consola de las
  corridas no se guardaron.
- No se ejecutó ninguna simulación nueva para elaborar este inventario.

---

## 4. Archivos que existen localmente pero no se suben

| Archivo | Tamaño | Motivo |
|---|---|---|
| `Arcgis shapes y curvas/Cauce.dbf` | 145,9 MB | Supera el límite de 100 MB de GitHub. Sin él, `Cauce.shp` no se puede abrir con atributos. |
| `Arcgis shapes y curvas/Curvas intermedias 1 25.dbf` | 108,4 MB | Supera el límite de 100 MB |
| `Arcgis shapes y curvas/Imagen Satelital_TESIS.jpg` | 397,8 MB | Supera el límite de 100 MB (lo usa `generar_dem_3d_threejs_satelital.py`) |
| `Arcgis shapes y curvas/Imagen Satelital_TESIS.jpg.ovr` | 255,8 MB | Supera el límite de 100 MB (pirámides de la imagen) |
| `Arcgis shapes y curvas.zip` (raíz) | 745,7 MB | Supera el límite; duplica la carpeta |
| `Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/WhatsApp Unknown 2026-07-14 at 8.52.30 AM/` y su `.zip` | 1,3 MB | Fotos de una vivienda particular; no son insumos del modelo |
| `resultados_refinados/_cache_*` | ≈ 390 MB | Cachés regenerables (WFS IGN 1:25k del SNIT, OSM, teselas, catastro). `_cache_curvas_raw/intermedia.geojson` pesa 291 MB. |

Los archivos grandes se pueden obtener de `Arcgis shapes y curvas.zip` o de
`E:\Tesis_Cariari`. La imagen satelital también está en
`Desktop/Tesis_Belen/Imagen Satelital_TESIS.jpg`. Si se necesitan en el
repositorio, la opción es Git LFS.

---

## 5. Insumos faltantes o con versión no identificada

1. **Original de la Figura 4** (mapa de permisos de construcción antes y
   después del 2000, Unidad de Topografía de la Municipalidad de Belén,
   11/3/2021). Solo existe la imagen de 894 × 688 px. Hay que pedirla a la
   Municipalidad (PDF o capa *PermisosAntes2000* / *PermisosDespues2000* /
   *MapaPredial*).
2. **`CRP.shp`, `CRN_CAPA.shp`, `Tragantes.shp`** (nombres antiguos citados
   en un comentario de `exportar_redes_hidraulicas.py`). No existen en el
   equipo, en el zip ni en el `.rar`. El código actual usa
   `CRP actualizado/CRP actualizado.shp`, `CRN (2).shp` y `Tragantes (2).shp`.
3. **`Puntos CRP.shp`**, referenciado por `Puntos CRP.qlr`.
4. Capas referenciadas por `Mapa cariari.mxd` que estaban en otro equipo:
   `Puntos de muestreo .gdb`, `CRP.gdb`, `OPuntos de muestreo.gdb`,
   `Google erath data/{Alcantarillado no confirmado, CRP, Pluvial confirmado}.kmz`
   y `Pluvial Cariari..kml`.
5. **`.shp` de `Curvas 1 25000` y `Curvas intermedias 1 25`**: ya faltaban
   en la carpeta original. Solo hay `.dbf`, `.shx`, `.prj` y `.cpg`.
6. **`CRP (2).cpg`**: no existe; el resto del shapefile está completo.
7. **`.rpt` y `.out` de las corridas citadas en la tesis**: se sobrescribieron
   (§3.3). Para recuperarlos habría que volver a ejecutar
   `scripts/simular_evento_real.py`, lo que no se hizo.
8. **Registros de ejecución** de las simulaciones: no existen.
9. **Lámina G-09 "Sector Este"** del PMASB: citada en el recuadro de la G-08,
   pero no está en el equipo. La Figura 5 no la necesita.

---

## 6. Cómo localizar los archivos de entrada desde los scripts

- Todos los scripts resuelven las rutas relativas a la **raíz del
  repositorio**. Los de `scripts/` usan
  `BASE_DIR = Path(__file__).resolve().parent.parent`;
  `generar_dem_3d_threejs_satelital.py` usa rutas relativas al directorio de
  trabajo, así que debe ejecutarse desde la raíz.
- Las constantes de ruta están al inicio de cada script:
  - `exportar_redes_hidraulicas.py`: líneas 34-44 (`*_SHP`).
  - `generar_dem_3d_threejs_satelital.py`: líneas 134-151 (`OUT_DIR`,
    `CACHE_*`, `CAUCE_SHP`, `OSM_KML`, `TRAGANTES_SHP`, …) y la imagen
    satelital en la línea 1504.
  - `scripts/generar_inp_pluvial.py`: `NODOS_SHP`, `LINKS_SHP`, `OUT_INP`.
  - `scripts/simular_evento_real.py`: `ERA5_CSV`, `SURFACE_GRID_PATH`,
    `UI_SCENARIO_PATH`, `OUT_BASE`, `IDF_SANTA_LUCIA`.
  - `scripts/generar_figuras_escrito.py`, `scripts/generar_figuras_536.py`,
    `scripts/generar_microcuencas.py`, `scripts/generar_antes_satelital.py`:
    variable `shp_dir` / `SHP`, bajo `BASE / "Arcgis shapes y curvas"`.
- Para reproducir con los archivos de §4, cópielos desde
  `Arcgis shapes y curvas.zip` a la misma ruta relativa dentro de
  `Arcgis shapes y curvas/`.
