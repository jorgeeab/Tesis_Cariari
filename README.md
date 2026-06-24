# Tesis_Cariari

Codigo para realizar mapas tecnicos sobre el comportamiento del drenaje pluvial.

## Guia SBNS con FastAPI

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar local:

```bash
uvicorn app:app --reload
```

Despliegue con Gunicorn:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app:app
```

Ruta principal:

```text
/
```

La aplicacion redirige a:

```text
/resultados_refinados/Guia/index.html
```

## MVP PySWMM

La integracion SWMM mantiene separado el modelo superficial del modelo hidraulico:

- Python calcula lluvia, infiltracion, escorrentia, rutas D8, areas tributarias e hidrogramas por tragante.
- SWMM no recalcula escorrentia. El INP base no debe incluir `[SUBCATCHMENTS]`.
- PySWMM recibe los hidrogramas con `node.generated_inflow(q_m3s)` y resuelve niveles, caudales e inundacion de nodos/enlaces.

Dependencia principal:

```bash
pip install "pyswmm[swmm5.2.4]"
```

Modelo base:

```text
models/cariari_base.inp
```

El archivo es intencionalmente minimo y usa nodos `T1`, `T2`, `J1`, `O1`.
Los nodos `T1` y `T2` son puntos de entrada para hidrogramas externos.

Endpoints principales:

```text
POST /api/scenarios
POST /api/hydrology/run
POST /api/swmm/validate
POST /api/swmm/run
GET  /api/swmm/results/{simulation_id}
GET  /api/swmm/results/{simulation_id}/nodes
GET  /api/swmm/results/{simulation_id}/links
GET  /api/swmm/results/{simulation_id}/summary
```

Ejemplo minimo de hidrologia:

```json
{
  "rainfall": {
    "scenario_name": "tormenta-base",
    "start_time": "2026-01-01T00:00:00",
    "duration_min": 30,
    "interval_min": 5,
    "intensity_mm_h": 60
  },
  "surface": {
    "nx": 2,
    "ny": 1,
    "cell_area_m2": 100,
    "receivers": [1, -1],
    "coverage": ["base", "base"],
    "sinks": [{"id": "T1", "cell_index": 1}]
  },
  "infiltration": {
    "base": 0.22,
    "vegetation": 0.9,
    "building": 0.02,
    "street": 0.06
  }
}
```

Ejemplo de corrida SWMM usando el `hydrology_id` devuelto:

```json
{
  "inp_path": "models/cariari_base.inp",
  "hydrology_id": "uuid-devuelto-por-hydrology",
  "node_mapping": {"T1": "T1"},
  "report_step_min": 5
}
```

Limitaciones del MVP:

- El almacen de escenarios/resultados es en memoria.
- El INP base es esquematico; longitudes, cotas y diametros deben calibrarse con datos de campo.
- El acople de hidrogramas usa retencion por tramo constante: cada caudal queda activo hasta el siguiente punto.
- Si PySWMM o EPA SWMM no estan instalados, `/api/swmm/run` responde `503` con la instruccion de instalacion.

## Convertir TIFF a MBTiles

El script `scripts/tiff_to_mbtiles.py` convierte un TIFF/GeoTIFF georreferenciado
a MBTiles raster usando GDAL o Rasterio.

Requisito externo: tener GDAL instalado o tener `rasterio` disponible en Python.
En Windows normalmente sirve instalar QGIS u OSGeo4W y agregar la carpeta `bin`
de GDAL al `PATH`. Si no existe `gdal_translate`, el script intenta usar
`rasterio` automaticamente.

Uso basico:

```bash
python scripts/tiff_to_mbtiles.py ruta/al/archivo.tif
```

Salida con ruta especifica:

```bash
python scripts/tiff_to_mbtiles.py ruta/al/archivo.tif -o resultados_refinados/archivo.mbtiles
```

Si GDAL no esta en el `PATH`, indique la carpeta `bin`:

```bash
python scripts/tiff_to_mbtiles.py ruta/al/archivo.tif --gdal-bin "C:\OSGeo4W\bin"
```

Para archivos mas livianos, puede usar JPEG:

```bash
python scripts/tiff_to_mbtiles.py ruta/al/archivo.tif --tile-format JPEG --quality 85 --overwrite
```

Para forzar un motor especifico:

```bash
python scripts/tiff_to_mbtiles.py ruta/al/archivo.tif --backend rasterio
```
