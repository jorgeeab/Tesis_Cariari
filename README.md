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
