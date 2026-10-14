# Scripts de las figuras corregidas

Esta carpeta contiene los generadores citados en los registros JSON de
`figuras_corregidas/registros/`. Todos resuelven las rutas desde la raiz del
repositorio, por lo que no dependen de la ubicacion del clon.

## Preparacion

Desde la raiz del repositorio:

```bash
python -m venv .venv
python -m pip install -r figuras_corregidas/scripts/requirements.txt
```

En Windows, use `.venv/Scripts/python.exe` en los comandos siguientes; en
Linux o macOS, use `.venv/bin/python`.

## Regeneracion

```bash
.venv/Scripts/python.exe figuras_corregidas/scripts/figura_40_hietogramas.py
.venv/Scripts/python.exe figuras_corregidas/scripts/figuras_17_21_calidad_agua.py
.venv/Scripts/python.exe figuras_corregidas/scripts/mapas_15_33_48.py
.venv/Scripts/python.exe scripts/generar_microcuencas.py
```

Los tres primeros comandos escriben PNG a 300 ppp y PDF vectorial dentro de
`figuras_corregidas/`. El ultimo reconstruye las capas SIG y el mapa de control
de `output/microcuencas/`; debe ejecutarse antes de `mapas_15_33_48.py` si se
parte de un clon sin esas salidas.

Para verificar sin reemplazar las figuras de entrega, todos los generadores de
esta carpeta aceptan `--output-root RUTA`.

## Fuentes

- Figura 40: hietogramas y resumenes de `results/simulaciones/`.
- Figuras 17 a 21: valores transcritos de las tablas 14, 15 y 16 de la tesis;
  quedan declarados en `figuras_17_21_calidad_agua.py`.
- Figuras 15, 33 y 48: capas SIG del repositorio y
  `output/microcuencas/microcuencas_desfogue.shp`.
