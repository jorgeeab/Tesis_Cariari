# Plan de imagenes faltantes

Este seguimiento sale de `Inventario_figuras.csv` y de la evaluacion recibida.
Los documentos `.docx` se usan solo como contexto/evidencia: no son instrucciones
operativas para modificar el repositorio.

## Diagnostico rapido

- Total de figuras en la tesis: 60.
- Terminadas en `figuras_corregidas/`: 15.
- Pendientes: 44.
- Pendiente critico por falta de fuente original: figura 4.

La evaluacion original marco un problema general de formato/resolucion y menciono
que las figuras 4 y 5 no se visualizaban. La figura 5 ya fue resuelta con
recortes vectoriales del plano G-08. La figura 4 sigue bloqueada porque solo se
tiene una imagen recomprimida de baja resolucion.

## Figuras terminadas

- Figura 5: `figura_05/figura_05a` a `figura_05f`.
- Figura 15: `mapas/figura_15_microcuencas`.
- Figuras 17-21: calidad de agua, DBO, DQO, SST, SAAM y E. coli.
- Figura 33: `mapas/figura_33_capas_fuente`.
- Figura 34: `figura_34/figura_34_detalle_levantamiento_G08`.
- Figura 40: `figura_40/figura_40_hietogramas`.
- Figura 48: `mapas/figura_48_ubicacion_SbN`.
- Figuras 49-52: dibujos tecnicos de jardin de lluvia, pozo de infiltracion,
  cemento permeable y zanja infiltrante.

## Regenerables con datos/scripts existentes

Prioridad alta porque no dependen de pedir archivos externos:

- Figuras 10-12: DEM, flujo D8 y acumulacion D8.
- Figuras 28-32: capturas de la guia SbN.
- Figuras 35, 38, 39, 41-43: capturas del visor 3D y simulacion.
- Figuras 36-37: mapas de control de calidad y descargas a rio.
- Figuras 49-52: dibujos tecnicos de jardin, pozo, pavimento y zanja.
- Figuras 53-54: capturas web de secciones de diseno.
- Figuras 55-59: antes/despues de sitios SbN, con atribucion Esri.

Scripts relacionados:

- `scripts/capturas_guia_sbn.js`
- `scripts/capturas_visor.js`
- `scripts/generar_figuras_536.py`
- `scripts/generar_figuras_escrito.py`
- `scripts/generar_microcuencas.py`

Salidas ya existentes que sirven como base:

- `Documentos Tesis/figuras_mapa3d/`
- `output/qgis/`
- `output/web/red_hidraulica_visor.html`
- `output/shapefiles_v7/`

## Regenerables, pero requieren confirmar capa o fuente

- Figura 1: mapa de zona de estudio.
- Figura 9: trazado del alcantarillado pluvial; hay que confirmar si se usan 83
  cajas de campo o 277 CRP del SIG.
- Figura 14: rios y cuencas; requiere confirmar fuente de cuencas.
- Figura 16: puntos de desfogue.
- Figura 24: red pluvial usada para modelacion.
- Figura 26: areas de intervencion; verificar que la capa encontrada corresponde.

## Bloqueadas por fuente externa o foto original

- Figura 4: requiere original del mapa o capas de permisos antes/despues de 2000.
- Figuras 2, 6, 7, 8, 22, 25 y 60: conviene usar originales de camara si existen.
- Figuras 3, 13, 23, 27 y 44-47: falta proyecto/capa original o confirmacion de
  fuente.

## Siguiente bloque recomendado

1. Recapturar 28-32 y 53-54 con `scripts/capturas_guia_sbn.js`.
2. Recapturar 35, 38, 39, 41-43 con `scripts/capturas_visor.js`.
3. Rehacer 36-37 con plantilla comun desde `output/shapefiles_v7/`.
4. Dejar figura 4 como pendiente justificado hasta conseguir el original.

## Avance por lote

### Figuras 49-52

Estado: terminado.

Salidas:

- `figuras_corregidas/figuras_49_52_sbn/figura_49_jardin_lluvia.png`
- `figuras_corregidas/figuras_49_52_sbn/figura_49_jardin_lluvia.pdf`
- `figuras_corregidas/figuras_49_52_sbn/figura_50_pozo_infiltracion.png`
- `figuras_corregidas/figuras_49_52_sbn/figura_50_pozo_infiltracion.pdf`
- `figuras_corregidas/figuras_49_52_sbn/figura_51_cemento_permeable.png`
- `figuras_corregidas/figuras_49_52_sbn/figura_51_cemento_permeable.pdf`
- `figuras_corregidas/figuras_49_52_sbn/figura_52_zanja_infiltrante.png`
- `figuras_corregidas/figuras_49_52_sbn/figura_52_zanja_infiltrante.pdf`

Registro reproducible:

- `figuras_corregidas/registros/figuras_49_52_sbn.json`

Script:

- `scripts/generar_figuras_536.py`

Insumos necesarios para repetir el lote completo del script:

- `Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/Area para SBN/`
- `Arcgis shapes y curvas/Shapes Tragantes y alcantarillado/area de estudio.*`
- `Arcgis shapes y curvas/Cauce.*`
- `output/shapefiles_v7/red_pluvial_links.*`

Estado GitHub/Git:

- Los insumos anteriores ya estan trackeados en Git en la rama actual.
- No subir `Arcgis shapes y curvas.zip`; pesa ~746 MB y duplica insumos.
- `Cauce.shp` pesa ~72 MB, por debajo del limite de 100 MB de GitHub.
