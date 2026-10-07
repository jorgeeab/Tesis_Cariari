# Datos de lluvia para el modelo de Ciudad Cariari

Documento de respaldo: de dónde salen los datos de lluvia usados en la
modelación, cómo descargarlos de nuevo (replicabilidad) y qué valores se
adoptaron. Actualizado: 2026-07-18.

## 1. Fuentes utilizadas

| Fuente | Qué aporta | Acceso |
|---|---|---|
| Reanálisis **ERA5** (ECMWF), serie horaria 2000–2026 para la celda de Cariari | Climatología local, estacionalidad, forma temporal de eventos reales | Descarga libre vía API Open-Meteo (ver §2) |
| **IMN — Curvas IDF de estaciones mecánicas** (estación 84-111 Santa Lucía, Heredia, 1983–2010) | Intensidades máximas de diseño por duración y período de retorno | PDF público, `referencias/IMN_IDF_estaciones_mecanicas.pdf` |
| **IMN — Curvas IDF de estaciones automáticas** (estación 84-195 Aeropuerto Pavas, 16 años) | Segunda estación para contraste (duraciones 5–30 min) | PDF público, `referencias/IMN_IDF_estaciones_automaticas.pdf` |

Estaciones IMN con IDF publicadas más cercanas al área de estudio
(Cariari ≈ 9.973 N, −84.161 O): 84-111 Santa Lucía (Heredia, ~7 km NE),
84-46 San Josecito (Heredia, ~7 km N), 84-195 Aeropuerto Pavas (~8 km SE),
84-191 RECOPE La Garita (~8 km O). No existe estación IMN dentro de Belén
con IDF publicada; Santa Lucía es la de registro más largo (29 años) con
duraciones hasta 24 h, por eso se adopta como estación de referencia.

## 2. Serie horaria ERA5 (descargable por cualquiera)

Archivo: `era5_cariari_horario_2000_2026.csv` (232 272 registros horarios,
2000-01-01 a 2026-06-30, hora local UTC−6). Comando de descarga:

```
curl "https://archive-api.open-meteo.com/v1/archive?latitude=9.9729&longitude=-84.1614&start_date=2000-01-01&end_date=2026-06-30&hourly=precipitation&timezone=America%2FCosta_Rica&format=csv" -o era5_cariari_horario_2000_2026.csv
```

La API devuelve la celda de reanálisis centrada en 9.947 N, −84.173 O
(elevación 952 m), que cubre el área de estudio.

### Climatología derivada (2000–2025)

- Precipitación media anual: **2 963 mm** (mín. 1 735 mm en 2015; máx. 4 390 mm en 2022).
- Estación seca diciembre–abril (18–85 mm/mes); estación lluviosa mayo–noviembre
  con máximos en septiembre (451 mm) y octubre (487 mm) y veranillo en julio
  (321 mm) — patrón clásico del Pacífico/Valle Central.

| Mes | mm | Mes | mm | Mes | mm | Mes | mm |
|---|---|---|---|---|---|---|---|
| Ene | 31 | Abr | 85 | Jul | 321 | Oct | 487 |
| Feb | 18 | May | 371 | Ago | 386 | Nov | 292 |
| Mar | 26 | Jun | 402 | Sep | 451 | Dic | 79 |

### Eventos extremos de la serie

| Rank | Máx. 1 h | Máx. 2 h | Máx. día |
|---|---|---|---|
| 1 | 27,3 mm (22-set-2019) | 46,4 mm (14-jun-2016) | 174,6 mm (22-may-2004) |
| 2 | 23,5 mm (14-jun-2016) | — | 159,5 mm (21-ago-2007) |

**Evento seleccionado para simulación: 14 de junio de 2016** — tormenta de
~14 h (10:00–24:00) con 120,2 mm acumulados, núcleo convectivo de 2 h
(46,4 mm) y pico horario de 23,5 mm/h. Es el evento con la mayor lámina en
2 horas de los 26,5 años de la serie. Hietograma en
`../../results/simulaciones/evento_real_2016-06-14/hietograma.csv`.

### Limitación (declararla en la tesis)

ERA5 es un reanálisis con celda de ~28 km: representa bien acumulados y
formas temporales, pero **suaviza los picos convectivos puntuales**. Su
máximo horario (27 mm/h) está muy por debajo de las intensidades puntuales
de las estaciones IMN (Santa Lucía T5-5min = 215 mm/h). Por eso las
intensidades de diseño se toman de las curvas IDF del IMN y ERA5 se usa
para el contexto climatológico y para reproducir un evento real observado.

## 3. Intensidades de diseño — IDF estación 84-111 Santa Lucía (IMN)

Cuadro 17 del documento IMN (mm/h):

| Duración | T=5 | T=10 | T=25 | T=50 |
|---|---|---|---|---|
| 5 min | 215,1 | 245,0 | 282,8 | 310,8 |
| 10 min | 155,4 | 169,5 | 187,3 | 200,4 |
| 15 min | 140,6 | 154,4 | 171,9 | 184,8 |
| 30 min | 109,7 | 121,9 | 137,3 | 148,8 |
| 60 min | 74,5 | 82,9 | 93,4 | 101,3 |
| 120 min | 43,2 | 47,9 | 53,8 | 58,2 |

Ecuación ajustada por el IMN: **I = 672,71 · T^0,119 / D^0,652** (R² = 0,977;
29 años). Contraste Pavas 84-195: I = 187,08 · T^0,101 / D^0,322 (T30-min:
T5 = 67,7; T10 = 74,3 mm/h).

Implicación: el evento de trabajo usado en el borrador de resultados
(50 mm en 1 h) equivale a un período de retorno de ~2 años en Santa Lucía
(T2-60min ≈ 46–50 mm/h). Para diseño de red pluvial menor la práctica
nacional usa T = 5–10 años → **82,9 mm/h en 60 min (T10)**, que es el
escenario de diseño simulado.

## 4. Tormenta de diseño simulada

`scripts/simular_evento_real.py --escenario diseno --retorno 10` construye
una tormenta de **bloques alternos** de 60 min (paso 5 min) a partir del
Cuadro 17: lámina total 82,9 mm, bloque pico 20,4 mm/5 min (245 mm/h),
pico al centro. Resultados en `../../results/simulaciones/diseno_T10_santa_lucia/`.

## 5. Resultados de las dos simulaciones (red pluvial v7, INP 380 nodos/401 conductos)

| Métrica | Evento real 14-jun-2016 | Diseño T10 (IDF Santa Lucía) |
|---|---|---|
| Lámina total | 120,2 mm en 14 h | 82,9 mm en 1 h |
| Intensidad pico | 23,5 mm/h | 245 mm/h (5 min) |
| Volumen captado por tragantes | 184 606 m³ | 127 259 m³ |
| Caudal pico total captado | 10,0 m³/s | 104,5 m³/s |
| Nodos con inundación | 2 (crp_52, tragante_65) | 57 |
| Volumen de inundación | 676 m³ | 63 373 m³ |

Lectura: un temporal largo de intensidad moderada es manejado por la red;
la tormenta convectiva corta de T10 la sobrepasa de forma generalizada.
`crp_52` aparece como punto crítico en **ambos** escenarios. Los diámetros
del INP son estimados (0,60 m), por lo que los volúmenes de inundación son
indicativos de *dónde* se concentra el problema, no magnitudes de diseño.

## 6. Citas sugeridas (APA)

- Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global
  reanalysis. *Quarterly Journal of the Royal Meteorological Society,
  146*(730), 1999–2049. https://doi.org/10.1002/qj.3803
- Zippenfenig, P. (2023). *Open-Meteo.com weather API* [Software].
  Zenodo. https://doi.org/10.5281/zenodo.7970649
- Instituto Meteorológico Nacional. (s. f.). *Curvas de intensidad
  duración frecuencia de algunas estaciones meteorológicas mecánicas*.
  IMN/MINAE. http://cglobal.imn.ac.cr/documentos/publicaciones/CuInDuFreAlEsMecanicas/
- Rojas Morales, N. (2011). *Curvas de intensidad duración frecuencia de
  algunas estaciones meteorológicas automáticas*. IMN/MINAE.
  http://cglobal.imn.ac.cr/documentos/publicaciones/CuInDuFreAlEsAutomaticas/
- Murillo Muñoz, R. E. (1998). Curvas de Intensidad-Duración-Frecuencia
  para el Valle Central de Costa Rica. *VI Congreso Nacional de Recursos
  Hídricos y Saneamiento Ambiental, ACREH*. https://hdl.handle.net/10669/85391
