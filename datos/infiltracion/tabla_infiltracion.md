# Datos de infiltración para el modelo de Ciudad Cariari

Respaldo documental de los parámetros de infiltración usados en la
modelación (hidrología superficial del visor 3D y Green-Ampt de SWMM).
Actualizado: 2026-07-18. Datos tabulares en `tabla_infiltracion.csv`.

## 1. Parámetros Green-Ampt de literatura (por textura de suelo)

Valores clásicos de Rawls, Brakensiek y Miller (1983), reproducidos en el
manual de referencia hidrológica de SWMM (Rossman y Huber, 2016). Son los
valores estándar que SWMM recomienda cuando no hay ensayos in situ.

| Textura (USDA) | K sat (mm/h) | Succión ψ (mm) | Porosidad efectiva |
|---|---|---|---|
| Arena | 120,4 | 49 | 0,417 |
| Arena francosa | 30,0 | 61 | 0,401 |
| **Franco arenoso** | **10,9** | **110** | **0,412** |
| Franco | 3,3 | 89 | 0,434 |
| Franco limoso | 6,6 | 170 | 0,486 |
| Franco arcillo-arenoso | 1,5 | 220 | 0,330 |
| Franco arcilloso | 1,0 | 210 | 0,309 |
| Franco arcillo-limoso | 1,0 | 270 | 0,432 |
| Arcilla limosa | 0,5 | 290 | 0,423 |
| Arcilla | 0,3 | 320 | 0,385 |

Los parámetros adoptados en la modelación preliminar (K = 5 mm/h,
ψ = 110 mm, déficit inicial 0,25) corresponden a un **franco arenoso** con
la conductividad reducida ~50 % por compactación urbana — criterio
conservador usual en suelos urbanizados.

## 2. Mediciones reales en el Valle Central (Macías, Vargas y Arellano, 2018)

Ensayos de conductividad hidráulica en zona no saturada en dos sitios del
Valle Central; el sitio de San Rafael de Alajuela **colinda con el cantón
de Belén** (≈3–4 km del área de estudio), sobre suelos limosos (MH):

| Sitio | Suelo | Método | Tasa medida |
|---|---|---|---|
| San Rafael de Alajuela (Finca AyA) | Limo (MH) | Doble anillo (tasa global) | 2,1–4,5 mm/h |
| San Rafael de Alajuela (Finca AyA) | Limo (MH) | Porchet | 10–60 mm/h |
| Sabanilla, Montes de Oca (UCR) | Limo arenoso (MH-ML) | Doble anillo (tasa global) | 45–99 mm/h |
| Sabanilla, Montes de Oca (UCR) | Limo arenoso (MH-ML) | Porchet | 158–1 086 mm/h |

Lectura: en el sitio vecino a Belén la infiltración superficial medida con
doble anillo (el método estándar para dimensionar SbN) fue **2–4,5 mm/h**,
consistente con el K = 5 mm/h adoptado. Los métodos de pozo (Porchet,
Guelph) dan valores mayores porque miden horizontes más profundos — por
eso los pozos de infiltración pueden ser viables aun donde la superficie
infiltra poco, pero **cada sitio SbN requiere su prueba in situ**.

## 3. Clasificación adoptada en la tesis (Tabla 16 del documento de resultados)

| Clase | Tasa | Tecnologías SbN compatibles |
|---|---|---|
| Alta | > 50 mm/h | Jardines de lluvia, biorretención |
| Media | 10–50 mm/h | Cunetas verdes, zanjas infiltrantes |
| Baja | < 10 mm/h | Pavimento permeable con base estructural |

Con las mediciones locales disponibles, el área de estudio se ubica de
forma preliminar en clase **Baja a Media** en superficie (limos urbanos
compactados) con horizontes profundos más permeables. Esto respalda la
decisión del documento de resultados de condicionar pozos de infiltración
a prueba de infiltración y de priorizar jardines de lluvia/zanjas donde
hay área verde (suelo no compactado).

## 4. Fracciones de infiltración por cobertura usadas en el modelo D8

El modelo superficial del visor 3D asigna a cada celda una fracción de
infiltración por cobertura (resto = escorrentía). Equivalencia con
coeficientes de escorrentía C de literatura (Chow, Maidment y Mays, 1994;
ver también tabla de C de la norma AyA, 2017):

| Cobertura del modelo | Infiltración modelo | C equivalente | C literatura |
|---|---|---|---|
| Techos/edificios | 2 % | 0,98 | 0,75–0,95 (techos) |
| Calles/pavimento | 6 % | 0,94 | 0,85–0,95 (asfalto/concreto) |
| Suelo abierto urbano ("base") | 22 % | 0,78 | 0,60–0,85 (residencial denso) |
| Vegetación/zona verde | 90 % | 0,10 | 0,10–0,35 (parques/césped) |

Los cuatro valores del modelo caen dentro de los rangos de literatura,
del lado conservador (más escorrentía) para las superficies duras.

## 5. Citas (APA)

- Rawls, W. J., Brakensiek, D. L., y Miller, N. (1983). Green-Ampt
  infiltration parameters from soils data. *Journal of Hydraulic
  Engineering, 109*(1), 62–70.
- Rossman, L. A., y Huber, W. C. (2016). *Storm Water Management Model
  Reference Manual, Volume I — Hydrology* (EPA/600/R-15/162A). U.S. EPA.
- Macías, J., Vargas, A., y Arellano, F. (2018). Conductividad hidráulica
  en dos sitios del Valle Central de Costa Rica: análisis comparativo de
  tres metodologías de ensayo en la zona no saturada. *Revista Geológica
  de América Central, 59*, 71–96. https://doi.org/10.15517/rgac.v59i0.34160
- Chow, V. T., Maidment, D. R., y Mays, L. W. (1994). *Hidrología
  aplicada*. McGraw-Hill.
- Instituto Costarricense de Acueductos y Alcantarillados. (2017). *Norma
  técnica para diseño y construcción de sistemas de abastecimiento de agua
  potable, de saneamiento y pluvial*. AyA.
