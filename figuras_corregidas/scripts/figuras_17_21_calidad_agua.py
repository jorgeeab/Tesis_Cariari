#!/usr/bin/env python3
"""Regenera las figuras 17 a 21 con los datos de las tablas 14 a 16."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
CM = 1 / 2.54
PUNTOS = ["P1", "P2", "P3", "P4", "P5", "P6"]
PERIODOS = ["seca", "transicion", "lluviosa"]
ETIQUETAS = ["Época seca", "Transición (mayo 2025)",
             "Época lluviosa (noviembre 2025)"]
COLORES = ["#efa400", "#50abe0", "#0879b5"]

# Transcripcion literal de las tablas 14, 15 y 16 del documento de tesis.
DATOS = {
    "DBO": {
        "seca": ["11.2 +/- 1.3", "7.1 +/- 1.0", "12.4 +/- 1.3", "15.9 +/- 1.6", "13.6 +/- 1.4", "19.0 +/- 1.9"],
        "transicion": ["6.9 +/- 1.0", "13.8 +/- 1.4", "18.0 +/- 1.8", "21.4 +/- 2.1", "18.3 +/- 1.8", "36.4 +/- 3.4"],
        "lluviosa": ["6.00 +/- 0.90", "7.6 +/- 1.0", "4.80 +/- 0.90", "3.90 +/- 0.80", "13.3 +/- 1.4", "20.3 +/- 2.0"],
    },
    "DQO": {
        "seca": ["d", "d", "d", "31.25 +/- 0.96", "44.17 +/- 0.93", "39.81 +/- 0.93"],
        "transicion": ["nd", "24.8 +/- 2.0", "d", "d", "33.2 +/- 1.2", "78.9 +/- 1.4"],
        "lluviosa": ["nd", "nd", "23.4 +/- 1.6", "21.7 +/- 1.6", "21.0 +/- 1.7", "50.2 +/- 1.5"],
    },
    "SST": {
        "seca": ["nd", "nd", "8.7 +/- 1.2", "12.8 +/- 1.7", "d", "15.5 +/- 2.0"],
        "transicion": ["nd", "nd", "d", "113.25 +/- 0.11", "9.06 +/- 0.10", "21.91 +/- 0.10"],
        "lluviosa": ["47.00 +/- 0.10", "1.63 +/- 0.10", "18.06 +/- 0.10", "nd", "15.52 +/- 0.10", "43.69 +/- 0.10"],
    },
    "SAAM": {
        "seca": ["nd", "d", "nd", "2.59 +/- 0.20", "1.90 +/- 0.15", "nd"],
        "transicion": ["nd", "d", "nd", "nd", "1.94 +/- 0.15", "1.185 +/- 0.099"],
        "lluviosa": ["nd", "nd", "d", "1.247 +/- 0.095", "0.724 +/- 0.067", "0.440 +/- 0.056"],
    },
    "Ecoli": {
        "seca": ["540", "350", "920", "920", "540", "1600"],
        "transicion": ["2.8e6", "2.8e7", "1.4e4", "1.7e5", "2.2e4", "1.1e5"],
        "lluviosa": ["4.9e4", "1.4e5", "ANS", "3.3e4", "1.1e5", "No disponible"],
    },
}

NOMBRES = {"DBO": "DBO (mg/L)", "DQO": "DQO (mg/L)",
           "SST": "SST (mg/L)", "SAAM": "SAAM (mg/L)",
           "Ecoli": "E. coli (NMP/100 mL)"}
ARCHIVOS = {"DBO": "figura_17_DBO", "DQO": "figura_18_DQO",
            "SST": "figura_19_SST", "SAAM": "figura_20_SAAM",
            "Ecoli": "figura_21_Ecoli"}


def separar(texto: str) -> tuple[float, float]:
    partes = re.split(r"\s*\+/-\s*", texto)
    try:
        return float(partes[0]), float(partes[1]) if len(partes) > 1 else 0.0
    except ValueError:
        return math.nan, 0.0


def etiqueta(texto: str) -> str:
    if "+/-" in texto:
        texto = texto.split("+/-", 1)[0].strip()
    try:
        valor = float(texto)
    except ValueError:
        return texto
    if valor >= 10_000:
        exponente = int(math.floor(math.log10(valor)))
        coef = valor / (10 ** exponente)
        return f"{coef:.1f}x10^{exponente}".replace(".", ",")
    dec = 0 if valor >= 100 else (2 if valor < 5 else 1)
    return f"{valor:.{dec}f}".replace(".", ",")


def generar_parametro(parametro: str, salida: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(15.9 * CM, 7.6 * CM))
    x = np.arange(len(PUNTOS))
    ancho = 0.26
    maximo = 0.0

    for j, periodo in enumerate(PERIODOS):
        valores, errores = zip(*(separar(t) for t in DATOS[parametro][periodo]))
        posiciones = x + (j - 1) * ancho
        barras = ax.bar(posiciones, np.nan_to_num(valores), ancho,
                        yerr=errores, capsize=2, color=COLORES[j],
                        label=ETIQUETAS[j], linewidth=0)
        for barra, texto, valor, error in zip(barras, DATOS[parametro][periodo],
                                               valores, errores):
            if math.isnan(valor):
                ax.text(barra.get_x() + barra.get_width() / 2, 0, texto,
                        ha="center", va="bottom", rotation=90, fontsize=7)
            else:
                maximo = max(maximo, valor + error)
                ax.text(barra.get_x() + barra.get_width() / 2,
                        valor + error + max(valor * 0.025, 0.05), etiqueta(texto),
                        ha="center", va="bottom", rotation=90, fontsize=7)

    if parametro == "Ecoli":
        ax.set_yscale("log")
        ax.set_ylim(100, maximo * 3.5)
    else:
        ax.set_ylim(0, maximo * 1.18 if maximo else 1)
    ax.set_ylabel(NOMBRES[parametro])
    ax.set_xlabel("Punto de muestreo")
    ax.set_xticks(x, PUNTOS)
    ax.grid(axis="y", color="#dfe3e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=3,
              frameon=False, handlelength=1.2, columnspacing=1.4)
    fig.text(0.01, 0.014,
             "nd: no detectado; d: detectado, no cuantificado; ANS: análisis no satisfactorio; s. d.: sin dato. Barras de error: incertidumbre informada.",
             fontsize=5.8, color="#5b6570")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.20)
    base = salida / ARCHIVOS[parametro]
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(base)


def generar(output_root: Path) -> None:
    salida = output_root / "figuras_17_21_calidad_agua"
    salida.mkdir(parents=True, exist_ok=True)
    for parametro in DATOS:
        generar_parametro(parametro, salida)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "figuras_corregidas")
    generar(parser.parse_args().output_root.resolve())
