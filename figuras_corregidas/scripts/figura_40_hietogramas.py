#!/usr/bin/env python3
"""Regenera la figura 40 a partir de los hietogramas de simulacion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
CM = 1 / 2.54


def formato_decimal(valor: float, decimales: int = 1) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def cargar(carpeta: str) -> tuple[pd.DataFrame, dict]:
    base = REPO / "results" / "simulaciones" / carpeta
    return pd.read_csv(base / "hietograma.csv"), json.loads(
        (base / "resumen.json").read_text(encoding="utf-8")
    )


def total_lluvia(datos: pd.DataFrame) -> float:
    paso_h = float(datos["minuto"].diff().dropna().median()) / 60
    return float(datos["intensidad_mm_h"].sum() * paso_h)


def generar(output_root: Path) -> None:
    real, _ = cargar("evento_real_2016-06-14")
    diseno, _ = cargar("diseno_T10_santa_lucia")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#5b6570",
        "axes.linewidth": 0.8,
        "xtick.color": "#5b6570",
        "ytick.color": "#5b6570",
    })
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.9 * CM, 6.8 * CM),
        gridspec_kw={"width_ratios": (1.35, 1), "wspace": 0.22},
    )

    horas = 10 + real["minuto"] / 60
    ax1.bar(horas, real["intensidad_mm_h"], width=0.9, color="#0879b5",
            edgecolor="white", linewidth=0.35)
    ax1.set_title("a) Evento ERA5, paso horario", loc="left", fontsize=8.5)
    ax1.set_xlabel("Hora local, 14-06-2016")
    ax1.set_ylabel("Intensidad (mm/h)")
    ax1.set_xlim(10, 24)
    ax1.set_ylim(0, 27.8)
    ax1.set_xticks(range(10, 25, 2), [f"{h}:00" for h in range(10, 25, 2)])
    ax1.text(0.02, 0.93, f"Total: {formato_decimal(total_lluvia(real))} mm",
             transform=ax1.transAxes, fontsize=9)
    i1 = real["intensidad_mm_h"].idxmax()
    ax1.text(horas.iloc[i1], real.loc[i1, "intensidad_mm_h"] + 0.35,
             f"{formato_decimal(real.loc[i1, 'intensidad_mm_h'])} mm/h",
             ha="center", va="bottom", fontsize=8.5)

    ax2.bar(diseno["minuto"], diseno["intensidad_mm_h"], width=4.5,
            color="#d95f02", edgecolor="white", linewidth=0.35)
    ax2.set_title("b) Diseño T = 10 años, bloques de 5 min", loc="left", fontsize=7.5)
    ax2.set_xlabel("Minuto de la tormenta")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 290)
    ax2.set_xticks(range(0, 61, 10))
    ax2.text(0.02, 0.93, f"Total: {formato_decimal(total_lluvia(diseno))} mm",
             transform=ax2.transAxes, fontsize=9)
    i2 = diseno["intensidad_mm_h"].idxmax()
    ax2.text(diseno.loc[i2, "minuto"], diseno.loc[i2, "intensidad_mm_h"] + 2,
             f"{formato_decimal(diseno.loc[i2, 'intensidad_mm_h'])} mm/h",
             ha="center", va="bottom", fontsize=8.5)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#dfe3e8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.01, 0.015, "Las escalas verticales de a) y b) son distintas.",
             color="#5b6570", fontsize=7.5)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.22)

    salida = output_root / "figura_40"
    salida.mkdir(parents=True, exist_ok=True)
    base = salida / "figura_40_hietogramas"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "figuras_corregidas")
    generar(parser.parse_args().output_root.resolve())
