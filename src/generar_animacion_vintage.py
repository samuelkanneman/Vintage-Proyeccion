"""
Script para generar la animación GIF de curvas vintage.

Lee data/processed/matriz_vintage.csv y genera una animación donde cada cohorte
se dibuja MOB a MOB. Al completarse, queda como rastro semitransparente mientras
la siguiente cohorte comienza a animarse.

El resultado se guarda en reports/curvas_vintage_animado.gif

Uso:
    py src/generar_animacion_vintage.py
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "matriz_vintage.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
OUTPUT_FILE = os.path.join(REPORTS_DIR, "curvas_vintage_animado.gif")

# ---------------------------------------------------------------------------
# Configuración de la animación
# ---------------------------------------------------------------------------
FPS = 8                          # Frames por segundo
PAUSA_FINAL = 12                 # Frames de pausa al completar cada cohorte
COLOR_2024 = "#2171b5"
COLOR_2025 = "#e6550d"
ALPHA_RASTRO = 0.20              # Transparencia de cohortes ya completadas
ALPHA_ACTIVA = 0.90              # Transparencia de la cohorte activa
LW_ACTIVA = 2.8
LW_RASTRO = 1.2


def color_cohorte(cohorte: str) -> str:
    """Devuelve el color según el año de la cohorte."""
    return COLOR_2024 if cohorte.startswith("2024") else COLOR_2025


def extraer_datos(matriz: pd.DataFrame, cohorte: str):
    """Extrae MOBs y valores no nulos para una cohorte."""
    vals = matriz.loc[cohorte].dropna()
    mobs = [int(col.replace("MOB_", "")) for col in vals.index]
    return mobs, vals.values


def main():
    print("=" * 60)
    print("Generación de Animación Vintage")
    print("=" * 60)

    # Verificar archivo de entrada
    if not os.path.isfile(INPUT_FILE):
        print(f"ERROR: No se encontró el archivo de entrada: {INPUT_FILE}")
        sys.exit(1)

    # Crear directorio de reportes si no existe
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Leer datos
    matriz = pd.read_csv(INPUT_FILE, sep=";", decimal=",", index_col=0)
    cohortes = sorted(matriz.index)
    mob_max = max(int(c.replace("MOB_", "")) for c in matriz.columns)

    print(f"Cohortes: {len(cohortes)} | MOB máximo: {mob_max}")

    # Armar secuencia de frames: para cada cohorte, un frame por MOB + pausa
    frames_info = []  # lista de (cohorte_idx, n_puntos_visibles)
    for ci, cohorte in enumerate(cohortes):
        mobs, vals = extraer_datos(matriz, cohorte)
        for n in range(1, len(mobs) + 1):
            frames_info.append((ci, n))
        # Pausa al completar la cohorte
        for _ in range(PAUSA_FINAL):
            frames_info.append((ci, len(mobs)))

    total_frames = len(frames_info)
    print(f"Total frames: {total_frames} | Duración: ~{total_frames / FPS:.0f}s")

    # -------------------------------------------------------------------
    # Crear animación
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))

    def init_anim():
        ax.set_xlim(0.5, mob_max + 1.5)
        ax.set_ylim(0, 0.38)
        ax.set_xlabel("Months on Books (MOB)", fontsize=12)
        ax.set_ylabel("Índice de Morosidad", fontsize=12)
        ax.set_title("OC - Curvas Vintage Animadas — Evolución del Índice de Morosidad",
                      fontsize=14, fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        ax.set_xticks(range(1, mob_max + 1))
        ax.grid(True, alpha=0.25, linestyle="--")
        plt.figtext(0.5, 0.01, "Autor: Kanneman, Samuel", ha="center", fontsize=10, color="gray")
        plt.tight_layout(pad=3.0)
        return []

    def animate(frame_idx):
        ax.clear()
        init_anim()

        ci_actual, n_puntos = frames_info[frame_idx]

        # Dibujar cohortes ya completadas (rastro)
        for ci in range(ci_actual):
            c = cohortes[ci]
            mobs, vals = extraer_datos(matriz, c)
            ax.plot(mobs, vals, color=color_cohorte(c),
                    alpha=ALPHA_RASTRO, linewidth=LW_RASTRO, zorder=1)

        # Dibujar cohorte activa (parcial)
        c_activa = cohortes[ci_actual]
        mobs, vals = extraer_datos(matriz, c_activa)
        mobs_vis = mobs[:n_puntos]
        vals_vis = vals[:n_puntos]

        ax.plot(mobs_vis, vals_vis, color=color_cohorte(c_activa),
                alpha=ALPHA_ACTIVA, linewidth=LW_ACTIVA, zorder=3)
        # Punto final
        ax.plot(mobs_vis[-1], vals_vis[-1], "o",
                color=color_cohorte(c_activa), markersize=8, zorder=4)

        # Etiqueta de cohorte activa
        ax.annotate(c_activa,
                    xy=(mobs_vis[-1], vals_vis[-1]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=11, fontweight="bold",
                    color=color_cohorte(c_activa), zorder=5)

        # Info del frame
        indice_pct = "{:.2%}".format(float(vals_vis[-1]))
        info_text = "Cohorte: {}\nMOB: {}\nÍndice: {}".format(
            c_activa, mobs_vis[-1], indice_pct)
        ax.text(0.02, 0.95, info_text,
                transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
                zorder=5)

        # Leyenda
        legend_elements = [
            Line2D([0], [0], color=COLOR_2024, lw=3, label="Cohortes 2024"),
            Line2D([0], [0], color=COLOR_2025, lw=3, label="Cohortes 2025"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        return []

    print("Generando animación (puede tardar unos minutos)...")
    anim = animation.FuncAnimation(fig, animate, init_func=init_anim,
                                   frames=total_frames, interval=1000 // FPS,
                                   blit=False, repeat=False)

    # Guardar como GIF
    anim.save(OUTPUT_FILE, writer="pillow", fps=FPS, dpi=100)
    plt.close(fig)
    print(f"GIF guardado en: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
