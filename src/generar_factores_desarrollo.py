"""
Script para generar la matriz de factores de desarrollo (link ratios).

El factor de desarrollo mide la variación del índice de morosidad entre
un MOB y el siguiente:

    factor(MOB_n) = indice(MOB_n) / indice(MOB_n-1)

Interpretación:
    - factor > 1 : la mora está creciendo respecto al mes anterior
    - factor = 1 : la mora se mantiene estable
    - factor < 1 : la mora está disminuyendo (recuperación o estabilización)

Esto permite detectar en qué momento del ciclo de vida del crédito
cambia la tendencia de la mora, y si ese patrón es consistente entre
cohortes o hay quiebres estructurales.

Lee data/processed/matriz_vintage.csv y genera:
    - data/processed/factores_desarrollo.csv

Uso:
    py src/generar_factores_desarrollo.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "matriz_vintage.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "factores_desarrollo.csv")


def main():
    print("=" * 60)
    print("Generación de Factores de Desarrollo")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró: {INPUT_FILE}")
        print("Ejecutá primero: py src/generar_matriz_vintage.py")
        sys.exit(1)

    matriz = pd.read_csv(INPUT_FILE, sep=";", decimal=",", index_col=0)
    print(f"Matriz leída: {matriz.shape[0]} cohortes x {matriz.shape[1]} MOBs")

    # Extraer números de MOB
    mob_nums = [int(c.replace("MOB_", "")) for c in matriz.columns]

    # Calcular factores: MOB_n / MOB_(n-1)
    factores = pd.DataFrame(index=matriz.index)
    for i in range(1, len(mob_nums)):
        col_actual = f"MOB_{mob_nums[i]}"
        col_anterior = f"MOB_{mob_nums[i-1]}"
        col_factor = f"{mob_nums[i-1]}->{mob_nums[i]}"
        factores[col_factor] = matriz[col_actual] / matriz[col_anterior]

    factores.index.name = "cohorte"

    # Guardar
    factores.to_csv(OUTPUT_FILE, sep=";", decimal=",")

    print(f"\nFactores generados: {factores.shape[0]} cohortes x {factores.shape[1]} transiciones")
    print(f"Archivo guardado: {OUTPUT_FILE}")
    print(f"\nVista previa:")
    print(factores.iloc[:, :8].to_string(float_format="{:.4f}".format))
    print("=" * 60)


if __name__ == "__main__":
    main()
