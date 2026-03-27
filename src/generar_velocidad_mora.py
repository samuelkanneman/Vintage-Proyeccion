"""
Script para generar la matriz de velocidad de mora.

La velocidad es la primera derivada del índice de morosidad respecto al
tiempo (MOB). Mide el cambio absoluto del índice entre MOBs consecutivos:

    velocidad(MOB_n) = indice(MOB_n) - indice(MOB_n-1)

Interpretación:
    - velocidad > 0 : la mora sigue creciendo (la cosecha aún se deteriora)
    - velocidad = 0 : la mora se estancó (plateau/meseta)
    - velocidad < 0 : la mora mejora (curación de la cosecha)

Combinando con el factor de desarrollo (segunda derivada):
    - velocidad > 0 y factor bajando hacia 1 : desaceleración (mora crece
      pero cada vez más lento, acercándose al plateau)
    - velocidad > 0 y factor subiendo        : aceleración (deterioro
      se intensifica)
    - velocidad < 0 y factor < 1             : curación activa

Lee data/processed/matriz_vintage.csv y genera:
    - data/processed/velocidad_mora.csv

Uso:
    py src/generar_velocidad_mora.py
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
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "velocidad_mora.csv")


def main():
    print("=" * 60)
    print("Generación de Velocidad de Mora (1ra derivada)")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró: {INPUT_FILE}")
        print("Ejecutá primero: py src/generar_matriz_vintage.py")
        sys.exit(1)

    matriz = pd.read_csv(INPUT_FILE, sep=";", decimal=",", index_col=0)
    print(f"Matriz leída: {matriz.shape[0]} cohortes x {matriz.shape[1]} MOBs")

    mob_nums = [int(c.replace("MOB_", "")) for c in matriz.columns]

    # Calcular velocidad: indice(MOB_n) - indice(MOB_(n-1))
    velocidad = pd.DataFrame(index=matriz.index)
    for i in range(1, len(mob_nums)):
        col_actual = f"MOB_{mob_nums[i]}"
        col_anterior = f"MOB_{mob_nums[i-1]}"
        col_vel = f"{mob_nums[i-1]}->{mob_nums[i]}"
        velocidad[col_vel] = matriz[col_actual] - matriz[col_anterior]

    velocidad.index.name = "cohorte"

    # Guardar
    velocidad.to_csv(OUTPUT_FILE, sep=";", decimal=",")

    print(f"\nVelocidad generada: {velocidad.shape[0]} cohortes x {velocidad.shape[1]} transiciones")
    print(f"Archivo guardado: {OUTPUT_FILE}")
    print(f"\nVista previa (primeras 8 transiciones):")
    print(velocidad.iloc[:, :8].to_string(float_format="{:.6f}".format))
    print("=" * 60)


if __name__ == "__main__":
    main()
