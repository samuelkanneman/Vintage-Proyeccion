"""
Script para generar la matriz vintage a partir del archivo consolidado.

Lee data/processed/vintage_consolidado.csv y genera una tabla pivoteada
donde las filas son las cohortes, las columnas son los MOB (months on books)
y los valores son el índice de morosidad.

El resultado se guarda en data/processed/matriz_vintage.csv

Uso:
    py src/generar_matriz_vintage.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "vintage_consolidado.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "matriz_vintage.csv")


def main():
    print("=" * 60)
    print("Generación de Matriz Vintage")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró el archivo consolidado: {INPUT_FILE}")
        print("Ejecutá primero: py src/consolidar_vintage.py")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, sep=";", decimal=",")
    print(f"Archivo leído: {len(df)} filas, {df['cohorte'].nunique()} cohortes")

    # Pivotear: filas=cohorte, columnas=mob, valores=indice
    matriz = df.pivot_table(index="cohorte", columns="mob", values="indice")

    # Ordenar cohortes cronológicamente y MOB de menor a mayor
    matriz = matriz.sort_index()
    matriz = matriz[sorted(matriz.columns)]

    # Renombrar columnas para mayor claridad: MOB_1, MOB_2, ...
    matriz.columns = [f"MOB_{int(c)}" for c in matriz.columns]

    print(f"\nMatriz vintage generada: {matriz.shape[0]} cohortes x {matriz.shape[1]} meses")
    print(f"\nCohortes: {list(matriz.index)}")
    print(f"MOBs: {list(matriz.columns)}")

    # Guardar
    matriz.to_csv(OUTPUT_FILE, sep=";", decimal=",")

    print(f"\nArchivo guardado: {OUTPUT_FILE}")
    print(f"\nVista previa:")
    print(matriz.to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
