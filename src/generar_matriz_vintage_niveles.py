"""
Genera matrices vintage por segmento (tipoope x clientenuevo x nivel_riesgo).

Lee data/processed/vintage_niveles_consolidado.csv y genera una matriz
pivoteada por cada uno de los 12 segmentos. Todas las matrices se guardan
en un solo archivo con una columna adicional 'segmento'.

Salida: data/processed/matrices_vintage_niveles.csv

Uso:
    python src/generar_matriz_vintage_niveles.py
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "vintage_niveles_consolidado.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "matrices_vintage_niveles.csv")


def main():
    print("=" * 60)
    print("Matrices Vintage por Nivel de Riesgo")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontro: {INPUT_FILE}")
        print("Ejecuta primero: python src/consolidar_vintage_niveles.py")
        sys.exit(1)

    df = pd.read_csv(INPUT_FILE, sep=";", decimal=",")
    print(f"Archivo leido: {len(df)} filas, {df['segmento'].nunique()} segmentos")

    matrices = []
    for segmento in sorted(df["segmento"].unique()):
        sub = df[df["segmento"] == segmento]
        matriz = sub.pivot_table(index="cohorte", columns="mob", values="indice")
        matriz = matriz.sort_index()
        matriz = matriz[sorted(matriz.columns)]
        matriz.columns = [f"MOB_{int(c)}" for c in matriz.columns]
        matriz.index.name = "cohorte"
        matriz["segmento"] = segmento
        matrices.append(matriz.reset_index())
        print(f"  {segmento}: {matriz.shape[0]} cohortes x {matriz.shape[1]-1} MOBs")

    resultado = pd.concat(matrices, ignore_index=True)

    # Reordenar columnas: segmento, cohorte, MOB_1, MOB_2, ...
    mob_cols = [c for c in resultado.columns if c.startswith("MOB_")]
    mob_cols = sorted(mob_cols, key=lambda x: int(x.replace("MOB_", "")))
    resultado = resultado[["segmento", "cohorte"] + mob_cols]

    resultado.to_csv(OUTPUT_FILE, sep=";", index=False, decimal=",")
    print(f"\nArchivo guardado: {OUTPUT_FILE}")
    print(f"Total: {len(resultado)} filas")
    print("=" * 60)


if __name__ == "__main__":
    main()
