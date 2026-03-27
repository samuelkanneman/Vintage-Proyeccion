"""
Genera factores de conversión acumulados para el análisis vintage.

El factor de conversión mide cuánto se multiplica la mora inicial (MOB 1)
al llegar a cada MOB posterior:

    factor_conversion(MOB_n) = índice(MOB_n) / índice(MOB_1)

- Factor > 1: la mora al MOB n es mayor que al nacer (deterioro neto)
- Factor = 1: la mora se mantuvo igual
- Factor < 1: la mora al MOB n es menor que al nacer (curación neta)

Genera dos archivos:
- factores_conversion.csv: matriz cohorte × MOB con los factores acumulados
- factores_conversion_resumen.csv: resumen estadístico por MOB (promedio,
  mediana, desvío, mín, máx, tendencia)

Uso:
    python src/generar_factores_conversion.py
"""

import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DIR, "matriz_vintage.csv")
OUTPUT_FACTORES = os.path.join(PROCESSED_DIR, "factores_conversion.csv")
OUTPUT_RESUMEN = os.path.join(PROCESSED_DIR, "factores_conversion_resumen.csv")


def calcular_factores_conversion(matriz: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula factor_conversion(MOB_n) = índice(MOB_n) / índice(MOB_1)
    para cada cohorte y MOB.
    """
    mob1 = matriz["MOB_1"]
    factores = matriz.div(mob1, axis=0)
    # MOB_1 siempre es 1.0 por definición, lo dejamos para referencia
    return factores


def generar_resumen(factores: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas descriptivas del factor de conversión por MOB.
    Incluye tendencia (comparación primera mitad vs segunda mitad de cohortes).
    """
    filas = []
    for col in factores.columns:
        vals = factores[col].dropna()
        if len(vals) < 2:
            continue

        mob = int(col.replace("MOB_", ""))
        n = len(vals)
        promedio = vals.mean()
        mediana = vals.median()
        desvio = vals.std()
        minimo = vals.min()
        maximo = vals.max()
        cohorte_min = vals.idxmin()
        cohorte_max = vals.idxmax()

        # Tendencia: primera mitad vs segunda mitad
        mid = n // 2
        prom_1h = vals.iloc[:mid].mean()
        prom_2h = vals.iloc[mid:].mean()
        if prom_1h > 0:
            cambio_pct = (prom_2h - prom_1h) / prom_1h * 100
        else:
            cambio_pct = 0.0

        if cambio_pct > 2:
            tendencia = "SUBIENDO"
        elif cambio_pct < -2:
            tendencia = "BAJANDO"
        else:
            tendencia = "ESTABLE"

        filas.append({
            "mob": mob,
            "n_cohortes": n,
            "promedio": promedio,
            "mediana": mediana,
            "desvio": desvio,
            "minimo": minimo,
            "maximo": maximo,
            "cohorte_min": cohorte_min,
            "cohorte_max": cohorte_max,
            "prom_1ra_mitad": prom_1h,
            "prom_2da_mitad": prom_2h,
            "cambio_pct": cambio_pct,
            "tendencia": tendencia,
        })

    return pd.DataFrame(filas)


def main():
    print("=" * 60)
    print("Generación de Factores de Conversión Acumulados")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró: {INPUT_FILE}")
        sys.exit(1)

    matriz = pd.read_csv(INPUT_FILE, sep=";", decimal=",", index_col=0)
    print(f"Matriz leída: {len(matriz)} cohortes x {len(matriz.columns)} MOBs")

    # Calcular factores
    factores = calcular_factores_conversion(matriz)
    factores.to_csv(OUTPUT_FACTORES, sep=";", decimal=",")
    print(f"\nFactores de conversión guardados: {OUTPUT_FACTORES}")

    # Resumen estadístico
    resumen = generar_resumen(factores)
    resumen.to_csv(OUTPUT_RESUMEN, sep=";", decimal=",", index=False)
    print(f"Resumen estadístico guardado: {OUTPUT_RESUMEN}")

    # Mostrar resumen
    print(f"\n{'MOB':>5s} | {'N':>3s} | {'Promedio':>8s} | {'Mediana':>8s} | {'Desvío':>7s} | {'Mín':>7s} | {'Máx':>7s} | {'Tendencia':>10s}")
    print("-" * 75)
    for _, row in resumen.iterrows():
        print(f"{int(row['mob']):>5d} | {int(row['n_cohortes']):>3d} | "
              f"{row['promedio']:>8.3f} | {row['mediana']:>8.3f} | "
              f"{row['desvio']:>7.3f} | {row['minimo']:>7.3f} | "
              f"{row['maximo']:>7.3f} | {row['tendencia']:>10s}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
