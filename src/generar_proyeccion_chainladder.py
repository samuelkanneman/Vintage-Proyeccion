"""
Script para proyectar el triangulo vintage usando el metodo Chain Ladder.

El metodo Chain Ladder completa el triangulo inferior de la matriz vintage
(las celdas sin datos observados) aplicando factores de desarrollo promedio
ponderados por volumen.

Para cada transicion MOB_n -> MOB_{n+1}:
    factor_promedio = sum(indice_{n+1}) / sum(indice_n)
    (sumando solo cohortes que tienen ambos MOBs observados)

Luego, para cada cohorte con datos hasta MOB_k:
    indice_proyectado(MOB_{k+1}) = indice_observado(MOB_k) * factor(k->k+1)
    indice_proyectado(MOB_{k+2}) = indice_proyectado(MOB_{k+1}) * factor(k+1->k+2)
    ...hasta MOB_OBJETIVO

Lee data/processed/matriz_vintage.csv y genera:
    - data/processed/factores_chainladder.csv (factores promedio por transicion)
    - data/processed/matriz_proyectada.csv (matriz completa hasta MOB_OBJETIVO)
    - data/processed/matriz_proyectada_marcadores.csv (True = proyectado, False = observado)

Uso:
    py src/generar_proyeccion_chainladder.py
"""

import os
import sys

import numpy as np
import pandas as pd
import chainladder as cl

# ---------------------------------------------------------------------------
# Configuracion de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "matriz_vintage.csv")
OUTPUT_FACTORES = os.path.join(PROCESSED_DATA_DIR, "factores_chainladder.csv")
OUTPUT_MATRIZ = os.path.join(PROCESSED_DATA_DIR, "matriz_proyectada.csv")
OUTPUT_MARCADORES = os.path.join(PROCESSED_DATA_DIR, "matriz_proyectada_marcadores.csv")

# ---------------------------------------------------------------------------
# Parametro principal: hasta que MOB proyectar
# ---------------------------------------------------------------------------
MOB_OBJETIVO = 18


def _matriz_a_triangulo_cl(matriz):
    """Convierte la matriz pandas en un triángulo chainladder."""
    mob_cols = sorted(
        [c for c in matriz.columns if c.startswith("MOB_")],
        key=lambda c: int(c.replace("MOB_", ""))
    )
    tri_data = matriz[mob_cols].copy()
    tri_data.index.name = "origin"
    tri_data.columns = [int(c.replace("MOB_", "")) for c in mob_cols]
    tri_data.columns.name = "development"
    tri = cl.Triangle(
        tri_data.reset_index(),
        origin="origin",
        development=[int(c.replace("MOB_", "")) for c in mob_cols],
        columns=[int(c.replace("MOB_", "")) for c in mob_cols],
    )
    return tri


def _construir_triangulo_cl(matriz):
    """
    Construye un triángulo chainladder a partir de la matriz vintage.
    Convierte la estructura cohorte × MOB (con NaN = no observado)
    al formato incremental/acumulado que espera chainladder.
    """
    mob_cols = sorted(
        [c for c in matriz.columns if c.startswith("MOB_")],
        key=lambda c: int(c.replace("MOB_", ""))
    )
    # Construir arrays para chainladder
    origins = list(matriz.index)
    devs = [int(c.replace("MOB_", "")) for c in mob_cols]
    values = matriz[mob_cols].values  # shape (n_origins, n_devs)

    # chainladder necesita un DataFrame en formato largo
    rows = []
    for i, origin in enumerate(origins):
        for j, dev in enumerate(devs):
            val = values[i, j]
            if not np.isnan(val):
                rows.append({"origin": origin, "dev": dev, "value": val})

    if not rows:
        return None

    df_long = pd.DataFrame(rows)
    tri = cl.Triangle(
        df_long, origin="origin", development="dev", columns="value"
    )
    return tri


def calcular_factores_chainladder(matriz, mob_objetivo):
    """
    Calcula factores de desarrollo promedio ponderados (Chain Ladder).

    Para cada transicion n -> n+1:
        factor = sum(MOB_{n+1}) / sum(MOB_n)
        (sobre cohortes con ambos valores observados)

    Returns:
        dict con clave "n->n+1" y valor el factor promedio.
    """
    mob_nums = [int(c.replace("MOB_", "")) for c in matriz.columns]
    factores = {}

    for i in range(len(mob_nums) - 1):
        mob_actual = mob_nums[i]
        mob_siguiente = mob_nums[i + 1]

        if mob_actual >= mob_objetivo:
            break

        col_actual = f"MOB_{mob_actual}"
        col_siguiente = f"MOB_{mob_siguiente}"

        # Solo cohortes con ambos valores observados
        mask = matriz[col_actual].notna() & matriz[col_siguiente].notna()
        n_cohortes = mask.sum()

        if n_cohortes > 0:
            suma_siguiente = matriz.loc[mask, col_siguiente].sum()
            suma_actual = matriz.loc[mask, col_actual].sum()
            factor = suma_siguiente / suma_actual if suma_actual != 0 else 1.0
            factores[f"{mob_actual}->{mob_siguiente}"] = factor

    return factores


def calcular_mack_diagnostico(matriz):
    """
    Calcula diagnóstico Mack Chain Ladder: intervalos de confianza,
    variabilidad de factores por transición, y test de estabilidad.

    Returns:
        dict con claves:
        - 'mack_std_error': error estándar Mack por cohorte (Series)
        - 'factor_variability': CV de factores por transición (dict)
        - 'n_cohortes_por_transicion': cantidad de cohortes usadas por transición
        - 'factores_individuales': factores por cohorte y transición (DataFrame)
    """
    mob_cols = sorted(
        [c for c in matriz.columns if c.startswith("MOB_")],
        key=lambda c: int(c.replace("MOB_", ""))
    )
    mob_nums = [int(c.replace("MOB_", "")) for c in mob_cols]

    # Factores individuales por cohorte
    factores_ind = pd.DataFrame(index=matriz.index)
    for i in range(len(mob_nums) - 1):
        m_act, m_sig = mob_nums[i], mob_nums[i + 1]
        col_a, col_s = f"MOB_{m_act}", f"MOB_{m_sig}"
        mask = matriz[col_a].notna() & matriz[col_s].notna() & (matriz[col_a] > 0)
        trans = f"{m_act}->{m_sig}"
        factores_ind[trans] = np.nan
        for idx in matriz.index[mask]:
            factores_ind.loc[idx, trans] = matriz.loc[idx, col_s] / matriz.loc[idx, col_a]

    # Variabilidad (CV) y conteo por transición
    factor_var = {}
    n_cohortes = {}
    for col in factores_ind.columns:
        vals = factores_ind[col].dropna()
        n_cohortes[col] = len(vals)
        if len(vals) >= 2:
            factor_var[col] = vals.std() / vals.mean() if vals.mean() != 0 else np.nan
        else:
            factor_var[col] = np.nan

    return {
        "factor_variability": factor_var,
        "n_cohortes_por_transicion": n_cohortes,
        "factores_individuales": factores_ind,
    }


def proyectar_triangulo(matriz, factores, mob_objetivo):
    """
    Completa el triangulo inferior aplicando los factores Chain Ladder.

    Returns:
        proyectada: DataFrame con valores observados + proyectados
        marcadores: DataFrame booleano (True = celda proyectada)
    """
    cols_objetivo = [f"MOB_{m}" for m in range(1, mob_objetivo + 1)]
    # Asegurar que existan todas las columnas hasta MOB_OBJETIVO
    for col in cols_objetivo:
        if col not in matriz.columns:
            matriz[col] = float("nan")

    proyectada = matriz[cols_objetivo].copy()
    marcadores = pd.DataFrame(False, index=matriz.index, columns=cols_objetivo)

    for cohorte in proyectada.index:
        valores_obs = proyectada.loc[cohorte].dropna()
        if len(valores_obs) == 0:
            continue

        ultimo_mob_obs = max(int(c.replace("MOB_", "")) for c in valores_obs.index)

        for mob in range(ultimo_mob_obs + 1, mob_objetivo + 1):
            transicion = f"{mob - 1}->{mob}"
            if transicion in factores:
                col_anterior = f"MOB_{mob - 1}"
                col_actual = f"MOB_{mob}"
                valor_anterior = proyectada.loc[cohorte, col_anterior]
                proyectada.loc[cohorte, col_actual] = valor_anterior * factores[transicion]
                marcadores.loc[cohorte, col_actual] = True

    return proyectada, marcadores


def main():
    print("=" * 60)
    print("Proyeccion Chain Ladder")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontro: {INPUT_FILE}")
        print("Ejecuta primero: py src/generar_matriz_vintage.py")
        sys.exit(1)

    matriz = pd.read_csv(INPUT_FILE, sep=";", decimal=",", index_col=0)
    print(f"Matriz leida: {matriz.shape[0]} cohortes x {matriz.shape[1]} MOBs")
    print(f"MOB objetivo: {MOB_OBJETIVO}")

    # 1. Calcular factores promedio ponderados
    factores = calcular_factores_chainladder(matriz, MOB_OBJETIVO)
    print(f"\nFactores Chain Ladder ({len(factores)} transiciones):")
    for trans, factor in factores.items():
        print(f"  {trans}: {factor:.6f}")

    # Guardar factores
    df_factores = pd.DataFrame([factores])
    df_factores.to_csv(OUTPUT_FACTORES, sep=";", decimal=",", index=False)
    print(f"\nFactores guardados: {OUTPUT_FACTORES}")

    # 2. Proyectar triangulo
    proyectada, marcadores = proyectar_triangulo(matriz, factores, MOB_OBJETIVO)

    n_proyectadas = marcadores.sum().sum()
    n_observadas = (~marcadores).sum().sum() - proyectada.isna().sum().sum()
    print(f"\nCeldas observadas: {int(n_observadas)}")
    print(f"Celdas proyectadas: {int(n_proyectadas)}")

    # Guardar matriz proyectada
    proyectada.to_csv(OUTPUT_MATRIZ, sep=";", decimal=",")
    print(f"Matriz proyectada guardada: {OUTPUT_MATRIZ}")

    # Guardar marcadores
    marcadores.to_csv(OUTPUT_MARCADORES, sep=";")
    print(f"Marcadores guardados: {OUTPUT_MARCADORES}")

    # 3. Diagnóstico Mack: variabilidad de factores
    diag = calcular_mack_diagnostico(matriz)
    print(f"\nDiagnóstico de estabilidad de factores CL:")
    print(f"  {'Transición':>10s} | {'Factor':>8s} | {'CV':>8s} | {'N cohortes':>10s} | {'Estable?':>8s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*10} | {'-'*8}")
    for trans in sorted(factores.keys(), key=lambda x: int(x.split("->")[0])):
        cv = diag["factor_variability"].get(trans, np.nan)
        n = diag["n_cohortes_por_transicion"].get(trans, 0)
        estable = "OK" if (not np.isnan(cv) and cv < 0.3 and n >= 5) else "REVISAR"
        print(f"  {trans:>10s} | {factores[trans]:>8.4f} | {cv:>7.3f}  | {n:>10d} | {estable:>8s}")

    # Guardar diagnóstico
    diag_path = os.path.join(PROCESSED_DATA_DIR, "diagnostico_cl.csv")
    diag_rows = []
    for trans in sorted(factores.keys(), key=lambda x: int(x.split("->")[0])):
        cv = diag["factor_variability"].get(trans, np.nan)
        n = diag["n_cohortes_por_transicion"].get(trans, 0)
        diag_rows.append({
            "transicion": trans,
            "factor_cl": factores[trans],
            "cv_factores": cv,
            "n_cohortes": n,
            "estable": cv < 0.3 and n >= 5 if not np.isnan(cv) else False,
        })
    pd.DataFrame(diag_rows).to_csv(diag_path, sep=";", decimal=",", index=False)
    print(f"\nDiagnóstico guardado: {diag_path}")

    # Vista previa
    print(f"\nVista previa (primeras 8 columnas):")
    print(proyectada.iloc[:, :8].to_string(float_format="{:.6f}".format))
    print("=" * 60)


if __name__ == "__main__":
    main()
