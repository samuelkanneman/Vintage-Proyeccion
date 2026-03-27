"""
Consolidación de datos vintage segmentados por nivel de riesgo.

Lee el CSV de otorgamientos con columnas de segmentación (tipoope,
clientenuevo, nivelriesgo), agrupa los niveles en 3 categorías de riesgo
(Bajo, Medio, Alto) y genera un consolidado por segmento.

Segmentos resultantes (12 combinaciones):
  - tipoope: CC, PP
  - clientenuevo: V, F
  - nivel_riesgo: Bajo, Medio, Alto

Reglas de agrupación de nivelriesgo:
  - Bajo:  todo lo que empieza con "1 Bajo" (1 BajoA, 1 BajoB, 1 BajoM, 1 BajoAA, etc.)
  - Medio: todo lo que empieza con "2 Medio" (2 MedioA, 2 MedioB, 2 MedioM, etc.)
  - Alto:  "3 Alto" y "4 MuyAlto"
  - Excluidos: "0 Bajo", "Error", NaN/blancos

Uso:
    python src/consolidar_vintage_niveles.py
"""

import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(RAW_DATA_DIR, "CA otorgamientos 2022-2025 (niveles).csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "vintage_niveles_consolidado.csv")

# Columnas numéricas que se suman al agrupar
COLUMNAS_SUMA = ["cantidad_operaciones", "total_vencido", "total_pagado", "moroso"]


def clasificar_nivel(valor):
    """
    Clasifica un valor de nivelriesgo en Bajo, Medio o Alto.

    Returns None para valores excluidos (0 Bajo, Error, NaN, blancos).
    """
    if pd.isna(valor) or not isinstance(valor, str):
        return None
    valor = valor.strip()
    if valor == "" or valor.lower() == "error":
        return None
    if valor.startswith("0 "):
        return None
    if valor.startswith("1 Bajo"):
        return "Bajo"
    if valor.startswith("2 Medio"):
        return "Medio"
    if valor in ("3 Alto", "4 MuyAlto"):
        return "Alto"
    return None


def consolidar():
    """
    Lee el CSV, clasifica niveles, agrupa y calcula MOB.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró el archivo: {INPUT_FILE}")
        sys.exit(1)

    print(f"Leyendo archivo: {os.path.basename(INPUT_FILE)}")
    df = pd.read_csv(INPUT_FILE, sep=";", decimal=",", encoding="utf-8")
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    print(f"  {len(df)} filas leídas")

    # Renombrar columnas al esquema estándar
    df = df.rename(columns={
        "mes_otorgamiento": "cohorte",
        "mes_mora": "id_tie_mes",
        "cantidad_creditos": "cantidad_operaciones",
    })

    # Clasificar nivel de riesgo
    df["nivel_riesgo"] = df["nivelriesgo"].apply(clasificar_nivel)

    # Estadísticas de clasificación
    total = len(df)
    excluidos = df["nivel_riesgo"].isna().sum()
    print(f"  Filas excluidas (0 Bajo, Error, NaN): {excluidos} ({excluidos/total*100:.1f}%)")
    print(f"  Distribucion de niveles originales -> agrupados:")
    for nivel_orig, grupo in df.groupby("nivelriesgo", dropna=False):
        nivel_agrup = grupo["nivel_riesgo"].iloc[0]
        print(f"    {nivel_orig!r:20s} -> {nivel_agrup!r:10s}  ({len(grupo)} filas)")

    # Filtrar excluidos
    df = df[df["nivel_riesgo"].notna()].copy()

    # Asegurar tipos numéricos
    for col in COLUMNAS_SUMA:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Agrupar por cohorte, id_tie_mes, tipoope, clientenuevo, nivel_riesgo
    # sumando montos y cantidad de operaciones
    df_agrup = (
        df.groupby(["cohorte", "id_tie_mes", "tipoope", "clientenuevo", "nivel_riesgo"], as_index=False)
        [COLUMNAS_SUMA]
        .sum()
    )

    # Recalcular índice después de agrupar
    df_agrup["indice"] = df_agrup["moroso"] / df_agrup["total_vencido"]
    df_agrup["indice"] = df_agrup["indice"].fillna(0)

    # Calcular MOB
    cohorte_dt = pd.to_datetime(df_agrup["cohorte"], format="%Y-%m")
    mes_obs_dt = pd.to_datetime(df_agrup["id_tie_mes"], format="%Y-%m")
    df_agrup["mob"] = ((mes_obs_dt.dt.year - cohorte_dt.dt.year) * 12
                       + (mes_obs_dt.dt.month - cohorte_dt.dt.month))

    # Crear columna de segmento para facilitar filtros
    df_agrup["segmento"] = (df_agrup["tipoope"] + "_"
                            + df_agrup["clientenuevo"] + "_"
                            + df_agrup["nivel_riesgo"])

    # Ordenar
    df_agrup = df_agrup.sort_values(
        ["tipoope", "clientenuevo", "nivel_riesgo", "cohorte", "mob"]
    ).reset_index(drop=True)

    return df_agrup


def main():
    print("=" * 60)
    print("Consolidación Vintage por Niveles de Riesgo")
    print("=" * 60)

    df = consolidar()

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, sep=";", index=False, decimal=",")

    print(f"\nArchivo generado: {OUTPUT_FILE}")
    print(f"Total de filas: {len(df)}")
    print(f"Cohortes: {sorted(df['cohorte'].unique())}")
    print(f"Segmentos: {sorted(df['segmento'].unique())}")
    print(f"Rango MOB: {df['mob'].min()} - {df['mob'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
