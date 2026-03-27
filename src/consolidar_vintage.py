"""
Script de consolidacion de datos para Análisis Vintage de mora crediticia.

Lee el archivo CSV único de otorgamientos desde data/raw/, normaliza las
columnas al esquema estándar del proyecto y genera un archivo listo para
análisis vintage en data/processed/.

El archivo de entrada contiene todas las cohortes (mes_otorgamiento) en un
solo CSV, con la evolución mes a mes de los montos vencidos, pagados,
morosos y el índice de morosidad.

Compatible con múltiples empresas (OC, CA, etc.). El script detecta
automáticamente las columnas presentes y las normaliza al esquema estándar:
  - mes_otorgamiento -> cohorte
  - cantidad_creditos / cantidad_operaciones -> cantidad_operaciones
  - Descarta columnas no necesarias (ft_clave_con_ven, desc_tie_mes)
  - Limpia formatos de moneda ($, puntos de miles) en columnas numéricas

El archivo de salida agrega la columna 'mob' (months on books), que indica
la cantidad de meses transcurridos desde el otorgamiento de la cohorte,
dato esencial para construir la matriz vintage.

Uso:
    python src/consolidar_vintage.py
"""

import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración de rutas
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_FILE = os.path.join(RAW_DATA_DIR, "CA otorgamientos 2022-2025.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "vintage_consolidado.csv")

# ---------------------------------------------------------------------------
# Columnas esperadas en el archivo de salida (esquema estándar del proyecto)
# ---------------------------------------------------------------------------
COLUMNAS_SALIDA = [
    "cohorte", "id_tie_mes", "total_vencido", "total_pagado",
    "moroso", "indice", "cantidad_operaciones", "mob",
]

# Columnas numéricas que deben ser float (pueden venir con formato moneda)
COLUMNAS_NUMERICAS = ["total_vencido", "total_pagado", "moroso", "indice"]


def limpiar_moneda(valor):
    """
    Convierte un valor con formato moneda a float.

    Maneja formatos como:
      - '$ 3.314.894.053,09' -> 3314894053.09
      - '1041160872' -> 1041160872.0
      - 186376710.0 -> 186376710.0 (ya numérico, pasa sin cambios)
    """
    if isinstance(valor, (int, float)):
        return float(valor)
    if not isinstance(valor, str):
        return valor
    # Quitar símbolo de moneda y espacios
    s = re.sub(r'[$\s]', '', valor)
    # Si tiene formato con puntos de miles y coma decimal (ej: 3.314.894.053,09)
    if '.' in s and ',' in s:
        s = s.replace('.', '').replace(',', '.')
    # Si solo tiene coma decimal (ej: 186376710,5)
    elif ',' in s:
        s = s.replace(',', '.')
    return float(s)


def leer_csv_otorgamientos(filepath: str) -> pd.DataFrame:
    """Lee el CSV único de otorgamientos con separador ';' y decimales con ','."""
    df = pd.read_csv(filepath, sep=";", decimal=",", encoding="utf-8")
    # Limpiar espacios en nombres de columnas y valores de texto
    df.columns = df.columns.str.strip()
    # Limpiar espacios en columnas de texto
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza columnas al esquema estándar del proyecto.

    Mapeos:
        mes_otorgamiento -> cohorte
        cantidad_creditos -> cantidad_operaciones (alias CA)

    Descarta columnas no necesarias para el análisis vintage:
        ft_clave_con_ven (identificador interno OC)
        desc_tie_mes (nombre del mes en texto, CA)
    """
    renombrar = {
        "mes_otorgamiento": "cohorte",
        "cantidad_creditos": "cantidad_operaciones",
    }
    df = df.rename(columns={k: v for k, v in renombrar.items() if k in df.columns})

    # Descartar columnas innecesarias para el análisis
    columnas_descartar = ["ft_clave_con_ven", "desc_tie_mes"]
    df = df.drop(columns=[c for c in columnas_descartar if c in df.columns])

    return df


def normalizar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que las columnas numéricas sean float.

    Limpia formatos de moneda ($, puntos de miles) que pueden aparecer
    en algunas filas del CSV fuente.
    """
    for col in COLUMNAS_NUMERICAS:
        if col not in df.columns:
            continue
        if df[col].dtype == object:
            print(f"  Limpiando formato moneda en columna '{col}'")
            df[col] = df[col].apply(limpiar_moneda)
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def calcular_mob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula months on books (MOB): meses transcurridos desde la cohorte.

    MOB = 1 para el primer mes posterior al otorgamiento, 2 para el segundo, etc.
    """
    cohorte_dt = pd.to_datetime(df["cohorte"], format="%Y-%m")
    mes_obs_dt = pd.to_datetime(df["id_tie_mes"], format="%Y-%m")
    df["mob"] = ((mes_obs_dt.dt.year - cohorte_dt.dt.year) * 12
                 + (mes_obs_dt.dt.month - cohorte_dt.dt.month))
    return df


def consolidar() -> pd.DataFrame:
    """
    Lee el CSV único de data/raw/, normaliza columnas y agrega MOB.

    Returns:
        DataFrame consolidado con todas las cohortes y la columna MOB.
    """
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encontró el archivo: {INPUT_FILE}")
        sys.exit(1)

    print(f"Leyendo archivo: {os.path.basename(INPUT_FILE)}")

    df = leer_csv_otorgamientos(INPUT_FILE)
    print(f"  {len(df)} filas leídas")

    # Normalizar al esquema estándar del proyecto
    df = normalizar_columnas(df)
    print(f"  Columnas normalizadas: {list(df.columns)}")

    # Asegurar tipos numéricos (limpiar formatos de moneda si existen)
    df = normalizar_numericas(df)

    # Calcular MOB
    df = calcular_mob(df)

    # Ordenar por cohorte y luego por MOB
    df = df.sort_values(["cohorte", "mob"]).reset_index(drop=True)

    return df


def main():
    print("=" * 60)
    print("Consolidación de datos para Análisis Vintage")
    print("=" * 60)

    df = consolidar()

    # Crear directorio de salida si no existe
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Guardar con separador ';' para mantener consistencia con el archivo fuente
    df.to_csv(OUTPUT_FILE, sep=";", index=False, decimal=",")

    print(f"\nArchivo consolidado generado: {OUTPUT_FILE}")
    print(f"Total de filas: {len(df)}")
    print(f"Cohortes encontradas: {sorted(df['cohorte'].unique())}")
    print(f"Rango de MOB: {df['mob'].min()} - {df['mob'].max()}")
    print(f"\nColumnas del archivo de salida:")
    print(f"  {list(df.columns)}")
    print(f"\nVista previa (primeras 5 filas):")
    print(df.head().to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
