"""
Genera cohortes sinteticas (futuras) por segmento (tipoope x clientenuevo x nivel_riesgo).

Para cada uno de los 12 segmentos:
1. Estima el MOB 1 futuro con tres metodos (regresion, WMA, ARIMA)
2. Construye la curva completa aplicando factores CL del segmento

Tambien genera una proyeccion general ponderada por operaciones.

Lee:
  - data/processed/matrices_vintage_niveles.csv
  - data/processed/factores_cl_niveles.csv
  - data/processed/vintage_niveles_consolidado.csv (para pesos)

Genera:
  - data/processed/sinteticas_niveles.csv (matriz por segmento)
  - data/processed/regresion_mob1_niveles.csv (estimacion MOB 1 por segmento)
  - data/processed/sinteticas_general_niveles.csv (promedio ponderado)

Uso:
    python src/generar_cohortes_sinteticas_niveles.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_MATRICES = os.path.join(PROCESSED_DIR, "matrices_vintage_niveles.csv")
INPUT_FACTORES = os.path.join(PROCESSED_DIR, "factores_cl_niveles.csv")
INPUT_CONSOLIDADO = os.path.join(PROCESSED_DIR, "vintage_niveles_consolidado.csv")

OUTPUT_SINTETICAS = os.path.join(PROCESSED_DIR, "sinteticas_niveles.csv")
OUTPUT_REGRESION = os.path.join(PROCESSED_DIR, "regresion_mob1_niveles.csv")
OUTPUT_GENERAL = os.path.join(PROCESSED_DIR, "sinteticas_general_niveles.csv")

# ---------------------------------------------------------------------------
# Parametros
# ---------------------------------------------------------------------------
MOB_MADURO = 15
N_COHORTES_FUTURAS = 3
N_COHORTES_REGRESION = 12
N_MEDIA_MOVIL = 6
ARIMA_ORDER = (1, 1, 1)
SEASONAL_ORDER = None
N_ARIMA = 12
METODO_SINTETICAS = "wma"

# ---------------------------------------------------------------------------
# Segmentos deshabilitados: no se venden actualmente, sinteticas forzadas a 0
# ---------------------------------------------------------------------------
SEGMENTOS_DESHABILITADOS = [
    "CC_V_Medio",
    "CC_V_Alto",
    "PP_V_Medio",
    "PP_V_Alto",
]


def siguiente_cohorte(cohorte_str):
    year, month = map(int, cohorte_str.split("-"))
    month += 1
    if month > 12:
        month = 1
        year += 1
    return f"{year:04d}-{month:02d}"


def estimar_mob1_regresion(mob1_series, n_regresion, n_futuras):
    """Regresion lineal sobre MOB 1."""
    cohortes_ord = list(mob1_series.index)
    cohortes_reg = cohortes_ord[-n_regresion:]
    x_reg = np.arange(len(cohortes_reg))
    y_reg = mob1_series.loc[cohortes_reg].values

    coefs = np.polyfit(x_reg, y_reg, 1)
    poly = np.poly1d(coefs)

    residuos = y_reg - poly(x_reg)
    std_reg = np.std(residuos, ddof=1) if len(residuos) > 1 else 0.0

    filas = []
    for c in cohortes_ord:
        en_reg = c in cohortes_reg
        idx = cohortes_reg.index(c) if en_reg else None
        filas.append({
            "cohorte": c,
            "mob1_observado": mob1_series[c],
            "mob1_regresion": poly(idx) if idx is not None else None,
            "mob1_reg_sup": (poly(idx) + std_reg) if idx is not None else None,
            "mob1_reg_inf": (poly(idx) - std_reg) if idx is not None else None,
            "tipo": "historica",
            "en_regresion": en_reg,
        })

    ultima = cohortes_ord[-1]
    for j in range(1, n_futuras + 1):
        ultima = siguiente_cohorte(ultima)
        val = poly(len(cohortes_reg) - 1 + j)
        filas.append({
            "cohorte": ultima,
            "mob1_observado": None,
            "mob1_regresion": val,
            "mob1_reg_sup": val + std_reg,
            "mob1_reg_inf": val - std_reg,
            "tipo": "sintetica",
            "en_regresion": False,
        })

    return pd.DataFrame(filas), std_reg


def estimar_mob1_wma(mob1_series, n_ventana, n_futuras):
    """Media movil ponderada exponencial."""
    cohortes_ord = list(mob1_series.index)
    cohortes_ventana = cohortes_ord[-n_ventana:]
    mob1_ventana = mob1_series.loc[cohortes_ventana].values

    pesos = np.exp(np.linspace(0, 2, n_ventana))
    pesos = pesos / pesos.sum()
    wma_actual = np.average(mob1_ventana, weights=pesos)

    diffs = np.diff(mob1_ventana)
    pesos_diff = np.exp(np.linspace(0, 2, len(diffs)))
    pesos_diff = pesos_diff / pesos_diff.sum()
    tendencia = np.average(diffs, weights=pesos_diff)

    residuos_wma = []
    for i in range(1, len(mob1_ventana)):
        sub = mob1_ventana[:i]
        w = np.exp(np.linspace(0, 2, len(sub)))
        w = w / w.sum()
        pred = np.average(sub, weights=w)
        residuos_wma.append(mob1_ventana[i] - pred)
    std_wma = np.std(residuos_wma, ddof=1) if len(residuos_wma) > 1 else 0.0

    filas = []
    for c in cohortes_ord:
        filas.append({"cohorte": c, "mob1_wma": None, "mob1_wma_sup": None,
                      "mob1_wma_inf": None, "tipo": "historica"})

    ultima = cohortes_ord[-1]
    for j in range(1, n_futuras + 1):
        ultima = siguiente_cohorte(ultima)
        val = wma_actual + tendencia * j
        filas.append({
            "cohorte": ultima,
            "mob1_wma": val,
            "mob1_wma_sup": val + std_wma,
            "mob1_wma_inf": val - std_wma,
            "tipo": "sintetica",
        })

    return pd.DataFrame(filas), std_wma


def estimar_mob1_arima(mob1_series, n_arima, n_futuras, order=(1,1,1),
                       seasonal_order=None):
    """ARIMA con selección automática de orden (auto_arima) sobre MOB 1."""
    datos = mob1_series.iloc[-n_arima:].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if len(datos) >= 8:
                resultado = auto_arima(
                    datos, start_p=0, max_p=3, start_q=0, max_q=3,
                    d=None, max_d=2, seasonal=False, stepwise=True,
                    suppress_warnings=True, error_action="ignore",
                    information_criterion="aic",
                )
            else:
                order_use = (1, 0, 0) if len(datos) < 5 else order
                modelo = ARIMA(datos, order=order_use)
                resultado = modelo.fit()
        except Exception:
            try:
                modelo = ARIMA(datos, order=order)
                resultado = modelo.fit()
            except Exception:
                modelo = ARIMA(datos, order=(1, 0, 0))
                resultado = modelo.fit()

    if hasattr(resultado, 'predict'):
        forecast = resultado.predict(n_periods=n_futuras)
    else:
        forecast = resultado.forecast(steps=n_futuras)

    residuos = resultado.resid() if callable(getattr(resultado, 'resid', None)) else resultado.resid
    std_arima = np.std(residuos, ddof=1)

    cohortes_ord = list(mob1_series.index)
    filas = []
    for c in cohortes_ord:
        filas.append({"cohorte": c, "mob1_arima": None, "mob1_arima_sup": None,
                      "mob1_arima_inf": None, "tipo": "historica"})

    ultima = cohortes_ord[-1]
    base_std = std_arima
    for j in range(n_futuras):
        ultima = siguiente_cohorte(ultima)
        val = float(forecast[j])
        h_std = base_std * np.sqrt(j + 1)  # incertidumbre crece con horizonte
        filas.append({
            "cohorte": ultima,
            "mob1_arima": val,
            "mob1_arima_sup": val + h_std,
            "mob1_arima_inf": val - h_std,
            "tipo": "sintetica",
        })

    return pd.DataFrame(filas), std_arima


def construir_sinteticas(df_reg, factores_cl, mob_maduro, columna_mob1="mob1_wma"):
    """Construye curvas sinteticas aplicando factores CL al MOB 1 estimado."""
    sinteticas = df_reg[df_reg["tipo"] == "sintetica"].copy()
    filas = []
    for _, row in sinteticas.iterrows():
        cohorte = row["cohorte"]
        mob1 = row[columna_mob1]
        if pd.isna(mob1):
            continue
        # Clamp MOB 1 al rango [0.001, 1.0]
        mob1 = max(0.001, min(1.0, mob1))
        valores = {"cohorte": cohorte, "MOB_1": mob1}
        valor_actual = mob1
        for mob_n in range(1, mob_maduro):
            trans = f"{mob_n}->{mob_n + 1}"
            if trans in factores_cl:
                valor_actual = valor_actual * factores_cl[trans]
                valores[f"MOB_{mob_n + 1}"] = valor_actual
            else:
                break
        filas.append(valores)
    return pd.DataFrame(filas).set_index("cohorte") if filas else pd.DataFrame()


def main():
    print("=" * 60)
    print("Cohortes Sinteticas por Niveles de Riesgo")
    print("=" * 60)

    if not os.path.exists(INPUT_MATRICES):
        print(f"[ERROR] No se encontro: {INPUT_MATRICES}")
        sys.exit(1)

    matrices_df = pd.read_csv(INPUT_MATRICES, sep=";", decimal=",")
    factores_df = pd.read_csv(INPUT_FACTORES, sep=";", decimal=",")
    df_consolidado = pd.read_csv(INPUT_CONSOLIDADO, sep=";", decimal=",")

    segmentos = sorted(matrices_df["segmento"].unique())
    print(f"Segmentos: {len(segmentos)}")
    print(f"Parametros: MOB_MADURO={MOB_MADURO}, N_FUTURAS={N_COHORTES_FUTURAS}, "
          f"METODO={METODO_SINTETICAS}")
    if SEGMENTOS_DESHABILITADOS:
        print(f"Segmentos deshabilitados (forzados a 0): {SEGMENTOS_DESHABILITADOS}")

    col_mob1_map = {"wma": "mob1_wma", "arima": "mob1_arima", "regresion": "mob1_regresion"}
    col_mob1 = col_mob1_map.get(METODO_SINTETICAS, "mob1_wma")

    all_sinteticas = []
    all_regresion = []
    all_sinteticas_dict = {}

    for segmento in segmentos:
        sub = matrices_df[matrices_df["segmento"] == segmento].copy()
        mob_cols = [c for c in sub.columns if c.startswith("MOB_")]
        matriz = sub.set_index("cohorte")[mob_cols]
        cohortes_ord = sorted(matriz.index)

        # Segmentos deshabilitados: sinteticas con todos los MOBs en 0
        if segmento in SEGMENTOS_DESHABILITADOS:
            ultima = cohortes_ord[-1]
            filas_sint = []
            filas_reg = []
            for j in range(1, N_COHORTES_FUTURAS + 1):
                ultima = siguiente_cohorte(ultima)
                fila_s = {"cohorte": ultima}
                for m in range(1, MOB_MADURO + 1):
                    fila_s[f"MOB_{m}"] = 0.0
                filas_sint.append(fila_s)
                filas_reg.append({
                    "cohorte": ultima, "mob1_observado": None,
                    "mob1_regresion": 0.0, "mob1_reg_sup": 0.0, "mob1_reg_inf": 0.0,
                    "mob1_wma": 0.0, "mob1_wma_sup": 0.0, "mob1_wma_inf": 0.0,
                    "mob1_arima": 0.0, "mob1_arima_sup": 0.0, "mob1_arima_inf": 0.0,
                    "tipo": "sintetica", "en_regresion": False, "segmento": segmento,
                })
            # Agregar historicas a regresion
            for c in cohortes_ord:
                mob1_val = matriz.loc[c, "MOB_1"] if "MOB_1" in matriz.columns else None
                filas_reg.insert(0, {
                    "cohorte": c, "mob1_observado": mob1_val,
                    "mob1_regresion": None, "mob1_reg_sup": None, "mob1_reg_inf": None,
                    "mob1_wma": None, "mob1_wma_sup": None, "mob1_wma_inf": None,
                    "mob1_arima": None, "mob1_arima_sup": None, "mob1_arima_inf": None,
                    "tipo": "historica", "en_regresion": False, "segmento": segmento,
                })
            df_sint = pd.DataFrame(filas_sint).set_index("cohorte")
            df_sint["segmento"] = segmento
            all_sinteticas.append(df_sint.reset_index())
            all_sinteticas_dict[segmento] = df_sint.drop(columns=["segmento"])
            all_regresion.append(pd.DataFrame(filas_reg))
            print(f"  {segmento}: DESHABILITADO -> sinteticas forzadas a 0")
            continue

        if "MOB_1" not in matriz.columns or matriz["MOB_1"].dropna().shape[0] < 4:
            print(f"  {segmento}: insuficientes datos MOB_1, saltando")
            continue

        mob1_series = matriz.loc[cohortes_ord, "MOB_1"].dropna()

        # Factores CL del segmento
        row_f = factores_df[factores_df["segmento"] == segmento]
        factores_cl = {}
        if not row_f.empty:
            for col in row_f.columns:
                if "->" in col:
                    val = row_f[col].iloc[0]
                    if pd.notna(val):
                        factores_cl[col] = float(val)

        # Ajustar ventanas al tamano disponible
        n_reg = min(N_COHORTES_REGRESION, len(mob1_series))
        n_wma = min(N_MEDIA_MOVIL, len(mob1_series))
        n_arima = min(N_ARIMA, len(mob1_series))

        # Estimar MOB 1
        df_reg, _ = estimar_mob1_regresion(mob1_series, n_reg, N_COHORTES_FUTURAS)

        df_wma, _ = estimar_mob1_wma(mob1_series, n_wma, N_COHORTES_FUTURAS)

        try:
            order = ARIMA_ORDER
            if n_arima < 5:
                order = (1, 0, 0)
            df_arima, _ = estimar_mob1_arima(mob1_series, n_arima, N_COHORTES_FUTURAS,
                                              order, SEASONAL_ORDER)
        except Exception:
            df_arima = df_wma.copy()
            df_arima = df_arima.rename(columns={"mob1_wma": "mob1_arima",
                                                 "mob1_wma_sup": "mob1_arima_sup",
                                                 "mob1_wma_inf": "mob1_arima_inf"})

        # Combinar
        df_combined = df_reg.copy()
        for col in ["mob1_wma", "mob1_wma_sup", "mob1_wma_inf"]:
            df_combined[col] = df_wma[col].values
        for col in ["mob1_arima", "mob1_arima_sup", "mob1_arima_inf"]:
            df_combined[col] = df_arima[col].values
        df_combined["segmento"] = segmento

        all_regresion.append(df_combined)

        # Construir sinteticas
        sinteticas = construir_sinteticas(df_combined, factores_cl, MOB_MADURO, col_mob1)
        if not sinteticas.empty:
            sinteticas["segmento"] = segmento
            all_sinteticas.append(sinteticas.reset_index())
            all_sinteticas_dict[segmento] = sinteticas

            print(f"  {segmento}: {len(sinteticas)} cohortes sinteticas, "
                  f"MOB_1={sinteticas['MOB_1'].mean():.4f}")
        else:
            print(f"  {segmento}: no se pudieron generar sinteticas")

    # Guardar regresion
    df_reg_all = pd.concat(all_regresion, ignore_index=True)
    df_reg_all.to_csv(OUTPUT_REGRESION, sep=";", decimal=",", index=False)
    print(f"\nRegresion MOB 1: {OUTPUT_REGRESION}")

    # Guardar sinteticas
    df_sint_all = pd.concat(all_sinteticas, ignore_index=True)
    mob_cols = ["segmento", "cohorte"] + [f"MOB_{m}" for m in range(1, MOB_MADURO + 1)
                                           if f"MOB_{m}" in df_sint_all.columns]
    df_sint_all = df_sint_all[[c for c in mob_cols if c in df_sint_all.columns]]
    df_sint_all.to_csv(OUTPUT_SINTETICAS, sep=";", decimal=",", index=False)
    print(f"Sinteticas por segmento: {OUTPUT_SINTETICAS}")

    # Proyeccion general ponderada (excluyendo segmentos que no se venden)
    print("\nGenerando sinteticas general ponderada por operaciones...")
    print(f"  Excluidos del general: {SEGMENTOS_DESHABILITADOS}")
    mob1_data = df_consolidado[df_consolidado["mob"] == 1]
    pesos = mob1_data.pivot_table(index="cohorte", columns="segmento",
                                   values="cantidad_operaciones", aggfunc="sum")

    # Cohortes sinteticas
    cohortes_sint = sorted(set(df_sint_all["cohorte"].unique()))
    mob_cols_gen = [f"MOB_{m}" for m in range(1, MOB_MADURO + 1)]
    general = pd.DataFrame(index=cohortes_sint, columns=mob_cols_gen, dtype=float)

    for cohorte in cohortes_sint:
        for col in mob_cols_gen:
            suma_pond = 0.0
            suma_pesos = 0.0
            for seg, sint_df in all_sinteticas_dict.items():
                # Excluir segmentos deshabilitados del promedio general
                if seg in SEGMENTOS_DESHABILITADOS:
                    continue
                if cohorte not in sint_df.index or col not in sint_df.columns:
                    continue
                val = sint_df.loc[cohorte, col]
                if pd.isna(val):
                    continue
                # Peso: promedio de operaciones del segmento (ultimas cohortes)
                if seg in pesos.columns:
                    peso = pesos[seg].dropna().iloc[-3:].mean() if len(pesos[seg].dropna()) >= 3 else pesos[seg].dropna().mean()
                else:
                    peso = 0
                if peso > 0:
                    suma_pond += val * peso
                    suma_pesos += peso
            if suma_pesos > 0:
                general.loc[cohorte, col] = suma_pond / suma_pesos

    general.index.name = "cohorte"
    general.to_csv(OUTPUT_GENERAL, sep=";", decimal=",")
    print(f"Sinteticas general: {OUTPUT_GENERAL}")
    print(f"  Cohortes: {list(general.index)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
