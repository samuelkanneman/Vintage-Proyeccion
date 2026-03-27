"""
Genera cohortes sintéticas (futuras) proyectando la mora mes a mes.

El proceso tiene dos pasos:
1. Estimar el MOB 1 (mora inicial) de cada cohorte futura usando tres métodos:
   a) Regresión lineal sobre los MOB 1 históricos (baseline).
   b) Media móvil ponderada exponencial (da más peso a cohortes recientes).
   c) ARIMA (modelo autorregresivo integrado de media móvil).
   Los tres métodos incluyen bandas de confianza (+/- 1 desvío estándar).
2. Construir la curva completa aplicando los factores de desarrollo Chain Ladder
   promedio ponderados al MOB 1 estimado, hasta el MOB maduro.

Genera:
- cohortes_sinteticas.csv: matriz con las cohortes proyectadas (MOB_1 a MOB_maduro)
- regresion_mob1.csv: datos de la regresión (cohortes históricas + estimadas)

Uso:
    python src/generar_cohortes_sinteticas.py
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

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_MATRIZ = os.path.join(PROCESSED_DIR, "matriz_vintage.csv")
INPUT_FACTORES = os.path.join(PROCESSED_DIR, "factores_desarrollo.csv")

OUTPUT_SINTETICAS = os.path.join(PROCESSED_DIR, "cohortes_sinteticas.csv")
OUTPUT_REGRESION = os.path.join(PROCESSED_DIR, "regresion_mob1.csv")

# ---------------------------------------------------------------------------
# Parámetros (pueden ser sobreescritos desde el notebook)
# ---------------------------------------------------------------------------
MOB_MADURO = 15           # MOB hasta el que proyectar cada cohorte sintética
N_COHORTES_FUTURAS = 3    # Cantidad de cohortes futuras a generar
N_COHORTES_REGRESION = 12 # Últimas N cohortes a usar para la regresión del MOB 1
N_MEDIA_MOVIL = 6         # Ventana de la media móvil ponderada
ARIMA_ORDER = (1, 1, 1)   # Orden ARIMA (p, d, q)
SEASONAL_ORDER = None     # Orden estacional (P, D, Q, s) o None para ARIMA sin estacionalidad
N_ARIMA = 12              # Últimas N cohortes para ajustar ARIMA/SARIMA
METODO_SINTETICAS = "wma" # Método para construir sintéticas: "regresion", "wma" o "arima"


def siguiente_cohorte(cohorte_str: str) -> str:
    """Dado '2026-01', devuelve '2026-02'. Dado '2026-12', devuelve '2027-01'."""
    year, month = map(int, cohorte_str.split("-"))
    month += 1
    if month > 12:
        month = 1
        year += 1
    return f"{year:04d}-{month:02d}"


def estimar_mob1(matriz: pd.DataFrame,
                 n_regresion: int,
                 n_futuras: int) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Estima el MOB 1 de cohortes futuras con regresion lineal.

    Usa las ultimas `n_regresion` cohortes para ajustar la recta.

    Returns:
        df_regresion: DataFrame con cohortes historicas y estimadas
        coefs: [pendiente, intercepto] de la regresion
        std_reg: desvio estandar de los residuos
    """
    cohortes_ord = sorted(matriz.index)
    mob1_all = matriz.loc[cohortes_ord, "MOB_1"]

    # Tomar últimas N para regresión
    cohortes_reg = cohortes_ord[-n_regresion:]
    mob1_reg = mob1_all.loc[cohortes_reg]

    # Índices numéricos (relativos al inicio de la regresión)
    x_reg = np.arange(len(cohortes_reg))
    y_reg = mob1_reg.values

    coefs = np.polyfit(x_reg, y_reg, 1)
    poly = np.poly1d(coefs)

    # Construir DataFrame con todo
    filas = []
    # Históricas
    for i, c in enumerate(cohortes_ord):
        en_regresion = c in cohortes_reg
        idx_reg = cohortes_reg.index(c) if en_regresion else None
        filas.append({
            "cohorte": c,
            "mob1_observado": mob1_all[c],
            "mob1_regresion": poly(idx_reg) if idx_reg is not None else None,
            "tipo": "historica",
            "en_regresion": en_regresion,
        })

    # Futuras
    ultima = cohortes_ord[-1]
    for j in range(1, n_futuras + 1):
        ultima = siguiente_cohorte(ultima)
        idx_futuro = len(cohortes_reg) - 1 + j
        mob1_est = poly(idx_futuro)
        filas.append({
            "cohorte": ultima,
            "mob1_observado": None,
            "mob1_regresion": mob1_est,
            "tipo": "sintetica",
            "en_regresion": False,
        })

    # Calcular desvío estándar de residuos de la regresión
    residuos_reg = y_reg - poly(x_reg)
    std_reg = np.std(residuos_reg, ddof=1)

    # Agregar bandas +/- 1 std a las filas
    for fila in filas:
        if fila["mob1_regresion"] is not None:
            fila["mob1_reg_sup"] = fila["mob1_regresion"] + std_reg
            fila["mob1_reg_inf"] = fila["mob1_regresion"] - std_reg
        else:
            fila["mob1_reg_sup"] = None
            fila["mob1_reg_inf"] = None

    return pd.DataFrame(filas), coefs, std_reg


def estimar_mob1_wma(matriz: pd.DataFrame,
                     n_ventana: int,
                     n_futuras: int) -> tuple[pd.DataFrame, float]:
    """
    Estima el MOB 1 de cohortes futuras con media movil ponderada exponencial.

    Los pesos decrecen exponencialmente: la cohorte mas reciente tiene el peso
    mas alto. Esto captura mejor cambios de tendencia recientes.

    Returns:
        df_wma: DataFrame con cohortes historicas y estimadas (col mob1_wma)
        std_wma: desvio estandar de los residuos dentro de la ventana
    """
    cohortes_ord = sorted(matriz.index)
    mob1_all = matriz.loc[cohortes_ord, "MOB_1"]

    # Ventana: ultimas n_ventana cohortes
    cohortes_ventana = cohortes_ord[-n_ventana:]
    mob1_ventana = mob1_all.loc[cohortes_ventana].values

    # Pesos exponenciales (mas peso a las recientes)
    pesos = np.exp(np.linspace(0, 2, n_ventana))
    pesos = pesos / pesos.sum()

    # Media movil ponderada del ultimo punto
    wma_actual = np.average(mob1_ventana, weights=pesos)

    # Tendencia ponderada (diferencias consecutivas ponderadas)
    diffs = np.diff(mob1_ventana)
    pesos_diff = np.exp(np.linspace(0, 2, len(diffs)))
    pesos_diff = pesos_diff / pesos_diff.sum()
    tendencia = np.average(diffs, weights=pesos_diff)

    # Desvio estandar de los residuos (WMA rolling dentro de la ventana)
    residuos_wma = []
    for i in range(1, len(mob1_ventana)):
        sub = mob1_ventana[:i]
        w = np.exp(np.linspace(0, 2, len(sub)))
        w = w / w.sum()
        pred = np.average(sub, weights=w)
        residuos_wma.append(mob1_ventana[i] - pred)
    std_wma = np.std(residuos_wma, ddof=1) if len(residuos_wma) > 1 else 0.0

    # WMA historica (rolling) para las cohortes en la ventana
    wma_historico = {}
    for i, c in enumerate(cohortes_ord):
        if c in cohortes_ventana:
            idx_v = cohortes_ventana.index(c)
            if idx_v >= 2:
                sub = mob1_ventana[:idx_v + 1]
                w = np.exp(np.linspace(0, 2, len(sub)))
                w = w / w.sum()
                wma_historico[c] = np.average(sub, weights=w)

    # Construir filas
    filas = []
    for c in cohortes_ord:
        wma_val = wma_historico.get(c, None)
        filas.append({
            "cohorte": c,
            "mob1_wma": wma_val,
            "mob1_wma_sup": wma_val + std_wma if wma_val is not None else None,
            "mob1_wma_inf": wma_val - std_wma if wma_val is not None else None,
            "tipo": "historica",
        })

    # Futuras
    ultima = cohortes_ord[-1]
    for j in range(1, n_futuras + 1):
        ultima = siguiente_cohorte(ultima)
        mob1_est = wma_actual + tendencia * j
        filas.append({
            "cohorte": ultima,
            "mob1_wma": mob1_est,
            "mob1_wma_sup": mob1_est + std_wma,
            "mob1_wma_inf": mob1_est - std_wma,
            "tipo": "sintetica",
        })

    return pd.DataFrame(filas), std_wma


def estimar_mob1_arima(matriz: pd.DataFrame,
                       n_arima: int,
                       n_futuras: int,
                       order: tuple = (1, 1, 1),
                       seasonal_order: tuple = None) -> tuple[pd.DataFrame, float, dict]:
    """
    Estima el MOB 1 de cohortes futuras con auto_arima (selección automática
    del mejor orden por AIC/BIC).

    Fallback al orden manual si auto_arima falla o si hay muy pocos datos.

    Returns:
        df_arima: DataFrame con cohortes historicas y estimadas (col mob1_arima)
        std_arima: desvio estandar de los residuos del modelo
        diagnostico: dict con AIC, BIC, orden seleccionado, test ADF
    """
    cohortes_ord = sorted(matriz.index)
    mob1_all = matriz.loc[cohortes_ord, "MOB_1"]

    # Datos para ARIMA
    datos_arima = mob1_all.iloc[-n_arima:].values
    diagnostico = {}

    # Test ADF de estacionariedad
    if len(datos_arima) >= 8:
        try:
            adf_stat, adf_pval, _, _, _, _ = adfuller(datos_arima, maxlag=4)
            diagnostico["adf_statistic"] = adf_stat
            diagnostico["adf_pvalue"] = adf_pval
            diagnostico["adf_estacionaria"] = adf_pval < 0.05
        except Exception:
            diagnostico["adf_estacionaria"] = None

    # auto_arima: selección automática del mejor orden por AIC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if len(datos_arima) >= 8:
                modelo_auto = auto_arima(
                    datos_arima,
                    start_p=0, max_p=3,
                    start_q=0, max_q=3,
                    d=None,  # auto-detectar diferenciación
                    max_d=2,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    information_criterion="aic",
                )
                resultado = modelo_auto
                order_sel = modelo_auto.order
            else:
                # Series muy cortas: usar orden simple
                order_sel = (1, 0, 0) if len(datos_arima) < 5 else order
                modelo = ARIMA(datos_arima, order=order_sel)
                resultado = modelo.fit()
        except Exception:
            # Fallback al orden manual
            order_sel = order
            try:
                modelo = ARIMA(datos_arima, order=order_sel)
                resultado = modelo.fit()
            except Exception:
                order_sel = (1, 0, 0)
                modelo = ARIMA(datos_arima, order=order_sel)
                resultado = modelo.fit()

    diagnostico["order"] = order_sel
    diagnostico["aic"] = resultado.aic() if callable(getattr(resultado, 'aic', None)) else getattr(resultado, 'aic', None)
    diagnostico["bic"] = resultado.bic() if callable(getattr(resultado, 'bic', None)) else getattr(resultado, 'bic', None)

    # Prediccion
    if hasattr(resultado, 'predict'):
        forecast = resultado.predict(n_periods=n_futuras)
    else:
        forecast = resultado.forecast(steps=n_futuras)

    # Residuos y desvio
    residuos = resultado.resid() if callable(getattr(resultado, 'resid', None)) else resultado.resid
    std_arima = np.std(residuos, ddof=1)
    diagnostico["std_residuos"] = std_arima

    # Fitted values
    if hasattr(resultado, 'fittedvalues'):
        fitted = resultado.fittedvalues() if callable(resultado.fittedvalues) else resultado.fittedvalues
    else:
        fitted = resultado.predict_in_sample()

    cohortes_arima = cohortes_ord[-n_arima:]

    # Bandas de confianza que crecen con el horizonte
    base_std = std_arima

    # Construir filas
    filas = []
    for c in cohortes_ord:
        arima_val = None
        if c in cohortes_arima:
            idx = cohortes_arima.index(c)
            if idx < len(fitted):
                arima_val = float(fitted[idx])
        filas.append({
            "cohorte": c,
            "mob1_arima": arima_val,
            "mob1_arima_sup": arima_val + base_std if arima_val is not None else None,
            "mob1_arima_inf": arima_val - base_std if arima_val is not None else None,
            "tipo": "historica",
        })

    # Futuras con bandas que crecen con sqrt(horizonte)
    ultima = cohortes_ord[-1]
    for j in range(n_futuras):
        ultima = siguiente_cohorte(ultima)
        mob1_est = float(forecast[j])
        h_std = base_std * np.sqrt(j + 1)  # incertidumbre crece con horizonte
        filas.append({
            "cohorte": ultima,
            "mob1_arima": mob1_est,
            "mob1_arima_sup": mob1_est + h_std,
            "mob1_arima_inf": mob1_est - h_std,
            "tipo": "sintetica",
        })

    return pd.DataFrame(filas), std_arima, diagnostico


def calcular_factores_cl_promedio(matriz: pd.DataFrame,
                                  factores: pd.DataFrame,
                                  mob_maduro: int) -> dict:
    """
    Calcula factores Chain Ladder promedio ponderados por volumen.

    Returns:
        dict: {transición: factor_promedio} para cada transición hasta mob_maduro
    """
    factores_cl = {}
    for mob_n in range(1, mob_maduro):
        col_src = f"MOB_{mob_n}"
        col_dst = f"MOB_{mob_n + 1}"
        trans = f"{mob_n}->{mob_n + 1}"

        if col_src not in matriz.columns or col_dst not in matriz.columns:
            break

        # Cohortes con ambos valores observados
        mask = matriz[col_src].notna() & matriz[col_dst].notna()
        if mask.sum() < 3:
            # Fallback: promedio simple de factores
            if trans in factores.columns:
                vals = factores[trans].dropna()
                if len(vals) >= 2:
                    factores_cl[trans] = vals.mean()
            continue

        cohs = matriz.index[mask]
        numerador = matriz.loc[cohs, col_dst].sum()
        denominador = matriz.loc[cohs, col_src].sum()
        if denominador > 0:
            factores_cl[trans] = numerador / denominador

    return factores_cl


def construir_sinteticas(df_regresion: pd.DataFrame,
                         factores_cl: dict,
                         mob_maduro: int,
                         columna_mob1: str = "mob1_regresion") -> pd.DataFrame:
    """
    Construye la matriz de cohortes sintéticas aplicando factores CL
    al MOB 1 estimado.

    Args:
        columna_mob1: columna del DataFrame con el MOB 1 estimado
                      ("mob1_regresion", "mob1_wma" o "mob1_arima")
    """
    sinteticas = df_regresion[df_regresion["tipo"] == "sintetica"].copy()

    filas = []
    for _, row in sinteticas.iterrows():
        cohorte = row["cohorte"]
        mob1 = row[columna_mob1]

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

    df_sint = pd.DataFrame(filas).set_index("cohorte")
    return df_sint


def main():
    print("=" * 60)
    print("Generacion de Cohortes Sinteticas")
    print("=" * 60)

    if not os.path.exists(INPUT_MATRIZ):
        print(f"[ERROR] No se encontro: {INPUT_MATRIZ}")
        sys.exit(1)

    matriz = pd.read_csv(INPUT_MATRIZ, sep=";", decimal=",", index_col=0)
    factores = pd.read_csv(INPUT_FACTORES, sep=";", decimal=",", index_col=0)

    print(f"Matriz leida: {len(matriz)} cohortes x {len(matriz.columns)} MOBs")
    print(f"Parametros: MOB_MADURO={MOB_MADURO}, N_FUTURAS={N_COHORTES_FUTURAS}, "
          f"N_REGRESION={N_COHORTES_REGRESION}, N_WMA={N_MEDIA_MOVIL}, "
          f"ARIMA={ARIMA_ORDER}, N_ARIMA={N_ARIMA}, METODO={METODO_SINTETICAS}")

    # 1. Regresion lineal del MOB 1
    df_reg, coefs, std_reg = estimar_mob1(matriz, N_COHORTES_REGRESION, N_COHORTES_FUTURAS)
    print(f"\nRegresion MOB 1 (ultimas {N_COHORTES_REGRESION} cohortes):")
    print(f"  Pendiente: {coefs[0]:+.5f} ({coefs[0]*100:+.3f} pp/mes)")
    print(f"  Intercepto: {coefs[1]:.5f}")
    print(f"  Desvio residuos: {std_reg:.5f} ({std_reg*100:.2f} pp)")

    # 2. Media movil ponderada del MOB 1
    df_wma, std_wma = estimar_mob1_wma(matriz, N_MEDIA_MOVIL, N_COHORTES_FUTURAS)
    print(f"\nMedia Movil Ponderada MOB 1 (ventana={N_MEDIA_MOVIL}):")
    print(f"  Desvio residuos: {std_wma:.5f} ({std_wma*100:.2f} pp)")

    # 3. ARIMA con selección automática de orden (auto_arima)
    df_arima, std_arima, diag_arima = estimar_mob1_arima(matriz, N_ARIMA, N_COHORTES_FUTURAS,
                                                          ARIMA_ORDER, SEASONAL_ORDER)
    order_sel = diag_arima.get("order", ARIMA_ORDER)
    print(f"\nauto_arima MOB 1 (ultimas {N_ARIMA} cohortes):")
    print(f"  Orden seleccionado: {order_sel} (por AIC)")
    print(f"  AIC: {diag_arima.get('aic', 'N/A')}")
    print(f"  BIC: {diag_arima.get('bic', 'N/A')}")
    if diag_arima.get("adf_pvalue") is not None:
        adf_est = "SÍ" if diag_arima["adf_estacionaria"] else "NO"
        print(f"  Test ADF: p={diag_arima['adf_pvalue']:.4f} -> estacionaria={adf_est}")
    print(f"  Desvio residuos: {std_arima:.5f} ({std_arima*100:.2f} pp)")

    # Combinar los tres DataFrames
    df_combined = df_reg.copy()
    cols_wma = ["mob1_wma", "mob1_wma_sup", "mob1_wma_inf"]
    for col in cols_wma:
        df_combined[col] = df_wma[col].values
    cols_arima = ["mob1_arima", "mob1_arima_sup", "mob1_arima_inf"]
    for col in cols_arima:
        df_combined[col] = df_arima[col].values

    sinteticas_comb = df_combined[df_combined["tipo"] == "sintetica"]
    print(f"\nMOB 1 estimado para cohortes futuras:")
    print(f"  {'Cohorte':>10s} | {'Regresion':>10s} | {'WMA':>10s} | {'ARIMA':>10s}")
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for _, row in sinteticas_comb.iterrows():
        print(f"  {row['cohorte']:>10s} | {row['mob1_regresion']:>9.4f}  | "
              f"{row['mob1_wma']:>9.4f}  | {row['mob1_arima']:>9.4f}")

    # 4. Factores CL
    factores_cl = calcular_factores_cl_promedio(matriz, factores, MOB_MADURO)
    print(f"\nFactores Chain Ladder promedio ({len(factores_cl)} transiciones):")
    for trans, f in factores_cl.items():
        print(f"  {trans}: {f:.5f}")

    # 5. Construir sinteticas (usando el metodo seleccionado)
    col_mob1_map = {"wma": "mob1_wma", "arima": "mob1_arima", "regresion": "mob1_regresion"}
    col_mob1 = col_mob1_map.get(METODO_SINTETICAS, "mob1_regresion")

    df_sint = construir_sinteticas(df_combined, factores_cl, MOB_MADURO, col_mob1)
    print(f"\nCohortes sinteticas generadas ({METODO_SINTETICAS}): {len(df_sint)}")

    for cohorte in df_sint.index:
        ultimo_mob = df_sint.loc[cohorte].dropna().index[-1]
        ultimo_val = df_sint.loc[cohorte].dropna().iloc[-1]
        mob1_val = df_sint.loc[cohorte, "MOB_1"]
        print(f"  {cohorte}: MOB_1={mob1_val:.4f} -> {ultimo_mob}={ultimo_val:.4f} "
              f"(factor conversion={ultimo_val/mob1_val:.3f})")

    # 6. Guardar
    df_combined.to_csv(OUTPUT_REGRESION, sep=";", decimal=",", index=False)
    df_sint.to_csv(OUTPUT_SINTETICAS, sep=";", decimal=",")

    print(f"\nArchivos guardados:")
    print(f"  {OUTPUT_REGRESION}")
    print(f"  {OUTPUT_SINTETICAS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
