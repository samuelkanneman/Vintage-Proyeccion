"""
Proyeccion Chain Ladder por segmento (tipoope x clientenuevo x nivel_riesgo).

Para cada uno de los 12 segmentos:
  1. Calcula factores CL promedio ponderados por volumen
  2. Proyecta el triangulo inferior hasta MOB_OBJETIVO

Tambien genera una proyeccion "General" ponderada por cantidad de operaciones
de cada segmento al MOB 1.

Lee: data/processed/matrices_vintage_niveles.csv
     data/processed/vintage_niveles_consolidado.csv (para pesos)

Genera:
  - data/processed/factores_cl_niveles.csv (factores por segmento)
  - data/processed/matrices_proyectadas_niveles.csv (matrices completas)
  - data/processed/marcadores_proyectados_niveles.csv (True=proyectado)
  - data/processed/proyeccion_general_niveles.csv (promedio ponderado)

Uso:
    python src/generar_proyeccion_chainladder_niveles.py
"""

import os
import sys

import numpy as np
import pandas as pd

from generar_proyeccion_chainladder import calcular_mack_diagnostico

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
INPUT_MATRICES = os.path.join(PROCESSED_DATA_DIR, "matrices_vintage_niveles.csv")
INPUT_CONSOLIDADO = os.path.join(PROCESSED_DATA_DIR, "vintage_niveles_consolidado.csv")
OUTPUT_FACTORES = os.path.join(PROCESSED_DATA_DIR, "factores_cl_niveles.csv")
OUTPUT_MATRICES = os.path.join(PROCESSED_DATA_DIR, "matrices_proyectadas_niveles.csv")
OUTPUT_MARCADORES = os.path.join(PROCESSED_DATA_DIR, "marcadores_proyectados_niveles.csv")
OUTPUT_GENERAL = os.path.join(PROCESSED_DATA_DIR, "proyeccion_general_niveles.csv")

MOB_OBJETIVO = 18

# ---------------------------------------------------------------------------
# Segmentos excluidos del cálculo general ponderado.
# Se proyectan individualmente (para monitoreo histórico) pero NO participan
# en la proyección general porque actualmente no se vende a estos segmentos.
# ---------------------------------------------------------------------------
SEGMENTOS_EXCLUIDOS_GENERAL = [
    "CC_V_Medio",
    "CC_V_Alto",
    "CC_F_Alto",
    "PP_V_Medio",
    "PP_V_Alto",
    "PP_F_Medio",
    "PP_F_Alto"
]


def calcular_factores_cl(matriz, mob_objetivo):
    """Calcula factores CL ponderados por volumen."""
    mob_nums = sorted([int(c.replace("MOB_", "")) for c in matriz.columns if c.startswith("MOB_")])
    factores = {}
    for i in range(len(mob_nums) - 1):
        m_act = mob_nums[i]
        m_sig = mob_nums[i + 1]
        if m_act >= mob_objetivo:
            break
        col_act = f"MOB_{m_act}"
        col_sig = f"MOB_{m_sig}"
        mask = matriz[col_act].notna() & matriz[col_sig].notna()
        if mask.sum() > 0:
            suma_sig = matriz.loc[mask, col_sig].sum()
            suma_act = matriz.loc[mask, col_act].sum()
            factores[f"{m_act}->{m_sig}"] = suma_sig / suma_act if suma_act != 0 else 1.0
    return factores


def proyectar_triangulo(matriz, factores, mob_objetivo):
    """Completa el triangulo inferior con factores CL."""
    cols_obj = [f"MOB_{m}" for m in range(1, mob_objetivo + 1)]
    for col in cols_obj:
        if col not in matriz.columns:
            matriz[col] = float("nan")

    proyectada = matriz[cols_obj].copy()
    marcadores = pd.DataFrame(False, index=matriz.index, columns=cols_obj)

    for cohorte in proyectada.index:
        vals_obs = proyectada.loc[cohorte].dropna()
        if len(vals_obs) == 0:
            continue
        ultimo_mob = max(int(c.replace("MOB_", "")) for c in vals_obs.index)
        for mob in range(ultimo_mob + 1, mob_objetivo + 1):
            trans = f"{mob-1}->{mob}"
            if trans in factores:
                col_ant = f"MOB_{mob-1}"
                col_act = f"MOB_{mob}"
                proyectada.loc[cohorte, col_act] = proyectada.loc[cohorte, col_ant] * factores[trans]
                marcadores.loc[cohorte, col_act] = True

    return proyectada, marcadores


def generar_proyeccion_general(all_proyectadas, df_consolidado, mob_objetivo,
                               segmentos_excluidos=None):
    """
    Genera una proyeccion general ponderada por cantidad de operaciones.

    Para cada cohorte y MOB, el indice general es el promedio ponderado
    de los indices de cada segmento, usando como peso la cantidad de
    operaciones al MOB 1 de cada segmento en esa cohorte.

    Los segmentos en segmentos_excluidos se omiten del cálculo ponderado
    (se proyectan individualmente pero no afectan el promedio general).
    """
    if segmentos_excluidos is None:
        segmentos_excluidos = []

    # Calcular pesos: ops al MOB 1 por segmento y cohorte
    mob1 = df_consolidado[df_consolidado["mob"] == 1].copy()
    pesos = mob1.pivot_table(index="cohorte", columns="segmento",
                             values="cantidad_operaciones", aggfunc="sum")

    cols_mob = [f"MOB_{m}" for m in range(1, mob_objetivo + 1)]
    cohortes_todas = sorted(set().union(*[set(p.index) for p in all_proyectadas.values()]))

    resultado = pd.DataFrame(index=cohortes_todas, columns=cols_mob, dtype=float)

    for cohorte in cohortes_todas:
        for col in cols_mob:
            suma_pond = 0.0
            suma_pesos = 0.0
            for seg, proy in all_proyectadas.items():
                if seg in segmentos_excluidos:
                    continue
                if cohorte not in proy.index or col not in proy.columns:
                    continue
                val = proy.loc[cohorte, col]
                if pd.isna(val):
                    continue
                peso = pesos.loc[cohorte, seg] if (cohorte in pesos.index and seg in pesos.columns
                                                    and pd.notna(pesos.loc[cohorte, seg])) else 0
                if peso > 0:
                    suma_pond += val * peso
                    suma_pesos += peso
            if suma_pesos > 0:
                resultado.loc[cohorte, col] = suma_pond / suma_pesos

    resultado.index.name = "cohorte"
    return resultado


def main():
    print("=" * 60)
    print("Proyeccion Chain Ladder por Niveles de Riesgo")
    print("=" * 60)

    if not os.path.exists(INPUT_MATRICES):
        print(f"[ERROR] No se encontro: {INPUT_MATRICES}")
        sys.exit(1)

    matrices_df = pd.read_csv(INPUT_MATRICES, sep=";", decimal=",")
    df_consolidado = pd.read_csv(INPUT_CONSOLIDADO, sep=";", decimal=",")
    segmentos = sorted(matrices_df["segmento"].unique())

    print(f"Segmentos: {len(segmentos)}")
    print(f"MOB objetivo: {MOB_OBJETIVO}")

    all_factores = []
    all_proyectadas_list = []
    all_marcadores_list = []
    all_proyectadas_dict = {}
    all_diagnosticos = []

    for segmento in segmentos:
        sub = matrices_df[matrices_df["segmento"] == segmento].copy()
        mob_cols = [c for c in sub.columns if c.startswith("MOB_")]
        matriz = sub.set_index("cohorte")[mob_cols].copy()

        factores = calcular_factores_cl(matriz, MOB_OBJETIVO)
        proyectada, marcadores = proyectar_triangulo(matriz, factores, MOB_OBJETIVO)

        # Diagnóstico Mack por segmento
        diag = calcular_mack_diagnostico(matriz)
        for trans, cv in diag["factor_variability"].items():
            n = diag["n_cohortes_por_transicion"].get(trans, 0)
            all_diagnosticos.append({
                "segmento": segmento, "transicion": trans,
                "factor_cl": factores.get(trans, np.nan),
                "cv_factores": cv, "n_cohortes": n,
                "estable": cv < 0.3 and n >= 5 if not np.isnan(cv) else False,
            })

        # Guardar factores
        row = {"segmento": segmento}
        row.update(factores)
        all_factores.append(row)

        # Guardar matrices
        proy_out = proyectada.copy()
        proy_out["segmento"] = segmento
        proy_out = proy_out.reset_index()
        all_proyectadas_list.append(proy_out)

        marc_out = marcadores.copy()
        marc_out["segmento"] = segmento
        marc_out = marc_out.reset_index()
        all_marcadores_list.append(marc_out)

        all_proyectadas_dict[segmento] = proyectada

        n_proy = marcadores.sum().sum()
        print(f"  {segmento}: {len(matriz)} cohortes, {len(factores)} factores, {int(n_proy)} celdas proyectadas")

    # Guardar factores
    df_factores = pd.DataFrame(all_factores)
    df_factores.to_csv(OUTPUT_FACTORES, sep=";", decimal=",", index=False)
    print(f"\nFactores guardados: {OUTPUT_FACTORES}")

    # Guardar matrices proyectadas
    df_proy = pd.concat(all_proyectadas_list, ignore_index=True)
    mob_cols = [f"MOB_{m}" for m in range(1, MOB_OBJETIVO + 1)]
    df_proy = df_proy[["segmento", "cohorte"] + mob_cols]
    df_proy.to_csv(OUTPUT_MATRICES, sep=";", decimal=",", index=False)
    print(f"Matrices proyectadas: {OUTPUT_MATRICES}")

    df_marc = pd.concat(all_marcadores_list, ignore_index=True)
    df_marc = df_marc[["segmento", "cohorte"] + mob_cols]
    df_marc.to_csv(OUTPUT_MARCADORES, sep=";", index=False)
    print(f"Marcadores: {OUTPUT_MARCADORES}")

    # Proyeccion general ponderada (excluyendo segmentos que no se venden)
    print("\nGenerando proyeccion general ponderada por operaciones...")
    if SEGMENTOS_EXCLUIDOS_GENERAL:
        print(f"  Excluidos del general: {SEGMENTOS_EXCLUIDOS_GENERAL}")
    general = generar_proyeccion_general(all_proyectadas_dict, df_consolidado, MOB_OBJETIVO,
                                          SEGMENTOS_EXCLUIDOS_GENERAL)
    general.to_csv(OUTPUT_GENERAL, sep=";", decimal=",")
    print(f"Proyeccion general: {OUTPUT_GENERAL}")
    print(f"  Cohortes: {len(general)}")

    # Mostrar pesos usados
    mob1 = df_consolidado[df_consolidado["mob"] == 1]
    pesos_totales = mob1.groupby("segmento")["cantidad_operaciones"].sum()
    total_ops = pesos_totales.sum()
    print(f"\nPesos (% operaciones al MOB 1):")
    for seg in segmentos:
        peso = pesos_totales.get(seg, 0)
        print(f"  {seg:20s}: {peso:>8,.0f} ops ({peso/total_ops*100:>5.1f}%)")
    print(f"  {'TOTAL':20s}: {total_ops:>8,.0f} ops")

    # Guardar diagnóstico por segmento
    if all_diagnosticos:
        diag_path = os.path.join(PROCESSED_DATA_DIR, "diagnostico_cl_niveles.csv")
        pd.DataFrame(all_diagnosticos).to_csv(diag_path, sep=";", decimal=",", index=False)
        print(f"\nDiagnóstico CL por segmento: {diag_path}")

        # Resumen: transiciones inestables
        df_diag = pd.DataFrame(all_diagnosticos)
        inestables = df_diag[~df_diag["estable"]]
        if not inestables.empty:
            print(f"\n  ALERTA: {len(inestables)} transiciones inestables (CV>0.3 o N<5):")
            for _, r in inestables.head(10).iterrows():
                print(f"    {r['segmento']:20s} {r['transicion']:>8s}  CV={r['cv_factores']:.3f}  N={r['n_cohortes']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
