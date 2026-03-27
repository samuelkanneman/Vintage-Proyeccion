# Diccionario de Datos

## Archivo fuente (`data/raw/otorgamientos 2024-2025.csv`)

CSV único con todas las cohortes. Separador `;` y decimales con `,`.

| Columna                | Tipo    | Descripción |
|------------------------|---------|-------------|
| `mes_otorgamiento`     | string  | Mes de otorgamiento del crédito (formato `YYYY-MM`). Identifica el grupo/cohorte de créditos. Se renombra a `cohorte` en el consolidado. |
| `id_tie_mes`           | string  | Mes de observación (formato `YYYY-MM`). Mes en que se mide el estado de la cartera. |
| `ft_clave_con_ven`     | string  | Identificador interno de OC (ej: "101"). Se descarta en la consolidación. |
| `total_vencido`        | numeric | Monto total acumulado que venció hasta el mes de observación (en moneda local). |
| `total_pagado`         | numeric | Monto total acumulado que fue efectivamente pagado hasta el mes de observación. |
| `moroso`               | numeric | Monto moroso acumulado: `total_vencido - total_pagado`. Representa el saldo impago. |
| `indice`               | float   | **Índice de morosidad**: `moroso / total_vencido`. Proporción del monto vencido que no fue pagado. Rango [0, 1]. Es la métrica clave del análisis vintage. |
| `cantidad_operaciones` | int     | Número de operaciones de crédito en la cohorte. |

## Archivo consolidado (`data/processed/vintage_consolidado.csv`)

Resultado de `consolidar_vintage.py`. Mismas columnas que el fuente con las siguientes diferencias:

- `mes_otorgamiento` se renombra a `cohorte`
- `ft_clave_con_ven` se descarta
- Se agrega la columna `mob`

| Columna                | Tipo    | Descripción |
|------------------------|---------|-------------|
| `cohorte`              | string  | Mes de otorgamiento (renombrado desde `mes_otorgamiento`). |
| `id_tie_mes`           | string  | Mes de observación (formato `YYYY-MM`). |
| `total_vencido`        | numeric | Monto total acumulado vencido. |
| `total_pagado`         | numeric | Monto total acumulado pagado. |
| `moroso`               | numeric | Monto moroso acumulado: `total_vencido - total_pagado`. |
| `indice`               | float   | Índice de morosidad: `moroso / total_vencido`. |
| `cantidad_operaciones` | int     | Número de operaciones de crédito en la cohorte. |
| `mob`                  | int     | **Months on Books**. Meses transcurridos desde el mes de la cohorte hasta el mes de observación. MOB=1 significa el primer mes posterior al otorgamiento. |

## Matriz vintage (`data/processed/matriz_vintage.csv`)

Resultado de `generar_matriz_vintage.py`. Tabla pivoteada del consolidado.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes (`cohorte`). Una fila por mes de otorgamiento. |
| Columnas  | `MOB_1`, `MOB_2`, ..., `MOB_25`. Cada columna representa un mes transcurrido desde el otorgamiento. |
| Valores   | `indice` de morosidad. `NaN` donde la cohorte aún no tiene datos observados (triángulo inferior). |

## Factores de desarrollo (`data/processed/factores_desarrollo.csv`)

Resultado de `generar_factores_desarrollo.py`. Link ratios por cohorte.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes. |
| Columnas  | Transiciones: `1->2`, `2->3`, ..., `24->25`. |
| Valores   | `indice(MOB_n) / indice(MOB_n-1)`. Factor >1 = mora crece, <1 = mora baja. |

## Velocidad de mora (`data/processed/velocidad_mora.csv`)

Resultado de `generar_velocidad_mora.py`. Primera derivada del índice por cohorte.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes. |
| Columnas  | Transiciones: `1->2`, `2->3`, ..., `24->25`. |
| Valores   | `indice(MOB_n) - indice(MOB_n-1)`. Velocidad >0 = deterioro, <0 = curación. |

## Factores Chain Ladder (`data/processed/factores_chainladder.csv`)

Resultado de `generar_proyeccion_chainladder.py`. Factores promedio ponderados por volumen.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Una sola fila (promedio ponderado de todas las cohortes). |
| Columnas  | Transiciones: `1->2`, `2->3`, ..., `17->18` (hasta MOB objetivo). |
| Valores   | `sum(MOB_{n+1}) / sum(MOB_n)` sobre cohortes con ambos valores observados. Es el factor estándar del método Chain Ladder. |

## Matriz proyectada (`data/processed/matriz_proyectada.csv`)

Resultado de `generar_proyeccion_chainladder.py`. Triángulo completado.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes. |
| Columnas  | `MOB_1`, `MOB_2`, ..., `MOB_18` (hasta MOB objetivo). |
| Valores   | `indice` de morosidad. Contiene tanto valores observados como proyectados. |

## Marcadores de proyección (`data/processed/matriz_proyectada_marcadores.csv`)

Resultado de `generar_proyeccion_chainladder.py`. Acompaña a la matriz proyectada.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes. |
| Columnas  | `MOB_1`, `MOB_2`, ..., `MOB_18`. |
| Valores   | Booleano. `True` = celda proyectada por Chain Ladder. `False` = dato observado. |

## Factores de conversión acumulados (`data/processed/factores_conversion.csv`)

Resultado de `generar_factores_conversion.py`. Ratio entre la mora en cada MOB y la mora inicial.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes. |
| Columnas  | `MOB_1`, `MOB_2`, ..., `MOB_25`. |
| Valores   | `índice(MOB_n) / índice(MOB_1)`. Factor >1 = deterioro neto vs inicio, <1 = curación neta. MOB_1 siempre es 1.0. |

## Resumen de factores de conversión (`data/processed/factores_conversion_resumen.csv`)

Resultado de `generar_factores_conversion.py`. Estadísticas descriptivas por MOB.

| Columna          | Tipo    | Descripción |
|------------------|---------|-------------|
| `mob`            | int     | MOB de referencia. |
| `n_cohortes`     | int     | Cantidad de cohortes con dato en ese MOB. |
| `promedio`       | float   | Factor de conversión promedio. |
| `mediana`        | float   | Mediana del factor. |
| `desvio`         | float   | Desvío estándar. |
| `minimo`         | float   | Factor mínimo observado. |
| `maximo`         | float   | Factor máximo observado. |
| `cohorte_min`    | string  | Cohorte con el factor mínimo. |
| `cohorte_max`    | string  | Cohorte con el factor máximo. |
| `prom_1ra_mitad` | float   | Promedio de la primera mitad de cohortes (cronológicamente). |
| `prom_2da_mitad` | float   | Promedio de la segunda mitad. |
| `cambio_pct`     | float   | Cambio porcentual entre mitades. |
| `tendencia`      | string  | `SUBIENDO`, `BAJANDO` o `ESTABLE` (umbral ±2%). |

## Cohortes sintéticas (`data/processed/cohortes_sinteticas.csv`)

Resultado de `generar_cohortes_sinteticas.py`. Cohortes futuras proyectadas.

| Dimensión | Descripción |
|-----------|-------------|
| Filas     | Cohortes sintéticas (ej: 2026-02, 2026-03, 2026-04). |
| Columnas  | `MOB_1`, `MOB_2`, ..., `MOB_15` (hasta MOB maduro). |
| Valores   | Índice de morosidad proyectado. MOB_1 estimado por regresión lineal, media móvil ponderada o ARIMA/SARIMA (configurable vía `METODO_SINTETICAS`), MOBs siguientes por factores CL. |

## Regresión MOB 1 (`data/processed/regresion_mob1.csv`)

Resultado de `generar_cohortes_sinteticas.py`. Datos de estimación del MOB 1 con tres métodos.

| Columna           | Tipo    | Descripción |
|-------------------|---------|-------------|
| `cohorte`         | string  | Mes de otorgamiento. |
| `mob1_observado`  | float   | Índice MOB 1 observado (null para sintéticas). |
| `mob1_regresion`  | float   | Valor estimado por la recta de regresión lineal (null si no está en el rango). |
| `mob1_reg_sup`    | float   | Banda superior de regresión: `mob1_regresion + 1 desvío estándar`. |
| `mob1_reg_inf`    | float   | Banda inferior de regresión: `mob1_regresion - 1 desvío estándar`. |
| `mob1_wma`        | float   | Valor estimado por media móvil ponderada exponencial (null si no hay suficientes datos). |
| `mob1_wma_sup`    | float   | Banda superior WMA: `mob1_wma + 1 desvío estándar`. |
| `mob1_wma_inf`    | float   | Banda inferior WMA: `mob1_wma - 1 desvío estándar`. |
| `mob1_arima`      | float   | Valor estimado por modelo ARIMA(p,d,q) o SARIMA(p,d,q)(P,D,Q,s) (null si no hay suficientes datos). |
| `mob1_arima_sup`  | float   | Banda superior ARIMA/SARIMA: `mob1_arima + 1 desvío estándar`. |
| `mob1_arima_inf`  | float   | Banda inferior ARIMA/SARIMA: `mob1_arima - 1 desvío estándar`. |
| `tipo`            | string  | `historica` o `sintetica`. |
| `en_regresion`    | bool    | Si la cohorte fue usada para ajustar la regresión lineal. |

## Notas para el análisis

- **Matriz Vintage**: Pivotear el DataFrame consolidado con `cohorte` como filas, `mob` como columnas y `indice` como valor.
- **Curvas Vintage**: Graficar `indice` (eje Y) vs `mob` (eje X), con una línea por cada `cohorte`.
- Un índice más alto indica mayor morosidad. Cohortes con curvas más altas representan otorgamientos de peor calidad crediticia.
- Los datos son **acumulados**: cada fila muestra el total acumulado hasta ese mes, no el incremental del mes.
- **Chain Ladder**: El método asume que los patrones históricos de desarrollo se mantendrán en el futuro. Los factores ponderados por volumen dan más peso a cohortes con índices más altos.
