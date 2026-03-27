# Análisis Vintage de Mora Crediticia (Moroso-Vencido)

## Descripción
(SIN DATOS REALES)
Proyecto de ciencia de datos para realizar **análisis vintage** sobre la cartera de créditos otorgados. El análisis vintage permite evaluar la calidad crediticia de cada cohorte de otorgamiento, observando cómo evoluciona la mora (ratio moroso/vencido) a medida que pasan los meses desde el otorgamiento.

## Estructura del Proyecto

```
├── data/
│   ├── raw/                  # Datos crudos (CSV único con todas las cohortes)
│   └── processed/            # Datos consolidados listos para análisis
├── notebooks/
│   ├── tablero_vintage.ipynb          # Tablero de análisis vintage (observado)
│   ├── tablero_proyeccion.ipynb       # Tablero de proyección Chain Ladder
│   ├── tablero_factores_conversion.ipynb  # Análisis de factores de conversión acumulados
│   └── tablero_cohortes_sinteticas.ipynb  # Proyección de cohortes futuras (sintéticas)
├── reports/                  # Reportes y gráficos generados
├── src/                      # Scripts de procesamiento
│   ├── consolidar_vintage.py              # Consolidación y normalización del CSV fuente
│   ├── generar_matriz_vintage.py          # Generación de la matriz vintage pivoteada
│   ├── generar_factores_desarrollo.py     # Factores de desarrollo (link ratios)
│   ├── generar_velocidad_mora.py          # Velocidad de mora (1ra derivada)
│   ├── generar_proyeccion_chainladder.py  # Proyección Chain Ladder del triángulo
│   ├── generar_factores_conversion.py     # Factores de conversión acumulados (MOB 1 → MOB n)
│   ├── generar_cohortes_sinteticas.py     # Proyección de cohortes futuras
│   └── generar_animacion_vintage.py       # Animación GIF de curvas vintage
├── README.md
├── DICCIONARIO_DATOS.md
├── SKILLS.md
└── requirements.txt
```

## Datos de Entrada

El archivo fuente está en `data/raw/otorgamientos 2024-2025.csv`. Es un **CSV único** que contiene todas las cohortes de otorgamiento, con la evolución mes a mes de:

- Montos vencidos acumulados
- Montos pagados acumulados
- Monto moroso (diferencia entre vencido y pagado)
- Índice de morosidad (moroso / vencido)
- Cantidad de operaciones por cohorte

El archivo usa `;` como separador y `,` como separador decimal.


## Uso

### 1. Consolidar datos

```bash
py src/consolidar_vintage.py
```

Lee el CSV fuente, normaliza las columnas al esquema estándar del proyecto (renombra `mes_otorgamiento` → `cohorte`, descarta `ft_clave_con_ven`) y agrega la columna `mob` (months on books). Genera `data/processed/vintage_consolidado.csv`.

### 2. Generar matriz vintage

```bash
py src/generar_matriz_vintage.py
```

Pivotea el consolidado en una tabla donde las filas son cohortes, las columnas son MOBs y los valores son el índice de morosidad. Genera `data/processed/matriz_vintage.csv`.

### 3. Generar factores de desarrollo

```bash
py src/generar_factores_desarrollo.py
```

Calcula los link ratios (`indice(MOB_n) / indice(MOB_n-1)`). Factor >1 = mora creciendo, <1 = recuperación. Genera `data/processed/factores_desarrollo.csv`.

### 4. Generar velocidad de mora

```bash
py src/generar_velocidad_mora.py
```

Calcula la primera derivada del índice (`indice(MOB_n) - indice(MOB_n-1)`). Velocidad >0 = deterioro, <0 = curación. Genera `data/processed/velocidad_mora.csv`.

### 5. Generar factores de conversión acumulados

```bash
py src/generar_factores_conversion.py
```

Calcula el ratio `índice(MOB_n) / índice(MOB_1)` para cada cohorte y MOB. Permite analizar si la relación entre mora inicial y mora madura es constante o tiene tendencia. Genera:
- `data/processed/factores_conversion.csv` — matriz cohorte × MOB con los factores
- `data/processed/factores_conversion_resumen.csv` — resumen estadístico por MOB (promedio, mediana, tendencia)

### 6. Generar cohortes sintéticas (futuras)

```bash
py src/generar_cohortes_sinteticas.py
```

Proyecta cohortes que aún no nacieron. Estima el MOB 1 con tres métodos complementarios y construye la curva completa con factores CL. Genera:
- `data/processed/cohortes_sinteticas.csv` — matriz con las cohortes proyectadas
- `data/processed/regresion_mob1.csv` — datos de estimación MOB 1 (históricas + estimadas, tres métodos)

**Métodos de estimación del MOB 1:**
- **Regresión lineal** (baseline): recta sobre las últimas N cohortes
- **Media móvil ponderada (WMA)**: pesos exponenciales, captura cambios de tendencia recientes
- **ARIMA/SARIMA**: modelos de series temporales con opción de estacionalidad anual (s=12)

Los tres métodos incluyen bandas de confianza de ±1 desvío estándar. El método usado para las sintéticas es configurable (`METODO_SINTETICAS`).

### 7. Proyección Chain Ladder

```bash
py src/generar_proyeccion_chainladder.py
```

Completa el triángulo inferior de la matriz vintage usando el método Chain Ladder (factores de desarrollo promedio ponderados por volumen). Proyecta cada cohorte hasta un MOB objetivo configurable (por defecto MOB 18). Genera:
- `data/processed/factores_chainladder.csv` — factores promedio por transición
- `data/processed/matriz_proyectada.csv` — matriz completa (observado + proyectado)
- `data/processed/matriz_proyectada_marcadores.csv` — indicador booleano de celdas proyectadas vs observadas

### 8. Tablero de análisis vintage (observado)

Abrir `notebooks/tablero_vintage.ipynb` para ejecutar el pipeline de análisis sobre datos observados y generar las visualizaciones (curvas vintage, factores de desarrollo, velocidad de mora, diagnóstico de fases).

### 9. Tablero de proyección Chain Ladder

Abrir `notebooks/tablero_proyeccion.ipynb` para ejecutar la proyección y visualizar:
- Factores Chain Ladder promedio (gráfico de barras)
- Matriz proyectada con celdas observadas vs proyectadas diferenciadas
- Curvas vintage: líneas sólidas (observado) + punteadas (proyectado)
- Tabla de mora última (ultimate) por cohorte
- Comparación de ultimates por año de originación
- Resumen ejecutivo

### 10. Tablero de factores de conversión

Abrir `notebooks/tablero_factores_conversion.ipynb` para analizar la relación entre mora inicial y mora madura:
- Curvas de factor de conversión por cohorte (con línea de MOB maduro)
- Factor al MOB maduro por cohorte (barras) + scatter mora inicial vs factor
- Evolución temporal del factor por MOB intermedio
- Heatmap cohorte × MOB
- Box plot de dispersión por MOB
- Conclusiones automáticas (tendencia, correlación, MOB pico)

### 11. Tablero de cohortes sintéticas

Abrir `notebooks/tablero_cohortes_sinteticas.ipynb` para proyectar cohortes futuras:
- Estimación MOB 1 con tres métodos: regresión lineal, WMA y ARIMA/SARIMA (±1σ)
- Backtesting walk-forward: validación cruzada comparando los tres métodos (MAE, RMSE, % en banda)
- Grid search: ranking automático de ~37 configuraciones (regresión, WMA, ARIMA y SARIMA con distintas ventanas y órdenes)
- Curvas sintéticas superpuestas sobre históricas
- Detalle MOB a MOB (factores CL aplicados)
- Comparación de mora última: observada vs proyectada
- Resumen ejecutivo

## Requisitos

```bash
pip install -r requirements.txt
```
