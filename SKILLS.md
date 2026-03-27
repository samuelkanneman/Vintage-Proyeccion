# Skills y Conocimientos Aplicados

Habilidades técnicas y conceptuales necesarias para construir este proyecto de análisis vintage de mora crediticia.

## Conocimientos de Negocio (Riesgo Crediticio)

- **Análisis vintage**: Técnica para evaluar la calidad crediticia agrupando créditos por mes de otorgamiento (cohorte) y observando su evolución en el tiempo.
- **Índice de morosidad**: Ratio `moroso / total_vencido` que mide qué proporción del monto vencido permanece impago. Métrica central del análisis.
- **Months on Books (MOB)**: Concepto de antigüedad de la cartera. MOB=1 es el primer mes después del otorgamiento. Permite comparar cohortes en el mismo punto de su ciclo de vida.
- **Ciclo de vida de la mora**: Comprensión de las fases típicas de una cosecha crediticia — deterioro inicial, plateau/meseta, y curación (o deterioro sostenido en carteras problemáticas).
- **Triángulo de desarrollo**: Estructura donde las cohortes más nuevas tienen menos meses observados, formando un triángulo con datos faltantes en la esquina inferior derecha.

## Técnicas Actuariales / Cuantitativas

- **Factores de desarrollo (link ratios)**: Ratios `índice(MOB_n) / índice(MOB_n-1)` que miden la variación relativa entre meses consecutivos. Herramienta estándar en reservas actuariales.
- **Velocidad de mora (1ra derivada)**: Cambio absoluto del índice `índice(MOB_n) - índice(MOB_n-1)`. Permite identificar aceleración, desaceleración y puntos de inflexión.
- **Diagnóstico de fase**: Combinación de velocidad y factor para clasificar cada cohorte en su fase actual (deterioro acelerando, desaceleración, plateau, curación).
- **Método Chain Ladder**: Técnica actuarial para completar triángulos de desarrollo. Calcula factores de desarrollo promedio ponderados por volumen (`sum(MOB_{n+1}) / sum(MOB_n)`) y los aplica secuencialmente para proyectar las cohortes incompletas hasta un MOB objetivo.
- **Mora última (ultimate)**: Valor proyectado del índice de morosidad al MOB objetivo. Permite estimar la pérdida esperada total antes de que la cohorte llegue a ese punto.
- **Factores de conversión acumulados**: Ratio `índice(MOB_n) / índice(MOB_1)` que mide cuánto se multiplica la mora inicial al llegar a cada MOB. Permite analizar si la relación mora inicial-mora madura es constante o tiene tendencia.
- **Cohortes sintéticas**: Proyección de cohortes futuras (que aún no nacieron) estimando su MOB 1 y aplicando factores CL para construir la curva completa.

## Series Temporales y Modelado Predictivo

- **Regresión lineal**: Baseline para estimar tendencia del MOB 1. Simple, interpretable, pero asume relación lineal y no captura cambios de régimen.
- **Media móvil ponderada (WMA)**: Promedio con pesos exponenciales que da más importancia a observaciones recientes. Incluye estimación de tendencia ponderada para proyección.
- **ARIMA (AutoRegressive Integrated Moving Average)**: Modelo que captura autocorrelación (p), tendencia vía diferenciación (d) y corrección por errores pasados (q). Configuración ARIMA(p,d,q).
- **SARIMA (Seasonal ARIMA)**: Extensión de ARIMA con componente estacional (P,D,Q,s). Con s=12 captura patrones anuales en la mora inicial (ej: meses estacionalmente altos o bajos).
- **Bandas de confianza**: Intervalos de ±1 desvío estándar de los residuos del modelo, que permiten cuantificar la incertidumbre de cada estimación.
- **Backtesting walk-forward**: Validación cruzada temporal donde para cada cohorte se predice su MOB 1 usando solo datos anteriores y se mide el error contra el valor real. Evita data leakage.
- **Grid search de modelos**: Evaluación sistemática de múltiples configuraciones (órdenes ARIMA, ventanas, métodos) rankeadas por MAE del backtesting. Permite seleccionar el modelo óptimo empíricamente.

## Programación y Herramientas

### Python
- **pandas**: Manipulación de DataFrames, pivot tables, operaciones vectorizadas, lectura/escritura de CSV con separadores y decimales personalizados.
- **matplotlib**: Gráficos de líneas (curvas vintage), heatmaps con gradiente condicional, gráficos de barras, box plots, paneles combinados, animaciones GIF.
- **numpy**: Operaciones numéricas auxiliares (regresión polinomial, pesos exponenciales, estadísticas).
- **statsmodels**: Modelos ARIMA y SARIMAX para series temporales. Ajuste, diagnóstico de residuos y forecasting.
- **Jupyter Notebooks**: Creación de tableros interactivos que combinan ejecución de scripts, visualización de datos y análisis narrativo.
- **subprocess**: Orquestación de scripts Python desde notebooks.

### Visualización de Datos
- **Curvas vintage**: Gráfico de líneas con una serie por cohorte, diferenciadas por color (año) y transparencia (destacadas vs individuales).
- **Heatmaps con formato condicional**: Matrices coloreadas con escala `RdYlGn_r` para identificar zonas de riesgo.
- **Gráficos de diagnóstico**: Paneles combinados de velocidad + factor con zonas de color (rojo = deterioro, verde = curación).
- **Diferenciación observado vs proyectado**: Líneas sólidas para datos reales, punteadas para proyecciones, con celdas en gris itálica en matrices.
- **Animaciones**: GIF frame-by-frame mostrando la evolución temporal de las curvas.
- **Leyendas y anotaciones**: Etiquetado de cohortes destacadas, formateo de ejes en porcentaje, grillas sutiles.

### Ingeniería de Datos
- **Pipeline reproducible**: Scripts modulares que se ejecutan secuencialmente, cada uno lee el output del anterior.
- **Normalización de esquema**: Renombrado de columnas, descarte de campos innecesarios, cálculo de campos derivados (MOB).
- **Manejo de CSV con formato regional**: Separador `;` y decimal `,` (formato europeo/latinoamericano).
- **Matrices de marcadores**: Archivos booleanos complementarios para distinguir datos observados de proyectados sin contaminar la matriz numérica.

### Gestión del Proyecto
- **Estructura de directorios estándar**: Separación clara entre datos crudos, procesados, código fuente, notebooks y reportes.
- **Documentación**: README con instrucciones de uso, diccionario de datos con esquema de cada archivo, archivo de skills.
- **Control de versiones**: Repositorio Git con `.gitignore` para archivos generados.
- **Parametrización**: Variables configurables en la parte superior de scripts y notebooks (`MOB_OBJETIVO`, `N_DESTACAR`, colores, transparencias).
