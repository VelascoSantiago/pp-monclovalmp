# **Arkham Data Science Challenge: Anomalías PML \- Zona Monclova**

Este repositorio contiene un pipeline ETL de producción diseñado para la extracción, transformación, detección de anomalías y visualización de los Precios Marginales Locales (PML) del Mercado Eléctrico Mayorista (CENACE).  
El proyecto fue construido con un enfoque de ingeniería de software y ciencia de datos, garantizando modularidad, resiliencia ante fallos de red y reproducibilidad estricta.

## **Metodología: El "Tridente Analítico"**

Debido a la asimetría y estacionalidad del mercado eléctrico (especialmente durante los meses de verano), el uso de un solo modelo estadístico puede generar un exceso de falsos positivos. Este pipeline implementa un ensamble de consenso:

> 1. **IQR Robusto (x3.0):** Establece una línea base estática para detectar valores históricamente extremos, filtrando el ruido estacional.  
> 2. **Z-Score Móvil (168h):** Control dinámico para detectar shocks de precio basados exclusivamente en la volatilidad de los últimos 7 días.  
> 3. **Isolation Forest (Machine Learning):** Algoritmo no supervisado que detecta incoherencias multivariadas cruzando el precio del nodo con la hora del día.

**Regla de Negocio:** Una alerta se cataloga como "Crítica" únicamente si 2 o más modelos coinciden en su clasificación. Adicionalmente, el pipeline computa métricas de evaluación no supervisadas (como el Índice de Jaccard) para comparar el nivel de acuerdo entre modelos.

## **Estructura del Repositorio**

El proyecto sigue una estructura orientada a producción:

> * catalog/: Catálogos maestros para la resolución dinámica de zonas de carga y nodos.  
> * data/raw/: Almacenamiento de datos crudos (ej. extracciones previas en CSV).  
> * data/processed/: Resultados del pipeline y cálculo de métricas.  
> * notebooks/: Entornos de experimentación y prototipado.  
> * reports/: Dashboards interactivos generados en HTML.  
> * src/pp\_ds\_monclovalmp/: Código fuente del orquestador y modelos (main.py).  
> * presentation/: Slide Deck ejecutivo con los resultados y roadmap técnico.

## **Reproducibilidad y Requisitos**

Este proyecto utiliza uv como gestor de paquetes y entornos para garantizar la paridad exacta de dependencias mediante el archivo uv.lock.

> 1. Instalar uv (si no se encuentra en el sistema):  
>    curl \-LsSf https://astral.sh/uv/install.sh | sh  
> 2. Clonar el repositorio y sincronizar el entorno (resolución estricta):  
>    uv sync \--frozen

## **Uso del Pipeline**

El pipeline incluye una Interfaz de Línea de Comandos (CLI) que permite interactuar directamente con el servicio SW-PML del CENACE o consumir un archivo local.  
Para ejecutar el pipeline utilizando un dataset local (modo recomendado para CI/CD y evaluación rápida):  
uv run python src/pp\_ds\_monclovalmp/main.py \--usar-csv-local data/raw/monclova\_pml\_2024\_2025.csv

Para extraer datos en vivo consultando la API para toda la zona de Monclova en un rango de fechas:  
uv run python src/pp\_ds\_monclovalmp/main.py \--nodos Monclova \--fecha-inicio 2024-01-01 \--fecha-fin 2025-06-30

Para consultar los parámetros de configuración de los modelos (umbrales, ventanas) y la API:  
uv run python src/pp\_ds\_monclovalmp/main.py \--help

## **Entregables Generados**

Al finalizar la ejecución, el script genera automáticamente los siguientes artefactos:

> 1. **Datos (data/processed/):**  
   * Archivo maestro con las banderas de anomalías.  
   * CSV con las métricas de evaluación de modelos por nodo.  
> 2. **Reportes Visuales (reports/):**  
   * *Dashboard de Nodos:* Gráfica de series de tiempo interactiva con menú desplegable para inspeccionar la línea base y el tridente analítico por nodo.  
   * *Termómetro de Riesgo:* Heatmap de frecuencia mensual de alertas críticas.  
   * *Mapa Operativo Global:* Categorical Scatter plot para identificar visualmente fallos sistémicos versus locales en la red.