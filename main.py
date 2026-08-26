"""
Pipeline de Producción Arkham
=============================
Detección de anomalías en el Precio Marginal Local (PML) de nodos del SIN
(Sistema Eléctrico Nacional), vía el servicio web SW-PML del CENACE.

Uso:
    python main.py --nodos Monclova --fecha-inicio 2024-01-01 --fecha-fin 2025-06-30
    python main.py --nodos 06AEO-115 06AHM-400 Huasteca --fecha-inicio 2024-01-01 --fecha-fin 2025-06-30
    python main.py --usar-csv-local monclova_pml_2024_2025.csv
"""

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests as rq
import warnings
from requests.adapters import HTTPAdapter
from sklearn.ensemble import IsolationForest
from urllib3.util.retry import Retry

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("arkham")


# --- 0. CONFIGURACIÓN ---

@dataclass
class PipelineConfig:
    """Configuración centralizada del pipeline. Todo lo que antes estaba
    hardcodeado en el cuerpo de las funciones vive aquí, y puede sobreescribirse
    vía argumentos de línea de comandos (ver build_arg_parser)."""

    nodos_input: list[str] = field(default_factory=lambda: ["Monclova"])
    fecha_inicio: date = date(2024, 1, 1)
    fecha_fin: date = date(2025, 6, 30)
    sistema: str = "SIN"
    proceso: str = "MDA"
    formato: str = "JSON"
    max_nodos_por_request: int = 20  # límite documentado del SW-PML
    max_dias_por_request: int = 7    # límite documentado del SW-PML
    request_timeout_s: int = 30
    request_sleep_s: float = 2.0     # cortesía entre requests
    max_reintentos: int = 3

    # Umbrales de los métodos de detección
    zscore_ventana_horas: int = 168
    zscore_umbral: float = 3.0
    iforest_contamination: float = 0.01
    iqr_multiplicador: float = 3.0
    alerta_critica_min_metodos: int = 2

    # Rutas
    catalogo_nodos_path: str = "catalog/nodos_catalogo.csv"
    output_dir_data: str = "data/processed"
    output_dir_reports: str = "reports"
    csv_local_override: str | None = None  # si se da, se usa en vez del API


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de detección de anomalías PML (CENACE)")
    parser.add_argument("--nodos", nargs="+", default=["Monclova"],
                         help="Claves de nodo (ej. 06AEO-115) y/o zonas de carga (ej. Monclova). Mezcla libre.")
    parser.add_argument("--fecha-inicio", type=str, default="2024-01-01")
    parser.add_argument("--fecha-fin", type=str, default="2025-06-30")
    parser.add_argument("--sistema", type=str, default="SIN", choices=["SIN", "BCA", "BCS"])
    parser.add_argument("--proceso", type=str, default="MDA", choices=["MDA", "MTR"])
    parser.add_argument("--zscore-umbral", type=float, default=3.0)
    parser.add_argument("--iforest-contamination", type=float, default=0.01)
    parser.add_argument("--iqr-multiplicador", type=float, default=3.0)
    parser.add_argument("--usar-csv-local", type=str, default=None,
                         help="Ruta a un CSV ya extraído; si se da, no se llama al API.")
    parser.add_argument("--catalogo", type=str, default="catalog/nodos_catalogo.csv")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        nodos_input=args.nodos,
        fecha_inicio=datetime.strptime(args.fecha_inicio, "%Y-%m-%d").date(),
        fecha_fin=datetime.strptime(args.fecha_fin, "%Y-%m-%d").date(),
        sistema=args.sistema,
        proceso=args.proceso,
        zscore_umbral=args.zscore_umbral,
        iforest_contamination=args.iforest_contamination,
        iqr_multiplicador=args.iqr_multiplicador,
        csv_local_override=args.usar_csv_local,
        catalogo_nodos_path=args.catalogo,
    )


# --- 1. RESOLUCIÓN DE NODOS (catálogo -> claves de nodo) ---

def load_node_catalog(catalog_path: str) -> pd.DataFrame:
    """Carga el catálogo de NodosP (columnas: sistema, zona_carga, clave, nombre, municipio)."""
    df = pd.read_csv(catalog_path)
    df["clave"] = df["clave"].astype(str).str.upper().str.strip()
    df["zona_carga"] = df["zona_carga"].astype(str).str.upper().str.strip()
    return df


def resolve_nodes(inputs: list[str], catalog: pd.DataFrame, max_nodos: int = 20) -> list[list[str]]:
    """Resuelve una lista mixta de claves de nodo y/o zonas de carga a una lista
    deduplicada de claves de nodo, y la trocea en bloques de tamaño <= max_nodos
    (límite documentado del SW-PML: 1 a 20 NodosP por consulta).

    Ejemplo: ["Monclova", "06AEO-115"] -> resuelve la zona a sus 9 nodos,
    "06AEO-115" ya viene incluido ahí -> el unique() lo deja en 9, no 10.
    """
    claves_validas = set(catalog["clave"])
    resolved: list[str] = []
    no_encontrados: list[str] = []

    for item in inputs:
        item_norm = item.strip().upper()
        if item_norm in claves_validas:
            resolved.append(item_norm)
            continue
        zona_match = catalog[catalog["zona_carga"] == item_norm]
        if not zona_match.empty:
            nodos_zona = zona_match["clave"].tolist()
            logger.info(f"Zona '{item}' resuelta a {len(nodos_zona)} nodos: {nodos_zona}")
            resolved.extend(nodos_zona)
        else:
            no_encontrados.append(item)

    if no_encontrados:
        logger.warning(
            f"Los siguientes inputs no coinciden con ninguna clave de nodo ni zona "
            f"de carga del catálogo y fueron ignorados: {no_encontrados}"
        )

    # unique() preservando orden de aparición
    seen: set[str] = set()
    unique_resolved: list[str] = []
    for n in resolved:
        if n not in seen:
            seen.add(n)
            unique_resolved.append(n)

    if not unique_resolved:
        raise ValueError(
            "Ningún nodo pudo resolverse a partir de los inputs dados. "
            "Verifica los nombres de nodo/zona contra el catálogo."
        )

    bloques = [unique_resolved[i:i + max_nodos] for i in range(0, len(unique_resolved), max_nodos)]
    if len(bloques) > 1:
        logger.info(
            f"{len(unique_resolved)} nodos resueltos exceden el límite de {max_nodos} por "
            f"request; se trocearon en {len(bloques)} bloques."
        )
    return bloques


# --- 2. MÓDULOS DE EXTRACCIÓN (EXTRACT) ---

def flattenData(form: dict[str, Any]) -> pd.DataFrame:
    """Aplana el JSON de respuesta del CENACE a un DataFrame tabular."""
    flattenedRows = []
    for nodo in form.get('Resultados', []):
        clave = nodo['clv_nodo']
        for valores in nodo['Valores']:
            valores['nodo'] = clave
            flattenedRows.append(valores)
    return pd.DataFrame(flattenedRows)


def getRequestURLs(startDate: date, endDate: date, lista_nodos: list[str], sistema: str = 'SIN',
                    proceso: str = 'MDA', formato: str = 'JSON', max_dias: int = 7) -> list[str]:
    """Genera la lista de URLs para iterar sobre la API del CENACE en bloques de
    `max_dias` días (límite documentado: 1 a 7 Días de Operación por consulta)."""
    nodos_str = ",".join(lista_nodos)
    URLlist = []
    current_date = startDate
    while current_date <= endDate:
        chunk_end = min(current_date + timedelta(days=max_dias - 1), endDate)
        f_ini = current_date.strftime("%Y/%m/%d")
        f_fin = chunk_end.strftime("%Y/%m/%d")
        url = f"https://ws01.cenace.gob.mx:8082/SWPML/SIM/{sistema}/{proceso}/{nodos_str}/{f_ini}/{f_fin}/{formato}"
        URLlist.append(url)
        current_date = chunk_end + timedelta(days=1)
    return URLlist


def _build_session(max_reintentos: int) -> rq.Session:
    """Sesión HTTP con reintentos automáticos (backoff exponencial) para errores
    transitorios de red y códigos 5xx/429."""
    session = rq.Session()
    retry = Retry(
        total=max_reintentos,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def extract_cenace_data(config: PipelineConfig, nodo_bloques: list[list[str]]) -> pd.DataFrame:
    """Ejecuta las peticiones a la API del CENACE (cruzando bloques de nodos x
    bloques de fecha) y consolida los datos. Revisa el campo `status` de cada
    respuesta -- un HTTP 200 puede traer `status=No Content` si no hay datos
    para ese rango, y eso no debe tratarse como un chunk válido pero vacío
    silenciosamente: se loguea explícitamente."""
    session = _build_session(config.max_reintentos)
    lista_chunks = []
    chunks_fallidos = []
    chunks_sin_contenido = []

    for bloque_idx, nodos in enumerate(nodo_bloques, start=1):
        urls = getRequestURLs(
            config.fecha_inicio, config.fecha_fin, nodos,
            sistema=config.sistema, proceso=config.proceso, formato=config.formato,
            max_dias=config.max_dias_por_request,
        )
        logger.info(f"Bloque de nodos {bloque_idx}/{len(nodo_bloques)} ({len(nodos)} nodos): {len(urls)} requests")

        for url in urls:
            try:
                respuesta = session.get(url, timeout=config.request_timeout_s)
            except rq.exceptions.RequestException as e:
                logger.error(f"Fallo de red tras reintentos en {url}: {e}")
                chunks_fallidos.append(url)
                continue

            if respuesta.status_code != 200:
                logger.error(f"HTTP {respuesta.status_code} en {url}")
                chunks_fallidos.append(url)
                continue

            try:
                payload = respuesta.json()
            except ValueError:
                logger.error(f"Respuesta no es JSON válido en {url}")
                chunks_fallidos.append(url)
                continue

            status = payload.get("status")
            if status in ("No Content",):
                logger.info(f"Sin datos (status=No Content) para {url}")
                chunks_sin_contenido.append(url)
                continue
            if status not in ("OK", "Created"):
                logger.warning(f"status inesperado '{status}' en {url}; se intenta procesar de todas formas")

            df_chunk = flattenData(payload)
            if not df_chunk.empty:
                lista_chunks.append(df_chunk)

            time.sleep(config.request_sleep_s)

    logger.info(
        f"Extracción completa: {len(lista_chunks)} chunks con datos, "
        f"{len(chunks_sin_contenido)} sin contenido, {len(chunks_fallidos)} fallidos"
    )
    if chunks_fallidos:
        logger.warning(f"URLs que fallaron y NO están en el dataset final: {chunks_fallidos}")

    if not lista_chunks:
        raise RuntimeError("La extracción no devolvió ningún dato utilizable. Revisa el log de errores arriba.")

    return pd.concat(lista_chunks, ignore_index=True)


# --- 3. MÓDULOS DE DETECCIÓN (TRANSFORM) ---

def getGlobalZscore(df_nodo: pd.DataFrame, ventana: int, umbral: float) -> pd.DataFrame:
    """Aplica el método estadístico Z-Score con ventana móvil configurable."""
    df_nodo['media_movil'] = df_nodo['pml'].rolling(window=ventana, min_periods=1).mean()
    df_nodo['std_movil'] = df_nodo['pml'].rolling(window=ventana, min_periods=1).std()

    df_nodo['z_score'] = np.where(
        df_nodo['std_movil'] > 0,
        (df_nodo['pml'] - df_nodo['media_movil']) / df_nodo['std_movil'],
        0
    )
    df_nodo['anom_zscore'] = np.where(df_nodo['z_score'].abs() > umbral, 1, 0)
    return df_nodo


def getIsolationForest(df_nodo: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Aplica Machine Learning (Isolation Forest) evaluando precio y hora."""
    X = df_nodo[['pml', 'hora']]
    modelo_if = IsolationForest(contamination=contamination, random_state=42)
    predicciones = modelo_if.fit_predict(X)
    df_nodo['anom_isotree'] = np.where(predicciones == -1, 1, 0)
    return df_nodo


def getIQR(df_nodo: pd.DataFrame, multiplicador: float = 3.0) -> pd.DataFrame:
    """Aplica el método de Rango Intercuartílico (IQR) con multiplicador configurable."""
    Q1 = df_nodo['pml'].quantile(0.25)
    Q3 = df_nodo['pml'].quantile(0.75)
    IQR_val = Q3 - Q1

    limite_inferior = Q1 - multiplicador * IQR_val
    limite_superior = Q3 + multiplicador * IQR_val

    df_nodo['anom_iqr'] = np.where(
        (df_nodo['pml'] < limite_inferior) | (df_nodo['pml'] > limite_superior),
        1, 0
    )
    return df_nodo


# --- 4. MÓDULO DE MÉTRICAS DE EVALUACIÓN (NO SUPERVISADAS) ---

def compute_evaluation_metrics(df_maestro: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de evaluación no supervisadas por nodo, ya que no existe
    un ground truth etiquetado de anomalías (las tres columnas de anomalía SON
    la salida de los métodos, no una validación externa).

    Métricas:
    - tasa_<metodo>: % de horas marcadas como anómalas por cada método.
    - jaccard_<m1>_<m2>: acuerdo (intersección/unión) entre cada par de métodos.
      Un Jaccard bajo indica que los métodos capturan fenómenos distintos.
    - tasa_alerta_critica: % de horas con consenso (2+ métodos de acuerdo).
    - abs_zscore_medio_flagged / no_flagged: separación de magnitud entre
      puntos marcados y no marcados -- evidencia de que lo detectado es un
      outlier real y no ruido de un método mal calibrado.
    """
    metodos = ['anom_zscore', 'anom_isotree', 'anom_iqr']
    filas = []

    for nodo, df_n in df_maestro.groupby('nodo'):
        fila: dict[str, Any] = {'nodo': nodo, 'n_horas': len(df_n)}

        for m in metodos:
            fila[f'tasa_{m}'] = round(df_n[m].mean(), 4)

        for i in range(len(metodos)):
            for j in range(i + 1, len(metodos)):
                a, b = metodos[i], metodos[j]
                inter = int(((df_n[a] == 1) & (df_n[b] == 1)).sum())
                union = int(((df_n[a] == 1) | (df_n[b] == 1)).sum())
                fila[f'jaccard_{a}_{b}'] = round(inter / union, 4) if union > 0 else np.nan

        fila['tasa_alerta_critica'] = round(df_n['alerta_critica'].mean(), 4)

        flagged = df_n[df_n['alerta_critica'] == 1]
        no_flagged = df_n[df_n['alerta_critica'] == 0]
        fila['abs_zscore_medio_flagged'] = round(flagged['z_score'].abs().mean(), 3) if len(flagged) else np.nan
        fila['abs_zscore_medio_no_flagged'] = round(no_flagged['z_score'].abs().mean(), 3) if len(no_flagged) else np.nan

        filas.append(fila)

    return pd.DataFrame(filas).sort_values('nodo').reset_index(drop=True)


# --- 5. MÓDULO DE REPORTING ---

def generate_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Genera un dashboard HTML interactivo con vista global promediada y detalle por nodo."""
    fig = go.Figure()
    nodos = df['nodo'].unique()

    df_global = df.groupby('fecha_hora').agg({
        'pml': 'mean',
        'alerta_critica': 'max'
    }).reset_index()

    fig.add_trace(go.Scatter(
        x=df_global['fecha_hora'], y=df_global['pml'], mode='lines',
        name='Promedio Red Monclova', line=dict(color='gray', width=1.5),
        visible=True
    ))

    criticas_globales = df_global[df_global['alerta_critica'] == 1]
    fig.add_trace(go.Scatter(
        x=criticas_globales['fecha_hora'], y=criticas_globales['pml'], mode='markers',
        name='Alerta Crítica (Red)',
        marker=dict(color='red', size=8, line=dict(width=1, color='black')),
        visible=True
    ))

    traces_per_node = 5

    for nodo in nodos:
        df_n = df[df['nodo'] == nodo]

        fig.add_trace(go.Scatter(
            x=df_n['fecha_hora'], y=df_n['pml'], mode='lines',
            name=f'{nodo} - PML', line=dict(color='lightgray', width=1), visible=False
        ))
        z_data = df_n[df_n['anom_zscore'] == 1]
        fig.add_trace(go.Scatter(
            x=z_data['fecha_hora'], y=z_data['pml'], mode='markers',
            name=f'{nodo} - Z-Score', marker=dict(color='orange', size=5), visible=False
        ))
        if_data = df_n[df_n['anom_isotree'] == 1]
        fig.add_trace(go.Scatter(
            x=if_data['fecha_hora'], y=if_data['pml'], mode='markers',
            name=f'{nodo} - I-Forest', marker=dict(color='blue', size=5), visible=False
        ))
        iqr_data = df_n[df_n['anom_iqr'] == 1]
        fig.add_trace(go.Scatter(
            x=iqr_data['fecha_hora'], y=iqr_data['pml'], mode='markers',
            name=f'{nodo} - IQR', marker=dict(color='green', size=5), visible=False
        ))
        criticas = df_n[df_n['alerta_critica'] == 1]
        fig.add_trace(go.Scatter(
            x=criticas['fecha_hora'], y=criticas['pml'], mode='markers',
            name=f'{nodo} - Crítica (2+)',
            marker=dict(color='red', size=8, line=dict(width=1, color='black')), visible=False
        ))

    buttons = []
    total_traces = 2 + (len(nodos) * traces_per_node)

    vis_global = [True, True] + [False] * (len(nodos) * traces_per_node)
    buttons.append(dict(
        label="Vista Global (Promedio)",
        method="update",
        args=[{"visible": vis_global}, {"title": "Promedio PML y Alertas - Zona de Carga Monclova"}]
    ))

    for i, nodo in enumerate(nodos):
        visibility = [False] * total_traces
        for j in range(traces_per_node):
            visibility[2 + (i * traces_per_node) + j] = True

        buttons.append(dict(
            label=f"Detalle: {nodo}",
            method="update",
            args=[{"visible": visibility}, {"title": f"Desglose del Tridente Analítico - Nodo {nodo}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.15, pad={"r": 10, "t": 10})],
        title="Promedio PML y Alertas - Zona de Carga Monclova",
        xaxis_title="Fecha", yaxis_title="Precio PML (MXN/MWh)",
        template="plotly_white",
        showlegend=True
    )

    fig.write_html(output_path)


def generar_vista_global_heatmap(df_maestro: pd.DataFrame, output_path: str) -> None:
    df_criticas = df_maestro[df_maestro['alerta_critica'] == 1].copy()
    df_criticas['mes_anio'] = df_criticas['fecha_hora'].dt.strftime('%Y-%m')

    tabla_calor = df_criticas.groupby(['nodo', 'mes_anio']).size().reset_index(name='conteo')
    matriz_calor = tabla_calor.pivot(index='nodo', columns='mes_anio', values='conteo').fillna(0)

    fig = px.imshow(
        matriz_calor,
        color_continuous_scale="YlOrRd",
        title="Termómetro de Riesgo: Frecuencia Mensual de Alertas Críticas",
        labels=dict(x="Mes", y="Nodo", color="N° Alertas")
    )

    fig.update_layout(template="plotly_white")
    fig.write_html(output_path)


def generar_vista_global_carriles(df_maestro: pd.DataFrame, output_path: str) -> None:
    df_criticas = df_maestro[df_maestro['alerta_critica'] == 1]

    fig = px.scatter(
        df_criticas,
        x="fecha_hora",
        y="nodo",
        size="pml",
        color="pml",
        color_continuous_scale="Reds",
        title="Mapa Operativo Global: Alertas Críticas en la Red Monclova",
        labels={"nodo": "Zona de Carga (Nodo)", "fecha_hora": "Línea de Tiempo", "pml": "PML (MXN)"}
    )

    fig.update_layout(template="plotly_white", height=500)
    fig.write_html(output_path)


# --- 6. PIPELINE PRINCIPAL (ORQUESTADOR) ---

def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)

    logger.info("=" * 50)
    logger.info("Iniciando Pipeline de Producción Arkham")
    logger.info(f"Config: nodos={config.nodos_input} | {config.fecha_inicio} -> {config.fecha_fin} | "
                f"sistema={config.sistema} proceso={config.proceso}")
    logger.info("=" * 50)
    tiempo_inicio = time.time()

    os.makedirs(config.output_dir_data, exist_ok=True)
    os.makedirs(config.output_dir_reports, exist_ok=True)

    logger.info("[1/4] Obteniendo datos de origen...")
    if config.csv_local_override:
        try:
            df_raw = pd.read_csv(config.csv_local_override)
            logger.info(f"Usando CSV local: {config.csv_local_override} ({len(df_raw)} filas)")
        except FileNotFoundError:
            logger.error(f"Archivo CSV no encontrado: {config.csv_local_override}")
            return
    else:
        catalogo = load_node_catalog(config.catalogo_nodos_path)
        nodo_bloques = resolve_nodes(config.nodos_input, catalogo, config.max_nodos_por_request)
        df_raw = extract_cenace_data(config, nodo_bloques)

    df_raw['fecha'] = pd.to_datetime(df_raw['fecha'], format='%Y-%m-%d')
    df_raw['hora'] = df_raw['hora'].astype(int) - 1
    df_raw['fecha_hora'] = df_raw['fecha'] + pd.to_timedelta(df_raw['hora'], unit='h')

    logger.info("[2/4] Aplicando Tridente Analítico (Z-Score, I-Forest, IQR)...")
    df_resultados = []
    nodos = df_raw['nodo'].unique()

    for nodo in nodos:
        df_actual = df_raw[df_raw['nodo'] == nodo].copy()
        df_actual['pml'] = pd.to_numeric(df_actual['pml'])
        df_actual = df_actual.sort_values('fecha_hora').reset_index(drop=True)

        df_actual = getGlobalZscore(df_actual, config.zscore_ventana_horas, config.zscore_umbral)
        df_actual = getIsolationForest(df_actual, config.iforest_contamination)
        df_actual = getIQR(df_actual, config.iqr_multiplicador)

        df_actual['alerta_critica'] = np.where(
            df_actual[['anom_zscore', 'anom_isotree', 'anom_iqr']].sum(axis=1) >= config.alerta_critica_min_metodos,
            1, 0
        )
        df_resultados.append(df_actual)

    df_maestro = pd.concat(df_resultados, ignore_index=True)

    logger.info("[3/4] Calculando métricas de evaluación...")
    df_metricas = compute_evaluation_metrics(df_maestro)
    metricas_path = os.path.join(config.output_dir_data, 'metricas_evaluacion.csv')
    df_metricas.to_csv(metricas_path, index=False)
    logger.info(f"Métricas exportadas a {metricas_path}")
    logger.info("\n" + df_metricas.to_string(index=False))

    logger.info("[4/4] Consolidando, exportando CSV y generando reportes...")
    columnas_finales = [
        'nodo', 'fecha', 'hora', 'fecha_hora',
        'pml', 'pml_ene', 'pml_per', 'pml_cng',
        'anom_zscore', 'anom_isotree', 'anom_iqr', 'alerta_critica'
    ]
    df_maestro[columnas_finales].to_csv(
        os.path.join(config.output_dir_data, 'resultado_anomalias.csv'), index=False
    )

    generate_dashboard(df_maestro, os.path.join(config.output_dir_reports, 'dashboard_anomalias.html'))
    generar_vista_global_heatmap(df_maestro, os.path.join(config.output_dir_reports, 'vista_global_heatmap.html'))
    generar_vista_global_carriles(df_maestro, os.path.join(config.output_dir_reports, 'vista_global_carriles.html'))

    tiempo_total = (time.time() - tiempo_inicio) / 60
    logger.info("=" * 50)
    logger.info(f"Pipeline ejecutado en {tiempo_total:.2f} minutos.")
    logger.info(f"Alertas Críticas globales detectadas: {df_maestro['alerta_critica'].sum()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()