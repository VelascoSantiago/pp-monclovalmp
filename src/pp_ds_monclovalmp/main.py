"""
Arkham Production Pipeline
=============================
Anomaly detection in the Local Marginal Price (PML) of SIN nodes
(National Electric System), via CENACE's SW-PML web service.

Usage:
    python main.py --nodes Monclova --start-date 2024-01-01 --end-date 2025-06-30
    python main.py --nodes 06AEO-115 06AHM-400 Huasteca --start-date 2024-01-01 --end-date 2025-06-30
    python main.py --use-local-csv monclova_pml_2024_2025.csv
"""

import argparse
import logging
from logging.handlers import RotatingFileHandler
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
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Set up logging
logger = logging.getLogger("arkham")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Handler for console output (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Handler for rotating log file output
log_suffix = datetime.now().strftime("%d%m%Y")
file_handler = RotatingFileHandler(f"logs/pipeline_{log_suffix}.log", maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --- 0. CONFIGURATION ---

@dataclass
class PipelineConfig:
    """Centralized pipeline configuration. Can be overridden via CLI arguments."""

    nodes_input: list[str] = field(default_factory=lambda: ["Monclova"])
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2025, 6, 30)
    system: str = "SIN"
    process: str = "MDA"
    format: str = "JSON"
    max_nodes_per_request: int = 20  # documented SW-PML limit
    max_days_per_request: int = 7    # documented SW-PML limit
    request_timeout_s: int = 30
    request_sleep_s: float = 1.0     # courtesy sleep between requests
    max_retries: int = 3

    # Detection method thresholds
    zscore_window_hours: int = 168
    zscore_threshold: float = 3.0
    iforest_contamination: float = 0.01
    iqr_multiplier: float = 3.0
    critical_alert_min_methods: int = 2

    # Paths
    catalog_nodes_path: str = str(Path(__file__).resolve().parent.parent.parent / "catalog" / "nodos_catalogo.csv")
    output_dir_data: str = "data/processed"
    output_dir_reports: str = "reports"
    local_csv_override: str | None = None  


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PML Anomaly Detection Pipeline (CENACE)")
    parser.add_argument("--nodes", nargs="+", default=["Monclova"],
                         help="Node keys (e.g. 06AEO-115) and/or load zones (e.g. Monclova). Free mix.")
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2025-06-30")
    parser.add_argument("--system", type=str, default="SIN", choices=["SIN", "BCA", "BCS"])
    parser.add_argument("--process", type=str, default="MDA", choices=["MDA", "MTR"])
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    parser.add_argument("--iforest-contamination", type=float, default=0.01)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--use-local-csv", type=str, default=None,
                         help="Path to an already extracted CSV; if provided, API is bypassed.")
    parser.add_argument("--catalog", type=str, default="catalog/nodos_catalogo.csv")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        nodes_input=args.nodes,
        start_date=datetime.strptime(args.start_date, "%Y-%m-%d").date(),
        end_date=datetime.strptime(args.end_date, "%Y-%m-%d").date(),
        system=args.system,
        process=args.process,
        zscore_threshold=args.zscore_threshold,
        iforest_contamination=args.iforest_contamination,
        iqr_multiplier=args.iqr_multiplier,
        local_csv_override=args.use_local_csv,
        catalog_nodes_path=args.catalog,
    )


# --- 1. NODE RESOLUTION (catalog -> node keys) ---

def load_node_catalog(catalog_path: str) -> pd.DataFrame:
    """Loads the NodosP catalog (columns: sistema, zona_carga, clave, nombre, municipio)."""
    df = pd.read_csv(catalog_path)
    df["clave"] = df["clave"].astype(str).str.upper().str.strip()
    df["zona_carga"] = df["zona_carga"].astype(str).str.upper().str.strip()
    return df


def resolve_nodes(inputs: list[str], catalog: pd.DataFrame, max_nodes: int = 20) -> list[list[str]]:
    """Resolves a mixed list of node keys and/or load zones to a deduplicated 
    list of node keys, chunked into blocks of size <= max_nodes."""
    valid_keys = set(catalog["clave"])
    resolved: list[str] = []
    not_found: list[str] = []

    for item in inputs:
        item_norm = item.strip().upper()
        if item_norm in valid_keys:
            resolved.append(item_norm)
            continue
        zone_match = catalog[catalog["zona_carga"] == item_norm]
        if not zone_match.empty:
            zone_nodes = zone_match["clave"].tolist()
            logger.info(f"Zone '{item}' resolved to {len(zone_nodes)} nodes: {zone_nodes}")
            resolved.extend(zone_nodes)
        else:
            not_found.append(item)

    if not_found:
        logger.warning(
            f"The following inputs did not match any node key or load zone "
            f"in the catalog and were ignored: {not_found}"
        )

    seen: set[str] = set()
    unique_resolved: list[str] = []
    for n in resolved:
        if n not in seen:
            seen.add(n)
            unique_resolved.append(n)

    if not unique_resolved:
        raise ValueError(
            "No nodes could be resolved from the given inputs. "
            "Verify node/zone names against the catalog."
        )

    blocks = [unique_resolved[i:i + max_nodes] for i in range(0, len(unique_resolved), max_nodes)]
    if len(blocks) > 1:
        logger.info(
            f"{len(unique_resolved)} resolved nodes exceed the limit of {max_nodes} per "
            f"request; chunked into {len(blocks)} blocks."
        )
    return blocks


# --- 2. EXTRACTION MODULES (EXTRACT) ---

def flatten_data(form: dict[str, Any]) -> pd.DataFrame:
    """Flattens the CENACE JSON response into a tabular DataFrame."""
    flattened_rows = []
    for node in form.get('Resultados', []):
        node_key = node['clv_nodo']
        for values in node['Valores']:
            values['nodo'] = node_key
            flattened_rows.append(values)
    return pd.DataFrame(flattened_rows)


def get_request_urls(start_date: date, end_date: date, node_list: list[str], system: str = 'SIN',
                    process: str = 'MDA', format: str = 'JSON', max_days: int = 7) -> list[str]:
    """Generates the list of URLs to iterate over the CENACE API in chunks of `max_days`."""
    nodes_str = ",".join(node_list)
    url_list = []
    current_date = start_date
    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=max_days - 1), end_date)
        f_ini = current_date.strftime("%Y/%m/%d")
        f_fin = chunk_end.strftime("%Y/%m/%d")
        url = f"https://ws01.cenace.gob.mx:8082/SWPML/SIM/{system}/{process}/{nodes_str}/{f_ini}/{f_fin}/{format}"
        url_list.append(url)
        current_date = chunk_end + timedelta(days=1)
    return url_list


def _build_session(max_retries: int) -> rq.Session:
    """HTTP Session with automatic exponential backoff retries."""
    session = rq.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def extract_cenace_data(config: PipelineConfig, node_blocks: list[list[str]]) -> pd.DataFrame:
    """Executes requests to the CENACE API and consolidates the data."""
    session = _build_session(config.max_retries)
    chunk_list = []
    failed_chunks = []
    empty_chunks = []

    for block_idx, nodes in enumerate(node_blocks, start=1):
        expected_nodes = set(nodes)
        urls = get_request_urls(
            config.start_date, config.end_date, nodes,
            system=config.system, process=config.process, format=config.format,
            max_days=config.max_days_per_request,
        )
        logger.info(f"Node block {block_idx}/{len(node_blocks)} ({len(nodes)} nodes): {len(urls)} requests")

        for url in urls:
            try:
                response = session.get(url, timeout=config.request_timeout_s)
            except rq.exceptions.RequestException as e:
                logger.error(f"Network failure after retries on {url}: {e}")
                failed_chunks.append(url)
                continue

            if response.status_code != 200:
                logger.error(f"HTTP {response.status_code} on {url}")
                failed_chunks.append(url)
                continue

            try:
                payload = response.json()
            except ValueError:
                logger.error(f"Invalid JSON response on {url}")
                failed_chunks.append(url)
                continue

            status = payload.get("status")
            if status in ("No Content",):
                logger.info(f"No data (status=No Content) for {url}")
                empty_chunks.append(url)
                continue
            if status not in ("OK", "Created"):
                logger.warning(f"Unexpected status '{status}' on {url}; attempting to process anyway")

            chunk_df = flatten_data(payload)

            if chunk_df.empty:
                logger.warning(
                    f"status='{status}' but no data arrived for any requested nodes "
                    f"on {url}: {sorted(expected_nodes)}"
                )
                empty_chunks.append(url)
                continue

            received_nodes = set(chunk_df['nodo'].unique())
            missing_nodes = expected_nodes - received_nodes
            if missing_nodes:
                logger.warning(
                    f"status='{status}' but data is missing for {sorted(missing_nodes)} "
                    f"on {url} (received: {sorted(received_nodes)})"
                )

            chunk_list.append(chunk_df)
            time.sleep(config.request_sleep_s)

    logger.info(
        f"Extraction complete: {len(chunk_list)} chunks with data, "
        f"{len(empty_chunks)} empty, {len(failed_chunks)} failed"
    )
    if failed_chunks:
        logger.warning(f"URLs that failed and are NOT in the final dataset: {failed_chunks}")

    if not chunk_list:
        raise RuntimeError("Extraction returned no usable data. Check the error log above.")
        sys.exit(1)

    return pd.concat(chunk_list, ignore_index=True)


# --- 3. DETECTION MODULES (TRANSFORM) ---

def apply_global_zscore(node_df: pd.DataFrame, window: int, threshold: float) -> pd.DataFrame:
    """Applies the Z-Score statistical method with a configurable rolling window."""
    node_df['rolling_mean'] = node_df['pml'].rolling(window=window, min_periods=1).mean()
    node_df['rolling_std'] = node_df['pml'].rolling(window=window, min_periods=1).std()

    node_df['z_score'] = np.where(
        node_df['rolling_std'] > 0,
        (node_df['pml'] - node_df['rolling_mean']) / node_df['rolling_std'],
        0
    )
    node_df['anom_zscore'] = np.where(node_df['z_score'].abs() > threshold, 1, 0)
    return node_df


def apply_isolation_forest(node_df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Applies Machine Learning (Isolation Forest) evaluating price and hour."""
    X = node_df[['pml', 'hora']]
    if_model = IsolationForest(contamination=contamination, random_state=42)
    predictions = if_model.fit_predict(X)
    node_df['anom_isotree'] = np.where(predictions == -1, 1, 0)
    return node_df


def apply_iqr(node_df: pd.DataFrame, multiplier: float = 3.0) -> pd.DataFrame:
    """Applies the Interquartile Range (IQR) method with a configurable multiplier."""
    Q1 = node_df['pml'].quantile(0.25)
    Q3 = node_df['pml'].quantile(0.75)
    iqr_val = Q3 - Q1

    lower_bound = Q1 - multiplier * iqr_val
    upper_bound = Q3 + multiplier * iqr_val

    node_df['anom_iqr'] = np.where(
        (node_df['pml'] < lower_bound) | (node_df['pml'] > upper_bound),
        1, 0
    )
    return node_df


# --- 4. EVALUATION METRICS MODULE (UNSUPERVISED) ---

def compute_evaluation_metrics(master_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates unsupervised evaluation metrics per node."""
    methods = ['anom_zscore', 'anom_isotree', 'anom_iqr']
    rows = []

    for node, n_df in master_df.groupby('nodo'):
        row: dict[str, Any] = {'node': node, 'n_hours': len(n_df)}

        for m in methods:
            row[f'rate_{m}'] = round(n_df[m].mean(), 4)

        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a, b = methods[i], methods[j]
                inter = int(((n_df[a] == 1) & (n_df[b] == 1)).sum())
                union = int(((n_df[a] == 1) | (n_df[b] == 1)).sum())
                row[f'jaccard_{a}_{b}'] = round(inter / union, 4) if union > 0 else np.nan

        row['critical_alert_rate'] = round(n_df['critical_alert'].mean(), 4)

        flagged = n_df[n_df['critical_alert'] == 1]
        no_flagged = n_df[n_df['critical_alert'] == 0]
        row['mean_abs_zscore_flagged'] = round(flagged['z_score'].abs().mean(), 3) if len(flagged) else np.nan
        row['mean_abs_zscore_no_flagged'] = round(no_flagged['z_score'].abs().mean(), 3) if len(no_flagged) else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values('node').reset_index(drop=True)


# --- 5. REPORTING MODULE ---

def generate_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Generates an interactive HTML dashboard with a global averaged view and node breakdown."""
    fig = go.Figure()
    nodes = df['nodo'].unique()

    global_df = df.groupby('fecha_hora').agg({
        'pml': 'mean',
        'critical_alert': 'max'
    }).reset_index()

    fig.add_trace(go.Scatter(
        x=global_df['fecha_hora'], y=global_df['pml'], mode='lines',
        name='Monclova Grid Average', line=dict(color='gray', width=1.5),
        visible=True
    ))

    global_criticals = global_df[global_df['critical_alert'] == 1]
    fig.add_trace(go.Scatter(
        x=global_criticals['fecha_hora'], y=global_criticals['pml'], mode='markers',
        name='Critical Alert (Grid)',
        marker=dict(color='red', size=8, line=dict(width=1, color='black')),
        visible=True
    ))

    traces_per_node = 5

    for node in nodes:
        n_df = df[df['nodo'] == node]

        fig.add_trace(go.Scatter(
            x=n_df['fecha_hora'], y=n_df['pml'], mode='lines',
            name=f'{node} - PML', line=dict(color='lightgray', width=1), visible=False
        ))
        z_data = n_df[n_df['anom_zscore'] == 1]
        fig.add_trace(go.Scatter(
            x=z_data['fecha_hora'], y=z_data['pml'], mode='markers',
            name=f'{node} - Z-Score', marker=dict(color='orange', size=5), visible=False
        ))
        if_data = n_df[n_df['anom_isotree'] == 1]
        fig.add_trace(go.Scatter(
            x=if_data['fecha_hora'], y=if_data['pml'], mode='markers',
            name=f'{node} - I-Forest', marker=dict(color='blue', size=5), visible=False
        ))
        iqr_data = n_df[n_df['anom_iqr'] == 1]
        fig.add_trace(go.Scatter(
            x=iqr_data['fecha_hora'], y=iqr_data['pml'], mode='markers',
            name=f'{node} - IQR', marker=dict(color='green', size=5), visible=False
        ))
        criticals = n_df[n_df['critical_alert'] == 1]
        fig.add_trace(go.Scatter(
            x=criticals['fecha_hora'], y=criticals['pml'], mode='markers',
            name=f'{node} - Critical (2+)',
            marker=dict(color='red', size=8, line=dict(width=1, color='black')), visible=False
        ))

    buttons = []
    total_traces = 2 + (len(nodes) * traces_per_node)

    vis_global = [True, True] + [False] * (len(nodes) * traces_per_node)
    buttons.append(dict(
        label="Global View (Average)",
        method="update",
        args=[{"visible": vis_global}, {"title": "PML Average and Alerts - Monclova Load Zone"}]
    ))

    for i, node in enumerate(nodes):
        visibility = [False] * total_traces
        for j in range(traces_per_node):
            visibility[2 + (i * traces_per_node) + j] = True

        buttons.append(dict(
            label=f"Detail: {node}",
            method="update",
            args=[{"visible": visibility}, {"title": f"Analytical Trident Breakdown - Node {node}"}]
        ))

    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.0, y=1.15, pad={"r": 10, "t": 10})],
        title="PML Average and Alerts - Monclova Load Zone",
        xaxis_title="Date", yaxis_title="PML Price (MXN/MWh)",
        template="plotly_white",
        showlegend=True
    )

    fig.write_html(output_path)


def generate_global_heatmap_view(master_df: pd.DataFrame, output_path: str) -> None:
    critical_df = master_df[master_df['critical_alert'] == 1].copy()
    critical_df['month_year'] = critical_df['fecha_hora'].dt.strftime('%Y-%m')

    heatmap_table = critical_df.groupby(['nodo', 'month_year']).size().reset_index(name='count')
    heatmap_matrix = heatmap_table.pivot(index='nodo', columns='month_year', values='count').fillna(0)

    fig = px.imshow(
        heatmap_matrix,
        color_continuous_scale="YlOrRd",
        title="Risk Thermometer: Monthly Frequency of Critical Alerts",
        labels=dict(x="Month", y="Node", color="Alert Count")
    )

    fig.update_layout(template="plotly_white")
    fig.write_html(output_path)


def generate_global_lanes_view(master_df: pd.DataFrame, output_path: str) -> None:
    critical_df = master_df[master_df['critical_alert'] == 1]

    fig = px.scatter(
        critical_df,
        x="fecha_hora",
        y="nodo",
        size="pml",
        color="pml",
        color_continuous_scale="Reds",
        title="Global Operational Map: Critical Alerts in Monclova Grid",
        labels={"nodo": "Load Zone (Node)", "fecha_hora": "Timeline", "pml": "PML (MXN)"}
    )

    fig.update_layout(template="plotly_white", height=500)
    fig.write_html(output_path)


# --- 6. MAIN PIPELINE (ORCHESTRATOR) ---

def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)

    logger.info("=" * 50)
    logger.info("Starting Arkham Production Pipeline")
    logger.info(f"Config: nodes={config.nodes_input} | {config.start_date} -> {config.end_date} | "
                f"system={config.system} process={config.process}")
    logger.info("=" * 50)
    start_time = time.time()
    date_suffix = datetime.now().strftime("%d%m%Y")

    os.makedirs(config.output_dir_data, exist_ok=True)
    os.makedirs(config.output_dir_reports, exist_ok=True)

    logger.info("[1/4] Fetching source data...")
    if config.local_csv_override:
        try:
            raw_df = pd.read_csv(config.local_csv_override)
            logger.info(f"Using local CSV: {config.local_csv_override} ({len(raw_df)} rows)")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {config.local_csv_override}")
            sys.exit(1) # <--- AQUÍ
    else:
        catalog = load_node_catalog(config.catalog_nodes_path)
        node_blocks = resolve_nodes(config.nodes_input, catalog, config.max_nodes_per_request)
        raw_df = extract_cenace_data(config, node_blocks)

    raw_df['fecha'] = pd.to_datetime(raw_df['fecha'], format='%Y-%m-%d')
    raw_df['hora'] = raw_df['hora'].astype(int) - 1
    raw_df['fecha_hora'] = raw_df['fecha'] + pd.to_timedelta(raw_df['hora'], unit='h')

    logger.info("[2/4] Applying Analytical Trident (Z-Score, I-Forest, IQR)...")
    results_dfs = []
    nodes = raw_df['nodo'].unique()

    for node in nodes:
        current_df = raw_df[raw_df['nodo'] == node].copy()
        current_df['pml'] = pd.to_numeric(current_df['pml'])
        current_df = current_df.sort_values('fecha_hora').reset_index(drop=True)

        current_df = apply_global_zscore(current_df, config.zscore_window_hours, config.zscore_threshold)
        current_df = apply_isolation_forest(current_df, config.iforest_contamination)
        current_df = apply_iqr(current_df, config.iqr_multiplier)

        current_df['critical_alert'] = np.where(
            current_df[['anom_zscore', 'anom_isotree', 'anom_iqr']].sum(axis=1) >= config.critical_alert_min_methods,
            1, 0
        )
        results_dfs.append(current_df)

    master_df = pd.concat(results_dfs, ignore_index=True)

    logger.info("[3/4] Computing evaluation metrics...")
    metrics_df = compute_evaluation_metrics(master_df)
    metrics_path = os.path.join(config.output_dir_data, f'evaluation_metrics_{date_suffix}.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics exported to {metrics_path}")
    logger.info("\n" + metrics_df.to_string(index=False))

    logger.info("[4/4] Consolidating, exporting CSV and generating reports...")
    final_columns = [
        'nodo', 'fecha', 'hora', 'fecha_hora',
        'pml', 'pml_ene', 'pml_per', 'pml_cng',
        'anom_zscore', 'anom_isotree', 'anom_iqr', 'critical_alert'
    ]
    master_df[final_columns].to_csv(
        os.path.join(config.output_dir_data, f'anomaly_results_{date_suffix}.csv'), index=False
    )

    generate_dashboard(master_df, os.path.join(config.output_dir_reports, f'anomaly_dashboard_{date_suffix}.html'))
    generate_global_heatmap_view(master_df, os.path.join(config.output_dir_reports, f'global_heatmap_view_{date_suffix}.html'))
    generate_global_lanes_view(master_df, os.path.join(config.output_dir_reports, f'global_lanes_view_{date_suffix}.html'))

    total_time = (time.time() - start_time) / 60
    logger.info("=" * 50)
    logger.info(f"Pipeline executed in {total_time:.2f} minutes.")
    logger.info(f"Global Critical Alerts detected: {master_df['critical_alert'].sum()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()