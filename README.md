# **Data Science Challenge: LMP Anomalies - Monclova Zone**

This repository contains a production-ready ETL pipeline designed for the extraction, transformation, anomaly detection, and visualization of Local Marginal Prices (LMP) from the Wholesale Electricity Market (CENACE).
The project was built with a software engineering and data science approach, ensuring modularity, resilience to network failures, and strict reproducibility.

## **Methodology: The "Analytical Trident"**

Due to the asymmetry and seasonality of the electricity market (especially during the summer months), relying on a single statistical model can generate an excess of false positives. This pipeline implements a consensus ensemble:

> 1. **Robust IQR (x3.0):** Establishes a static baseline to detect historically extreme values, filtering out seasonal noise.
> 2. **Rolling Z-Score (168h):** Dynamic control to detect price shocks based exclusively on the volatility of the last 7 days.
> 3. **Isolation Forest (Machine Learning):** Unsupervised algorithm that detects multivariate inconsistencies by cross-referencing the node price with the time of day.

**Business Rule:** An alert is categorized as "Critical" only if 2 or more models agree on their classification. Additionally, the pipeline computes unsupervised evaluation metrics (such as the Jaccard Index) to compare the level of agreement between models.

## **Repository Structure**

The project follows a production-oriented structure:

> * catalog/: Master catalogs for dynamic resolution of load zones and nodes.
> * data/raw/: Raw data storage (e.g., previous CSV extractions).
> * data/processed/: Pipeline results and metric calculations.
> * notebooks/: Environments for experimentation and prototyping.
> * reports/: Interactive HTML dashboards.
> * src/pp_ds_monclovalmp/: Source code for the orchestrator and models (main.py).
> * presentation/: Executive Slide Deck with results and technical roadmap.

## **Reproducibility and Requirements**

This project uses uv as a package and environment manager to guarantee exact dependency parity via the uv.lock file.

> 1. Install uv (if not found on the system):
>    curl -LsSf https://astral.sh/uv/install.sh | sh
> 2. Clone the repository and sync the environment (strict resolution):
>    uv sync --frozen

## **Pipeline Usage**

The pipeline includes a Command Line Interface (CLI) that allows you to interact directly with the CENACE SW-PML service or consume a local file.

| Parameter / Flag | What it controls | Default Value | When to modify it |
| :--- | :--- | :--- | :--- |
| `--nodes` | Nodes or Load Zones to extract. Supports multiple mixed values. | `Monclova` | When you need to analyze other regions or add specific nodes (e.g., `--nodes Huasteca 06AEO-115`). |
| `--start-date` / `--end-date` | Date range for extraction (`YYYY-MM-DD` format). | `2024-01-01` / `2025-06-30` | In periodic runs. For example, a weekly run would only request the last 7 days. |
| `--system` | Interconnected System to query (`SIN`, `BCA`, `BCS`). | `SIN` | Only if the analysis expands to the Baja California grid. |
| `--process` | Market type to query: `MDA` (Day-Ahead) or `MTR` (Real-Time). | `MDA` | For real-time prices. **Note:** `MTR` data has a publishing lag of ~7 days by CENACE. |
| `--zscore-threshold` | Standard deviations required to flag an anomaly. | `3.0` | To adjust the sensitivity of the rolling model based on observed metrics. |
| `--iforest-contamination` | Expected proportion of anomalies in the dataset. | `0.01` (1%) | If you consider there are more or less than 1% of true anomalies in your data. |
| `--iqr-multiplier` | Width of the normal historical range before marking an outlier. | `3.0` | To tighten or loosen the static detection of extreme spikes. |
| `--use-local-csv` | Path to a local dataset. When used, **it skips API calls**. | *None* | For rapid iteration during development or local testing. |
| `--catalog` | Path to the node catalog CSV file. | `catalog/nodos_catalogo.csv` | Only if you modify the repository's directory structure. |

To run the pipeline using a local dataset (recommended mode for CI/CD and rapid evaluation):
uv run python src/pp_ds_monclovalmp/main.py --use-local-csv data/raw/monclova_pml_2024_2025.csv

To extract live data by querying the API for the entire Monclova zone over a date range:
uv run python src/pp_ds_monclovalmp/main.py --nodes Monclova --start-date 2024-01-01 --end-date 2025-06-30

To view model configuration parameters (thresholds, windows) and API options:
uv run python src/pp_ds_monclovalmp/main.py --help

## **Generated Deliverables**

Upon completion, the script automatically generates the following artifacts:

> 1. **Data (data/processed/):**
   * Master file with anomaly flags.
   * CSV with model evaluation metrics per node.
> 2. **Visual Reports (reports/):**
   * *Node Dashboard:* Interactive time series plot with a dropdown menu to inspect the baseline and the analytical trident per node.
   * *Risk Thermometer:* Heatmap of the monthly frequency of critical alerts.
   * *Global Operational Map:* Categorical scatter plot to visually identify systemic versus local failures in the grid.