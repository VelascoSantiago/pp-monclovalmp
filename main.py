import pandas as pd
import numpy as np
import time
import os
import requests as rq
from datetime import date, timedelta
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- 0. MÓDULOS DE EXTRACCIÓN (EXTRACT) ---

def flattenData(form: dict) -> pd.DataFrame:
    """Aplana el JSON de respuesta del CENACE a un DataFrame tabular."""
    flattenedRows = []
    for nodo in form['Resultados']:
        clave = nodo['clv_nodo']
        for valores in nodo['Valores']:
            valores['nodo'] = clave 
            flattenedRows.append(valores)
    return pd.DataFrame(flattenedRows)

def getRequestURLs(startDate: date, endDate: date, lista_nodos: str, sistema: str='SIN', proceso: str='MDA', formato: str='JSON') -> list:
    """Genera la lista de URLs para iterar sobre la API del CENACE en bloques de 7 días."""
    URLlist = []
    current_date = startDate
    while current_date <= endDate: 
        chunk_end = current_date + timedelta(days=6)
        if chunk_end > endDate:
            chunk_end = endDate
        f_ini = current_date.strftime("%Y/%m/%d")
        f_fin = chunk_end.strftime("%Y/%m/%d")
        url = f"https://ws01.cenace.gob.mx:8082/SWPML/SIM/{sistema}/{proceso}/{lista_nodos}/{f_ini}/{f_fin}/{formato}"
        URLlist.append(url)
        current_date = chunk_end + timedelta(days=1)
    return URLlist

def extract_cenace_data(start_date: date, end_date: date, nodos_str: str) -> pd.DataFrame:
    """Ejecuta las peticiones a la API del CENACE y consolida los datos."""
    urls = getRequestURLs(start_date, end_date, nodos_str)
    lista_chunks = []
    for link in urls:
        respuesta = rq.get(link)
        if respuesta.status_code == 200:
            lista_chunks.append(flattenData(respuesta.json()))
        time.sleep(2) # Respetar límites de la API
    return pd.concat(lista_chunks, ignore_index=True)


# --- 1. MÓDULOS DE DETECCIÓN (TRANSFORM) ---

def getGlobalZscore(df_nodo: pd.DataFrame) -> pd.DataFrame:
    """Aplica el método estadístico Z-Score con ventana móvil de 168 horas."""
    ventana = 168
    df_nodo['media_movil'] = df_nodo['pml'].rolling(window=ventana, min_periods=1).mean()
    df_nodo['std_movil'] = df_nodo['pml'].rolling(window=ventana, min_periods=1).std()
    
    df_nodo['z_score'] = np.where(
        df_nodo['std_movil'] > 0,
        (df_nodo['pml'] - df_nodo['media_movil']) / df_nodo['std_movil'],
        0
    )
    df_nodo['anom_zscore'] = np.where(df_nodo['z_score'].abs() > 3, 1, 0)
    return df_nodo

def getIsolationForest(df_nodo: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """Aplica Machine Learning (Isolation Forest) evaluando precio y hora."""
    X = df_nodo[['pml', 'hora']]
    modelo_if = IsolationForest(contamination=contamination, random_state=42)
    predicciones = modelo_if.fit_predict(X)
    df_nodo['anom_isotree'] = np.where(predicciones == -1, 1, 0)
    return df_nodo

def getIQR(df_nodo: pd.DataFrame) -> pd.DataFrame:
    """Aplica el método de Rango Intercuartílico (IQR) con multiplicador 3.0."""
    Q1 = df_nodo['pml'].quantile(0.25)
    Q3 = df_nodo['pml'].quantile(0.75)
    IQR_val = Q3 - Q1
    
    limite_inferior = Q1 - 3.0 * IQR_val
    limite_superior = Q3 + 3.0 * IQR_val
    
    df_nodo['anom_iqr'] = np.where(
        (df_nodo['pml'] < limite_inferior) | (df_nodo['pml'] > limite_superior), 
        1, 0
    )
    return df_nodo

# --- 2. MÓDULO DE REPORTING ---

def generate_dashboard(df: pd.DataFrame, output_path: str) -> None:
    """Genera un dashboard HTML interactivo con menú desplegable por nodo."""
    fig = go.Figure()
    nodos = df['nodo'].unique()
    
    for nodo in nodos:
        df_n = df[df['nodo'] == nodo]
        fig.add_trace(go.Scatter(
            x=df_n['fecha_hora'], y=df_n['pml'], mode='lines', 
            name=f'{nodo} - PML', line=dict(color='lightgray'), 
            visible=(nodo == nodos[0])
        ))
        criticas = df_n[df_n['alerta_critica'] == 1]
        fig.add_trace(go.Scatter(
            x=criticas['fecha_hora'], y=criticas['pml'], mode='markers', 
            name=f'{nodo} - Alerta', marker=dict(color='red', size=8, line=dict(width=1, color='black')), 
            visible=(nodo == nodos[0])
        ))

    buttons = []
    for i, nodo in enumerate(nodos):
        visibility = [False] * (len(nodos) * 2)
        visibility[i*2] = True
        visibility[i*2+1] = True
        buttons.append(dict(
            label=nodo, method="update", 
            args=[{"visible": visibility}, {"title": f"Alertas Críticas PML - Nodo {nodo}"}]
        ))
        
    fig.update_layout(
        updatemenus=[dict(active=0, buttons=buttons, x=0.15, y=1.15, pad={"r": 10, "t": 10})],
        title=f"Alertas Críticas PML - Nodo {nodos[0]}",
        xaxis_title="Fecha", yaxis_title="Precio PML (MXN/MWh)",
        template="plotly_white"
    )
    fig.write_html(output_path)

# --- 3. PIPELINE PRINCIPAL (ORQUESTADOR) ---

def main():
    print("="*50)
    print("[SYSTEM] Iniciando Pipeline de Producción Arkham")
    print("="*50)
    tiempo_inicio = time.time()
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("[1/3] Obteniendo datos de origen...")
    
    # OPCIÓN A: Extracción en vivo desde la API (Comentada para no saturar al correr pruebas)
    # nodos_str = '06AEO-115,06AHM-400,06FRO-115,06GRA-115,06MON-115,06MON-230,06RGS-115,06TKS-115,06XOC-115'
    # df_raw = extract_cenace_data(date(2024,1,1), date(2025,6,30), nodos_str)
    
    # OPCIÓN B: Lectura de archivo local (Por practicidad y velocidad en producción)
    # Asumiendo que el archivo está en la raíz o en data/raw/
    try:
        df_raw = pd.read_csv('monclova_pml_2024_2025.csv') 
    except FileNotFoundError:
        print("[ERROR] Archivo CSV no encontrado. Verifique la ruta.")
        return

    # Transformación del tiempo (Feature Engineering)
    df_raw['fecha'] = pd.to_datetime(df_raw['fecha'], format='%Y-%m-%d')
    df_raw['hora'] = df_raw['hora'].astype(int) - 1
    df_raw['fecha_hora'] = df_raw['fecha'] + pd.to_timedelta(df_raw['hora'], unit='h')
    
    print("[2/3] Aplicando Tridente Analítico (Z-Score, I-Forest, IQR)...")
    df_resultados = []
    nodos = df_raw['nodo'].unique()
    
    for nodo in nodos:
        df_actual = df_raw[df_raw['nodo'] == nodo].copy()
        df_actual['pml'] = pd.to_numeric(df_actual['pml'])
        df_actual = df_actual.sort_values('fecha_hora').reset_index(drop=True)
        
        df_actual = getGlobalZscore(df_actual)
        df_actual = getIsolationForest(df_actual)
        df_actual = getIQR(df_actual)
        
        df_actual['alerta_critica'] = np.where(
            df_actual[['anom_zscore', 'anom_isotree', 'anom_iqr']].sum(axis=1) >= 2, 1, 0
        )
        df_resultados.append(df_actual)

    print("[3/3] Consolidando, exportando CSV y generando Dashboard HTML...")
    df_maestro = pd.concat(df_resultados, ignore_index=True)
    
    # Exportar CSV limpio

    columnas_finales = [
        'nodo', 'fecha', 'hora', 'fecha_hora', 
        'pml', 'pml_ene', 'pml_per', 'pml_cng', 
        'anom_zscore', 'anom_isotree', 'anom_iqr', 'alerta_critica'
    ]
    
    df_maestro[columnas_finales].to_csv('data/processed/resultado_anomalias.csv', index=False)
    generate_dashboard(df_maestro, 'reports/dashboard_anomalias.html')
    
    tiempo_total = (time.time() - tiempo_inicio) / 60
    print("="*50)
    print(f"[SUCCESS] Pipeline ejecutado en {tiempo_total:.2f} minutos.")
    print(f"[INFO] Alertas Críticas globales detectadas: {df_maestro['alerta_critica'].sum()}")
    print("="*50)

if __name__ == "__main__":
    main()