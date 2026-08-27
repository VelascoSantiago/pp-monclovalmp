import pytest
import pandas as pd
from src.pp_ds_monclovalmp.main import flatten_data

def test_flatten_data():
    """Prueba que el aplanado del JSON de CENACE funcione correctamente."""
    mock_json = {
        "status": "OK",
        "Resultados": [
            {
                "clv_nodo": "06MON-115",
                "Valores": [
                    {"fecha": "2024-01-01", "hora": "1", "pml": "500.5"}
                ]
            }
        ]
    }
    
    df = flatten_data(mock_json)
    
    assert not df.empty, "El DataFrame no debería estar vacío"
    assert len(df) == 1, "Debería haber exactamente 1 fila"
    assert df.iloc[0]["nodo"] == "06MON-115", "La clave del nodo no coincide"
    assert df.iloc[0]["pml"] == "500.5", "El valor del PML no se aplanó correctamente"