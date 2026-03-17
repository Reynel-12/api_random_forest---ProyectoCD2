import requests
import json

# Datos de ejemplo para probar la API
data = {
    "ventas": [
        {"codigo_producto": "PROD001", "fecha": "2023-01-01", "cantidad_vendida": 10},
        {"codigo_producto": "PROD001", "fecha": "2023-01-08", "cantidad_vendida": 15},
        {"codigo_producto": "PROD001", "fecha": "2023-01-15", "cantidad_vendida": 12},
        {"codigo_producto": "PROD001", "fecha": "2023-01-22", "cantidad_vendida": 8},
        {"codigo_producto": "PROD001", "fecha": "2023-01-29", "cantidad_vendida": 20},
        {"codigo_producto": "PROD002", "fecha": "2023-01-01", "cantidad_vendida": 5},
        {"codigo_producto": "PROD002", "fecha": "2023-01-08", "cantidad_vendida": 7},
        {"codigo_producto": "PROD002", "fecha": "2023-01-15", "cantidad_vendida": 6},
        {"codigo_producto": "PROD002", "fecha": "2023-01-22", "cantidad_vendida": 4},
        {"codigo_producto": "PROD002", "fecha": "2023-01-29", "cantidad_vendida": 9}
    ],
    "stock": [
        {"codigo_producto": "PROD001", "stock_actual": 50, "stock_minimo": 10, "proveedor_id": "PROV001"},
        {"codigo_producto": "PROD002", "stock_actual": 30, "stock_minimo": 5, "proveedor_id": "PROV002"}
    ],
    "proveedores": [
        {"proveedor_id": "PROV001", "dias_reposicion": "lunes, miercoles, viernes"},
        {"proveedor_id": "PROV002", "dias_reposicion": "martes, jueves"}
    ]
}

# Hacer la petición a la API
url = "http://localhost:5000/predecir_compra"
response = requests.post(url, json=data)

if response.status_code == 200:
    result = response.json()
    print("Métricas del modelo:")
    print(f"MAE: {result['mae']}")
    print(f"MSE: {result['mse']}")
    print(f"RMSE: {result['rmse']}")
    print(f"R²: {result['r2']}")
    print("\nImportancia de features:")
    for feature, importance in result['importancia_features'].items():
        print(f"{feature}: {importance}")
    print("\nRecomendaciones:")
    for rec in result['recomendaciones']:
        print(rec)
else:
    print(f"Error: {response.status_code} - {response.text}")