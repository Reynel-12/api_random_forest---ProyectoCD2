from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# --- FUNCIONES DE LÓGICA (Tus funciones originales mejoradas) ---

def calcular_lead_time_real(dias_str):
    mapa_dias = {'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4, 'sábado': 5, 'domingo': 6}
    dias = [mapa_dias[d.strip().lower()] for d in str(dias_str).split(',') if d.strip().lower() in mapa_dias]
    if not dias: return 7
    dias.sort()
    huecos = []
    for i in range(len(dias)):
        if i < len(dias) - 1:
            huecos.append(dias[i+1] - dias[i])
        else:
            huecos.append(7 - dias[i] + dias[0])
    return max(huecos)

def sugerir_compra_profesional(row, mae):
    factor_espera = row["lead_time_dias"] / 7
    demanda_critica = row["demanda_predicha_7dias"] * factor_espera
    seguridad = (demanda_critica * 0.20) + (mae * factor_espera)
    punto_reorden = demanda_critica + seguridad
    
    if row["stock_actual"] < punto_reorden:
        cantidad = (punto_reorden * 1.2) - row["stock_actual"]
        return max(0, int(round(cantidad)))
    return 0

# --- ENDPOINT PRINCIPAL ---

@app.route('/predecir_compra', methods=['POST'])
def predecir_compra():
    try:
        data = request.get_json()

        # --- BLOQUE DE SEGURIDAD: Validación de Campos ---
        campos_requeridos = ['ventas', 'stock', 'proveedores']
        for campo in campos_requeridos:
            if campo not in data or not data[campo]:
                return jsonify({
                    "status": "error", 
                    "message": f"Falta el campo requerido o está vacío: {campo}"
                }), 400

        # Verificar columnas mínimas en ventas
        cols_ventas = ['codigo_producto', 'fecha', 'cantidad_vendida']
        if not all(col in data['ventas'][0] for col in cols_ventas):
            return jsonify({"status": "error", "message": "Estructura de 'ventas' incorrecta"}), 400
        # ------------------------------------------------
        
        # 1. Convertir JSON a DataFrames
        df_ventas = pd.DataFrame(data['ventas'])
        df_stock = pd.DataFrame(data['stock'])
        df_proveedores = pd.DataFrame(data['proveedores'])
        
        df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])
        
        # 2. Procesamiento de Proveedores y Lead Time
        df_proveedores['lead_time_dias'] = df_proveedores['dias_reposicion'].apply(calcular_lead_time_real)
        lead_time_map = df_proveedores.groupby('proveedor_id')['lead_time_dias'].mean().reset_index()
        
        # 3. Feature Engineering
        ventas_semanales = df_ventas.groupby(['codigo_producto', pd.Grouper(key='fecha', freq='W')]).agg({
            'cantidad_vendida': 'sum'
        }).reset_index().sort_values(['codigo_producto', 'fecha'])

        ventas_semanales['venta_semana_anterior'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(1)
        ventas_semanales['venta_hace_2_semanas'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(2)
        ventas_semanales['tendencia'] = ventas_semanales['cantidad_vendida'] - ventas_semanales['venta_semana_anterior']
        ventas_semanales['mes'] = ventas_semanales['fecha'].dt.month
        ventas_semanales['demanda_proxima_semana'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(-1)

        # 4. Entrenamiento rápido (On-the-fly)
        data_ml = ventas_semanales.dropna().merge(df_stock, on="codigo_producto", how="left").fillna(0)
        data_ml = data_ml.merge(lead_time_map, on="proveedor_id", how="left").fillna(7)
        
        features = ["cantidad_vendida", "venta_semana_anterior", "venta_hace_2_semanas", "tendencia", "mes", "stock_minimo", "lead_time_dias"]
        X = data_ml[features]
        y = data_ml["demanda_proxima_semana"]

        modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        modelo.fit(X, y)
        mae = mean_absolute_error(y, modelo.predict(X))

        # 5. Predicción para la última semana
        ultima_foto = ventas_semanales.groupby('codigo_producto').last().reset_index()
        ultima_foto = ultima_foto.merge(df_stock, on="codigo_producto", how="left").fillna(0)
        ultima_foto = ultima_foto.merge(lead_time_map, on="proveedor_id", how="left").fillna(7)
        
        X_predict = ultima_foto[features]
        ultima_foto["demanda_predicha_7dias"] = modelo.predict(X_predict)
        ultima_foto["cantidad_a_comprar"] = ultima_foto.apply(lambda r: sugerir_compra_profesional(r, mae), axis=1)

        # 6. Preparar Respuesta
        resultado = ultima_foto[["codigo_producto", "cantidad_a_comprar"]].to_dict(orient='records')
        
        return jsonify({
            "status": "success",
            "mae": round(mae, 2),
            "recomendaciones": resultado
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)