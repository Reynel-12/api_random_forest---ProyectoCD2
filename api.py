from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# --- FUNCIONES DE LÓGICA ---

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


def calcular_features_temporales(df_ventas_producto, fecha_referencia):
    """
    Calcula features que capturan el comportamiento temporal de ventas de un producto.
    Esto permite al modelo entender si un producto se vende poco, mucho, o lleva tiempo inactivo.
    
    Features generadas:
    - dias_desde_ultima_venta: días transcurridos desde la última venta registrada
    - semanas_sin_venta_consecutivas: semanas seguidas con cero ventas al final del historial
    - frecuencia_venta_pct: % de semanas con al menos una venta (0.0 a 1.0)
    - velocidad_venta_activa: promedio de ventas SOLO en semanas donde hubo ventas
    - cv_ventas: coeficiente de variación (irregularidad de demanda)
    - tendencia_reciente: diferencia promedio entre las últimas 4 semanas
    """
    if df_ventas_producto.empty:
        return {
            'dias_desde_ultima_venta': 999,
            'semanas_sin_venta_consecutivas': 52,
            'frecuencia_venta_pct': 0.0,
            'velocidad_venta_activa': 0.0,
            'cv_ventas': 0.0,
            'tendencia_reciente': 0.0,
        }

    # Agrupar en semanas para análisis consistente
    semanas = (
        df_ventas_producto
        .groupby(pd.Grouper(key='fecha', freq='W'))['cantidad_vendida']
        .sum()
        .sort_index()
    )

    # 1. Días desde última venta real
    ventas_con_cantidad = df_ventas_producto[df_ventas_producto['cantidad_vendida'] > 0]
    if not ventas_con_cantidad.empty:
        ultima_venta = ventas_con_cantidad['fecha'].max()
        dias_desde_ultima = (fecha_referencia - ultima_venta).days
    else:
        dias_desde_ultima = 999  # Nunca vendido

    # 2. Semanas consecutivas sin venta al final del historial
    semanas_sin_venta = 0
    for val in reversed(semanas.values):
        if val == 0:
            semanas_sin_venta += 1
        else:
            break

    # 3. Frecuencia de venta (% semanas activas)
    total_semanas = len(semanas)
    semanas_activas = (semanas > 0).sum()
    frecuencia_pct = semanas_activas / total_semanas if total_semanas > 0 else 0.0

    # 4. Velocidad de venta en semanas activas (excluye ceros para no subestimar productos estacionales)
    ventas_en_semanas_activas = semanas[semanas > 0]
    velocidad_activa = ventas_en_semanas_activas.mean() if len(ventas_en_semanas_activas) > 0 else 0.0

    # 5. Coeficiente de variación (mide irregularidad de la demanda)
    std_ventas = semanas.std()
    mean_ventas = semanas.mean()
    cv = (std_ventas / mean_ventas) if mean_ventas > 0 else 0.0

    # 6. Tendencia reciente: pendiente de las últimas 4 semanas
    ultimas_4 = semanas.tail(4).values
    if len(ultimas_4) >= 2:
        tendencia = float(np.polyfit(range(len(ultimas_4)), ultimas_4, 1)[0])
    else:
        tendencia = 0.0

    return {
        'dias_desde_ultima_venta': dias_desde_ultima,
        'semanas_sin_venta_consecutivas': semanas_sin_venta,
        'frecuencia_venta_pct': round(frecuencia_pct, 4),
        'velocidad_venta_activa': round(velocidad_activa, 4),
        'cv_ventas': round(cv, 4),
        'tendencia_reciente': round(tendencia, 4),
    }


def sugerir_compra_profesional(row, mae):
    """
    Calcula la cantidad a comprar usando punto de reorden dinámico.
    Incorpora penalización por inactividad: si el producto lleva muchos días
    sin venderse, se reduce la sugerencia de compra.
    """
    factor_espera = row["lead_time_dias"] / 7
    demanda_critica = row["demanda_predicha_7dias"] * factor_espera
    seguridad = (demanda_critica * 0.20) + (mae * factor_espera)
    punto_reorden = demanda_critica + seguridad

    # Penalización por inactividad: si el producto lleva más de 30 días sin
    # venderse, se aplica un descuento progresivo a la sugerencia de compra.
    # Esto evita comprar stock de productos que ya no rotan.
    dias_inactivo = row.get('dias_desde_ultima_venta', 0)
    if dias_inactivo > 30:
        # Penalización máxima del 80% para productos con > 90 días sin vender
        factor_penalizacion = max(0.20, 1.0 - (dias_inactivo - 30) / 90)
    else:
        factor_penalizacion = 1.0

    if row["stock_actual"] < punto_reorden:
        cantidad = ((punto_reorden * 1.2) - row["stock_actual"]) * factor_penalizacion
        return max(0, int(round(cantidad)))
    return 0


# --- ENDPOINT PRINCIPAL ---

@app.route('/predecir_compra', methods=['POST'])
def predecir_compra():
    try:
        data = request.get_json()

        # --- VALIDACIÓN DE CAMPOS ---
        campos_requeridos = ['ventas', 'stock', 'proveedores']
        for campo in campos_requeridos:
            if campo not in data or not data[campo]:
                return jsonify({
                    "status": "error",
                    "message": f"Falta el campo requerido o está vacío: {campo}"
                }), 400

        cols_ventas = ['codigo_producto', 'fecha', 'cantidad_vendida']
        if not all(col in data['ventas'][0] for col in cols_ventas):
            return jsonify({"status": "error", "message": "Estructura de 'ventas' incorrecta"}), 400

        # 1. Convertir JSON a DataFrames
        df_ventas = pd.DataFrame(data['ventas'])
        df_stock = pd.DataFrame(data['stock'])
        df_proveedores = pd.DataFrame(data['proveedores'])

        df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])

        # 2. Procesar proveedores y lead time
        df_proveedores['lead_time_dias'] = df_proveedores['dias_reposicion'].apply(calcular_lead_time_real)
        lead_time_map = df_proveedores.groupby('proveedor_id')['lead_time_dias'].mean().reset_index()

        # 3. Feature Engineering semanal (igual que antes)
        ventas_semanales = df_ventas.groupby(
            ['codigo_producto', pd.Grouper(key='fecha', freq='W')]
        ).agg({'cantidad_vendida': 'sum'}).reset_index().sort_values(['codigo_producto', 'fecha'])

        ventas_semanales['venta_semana_anterior'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(1)
        ventas_semanales['venta_hace_2_semanas'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(2)
        ventas_semanales['tendencia'] = ventas_semanales['cantidad_vendida'] - ventas_semanales['venta_semana_anterior']
        ventas_semanales['mes'] = ventas_semanales['fecha'].dt.month
        ventas_semanales['demanda_proxima_semana'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(-1)

        # 4. Calcular features temporales por producto
        fecha_referencia = df_ventas["fecha"].max()
        features_temporales = []

        for codigo, grupo in df_ventas.groupby('codigo_producto'):
            ft = calcular_features_temporales(grupo.copy(), fecha_referencia)
            ft['codigo_producto'] = codigo
            features_temporales.append(ft)

        df_features_temporales = pd.DataFrame(features_temporales)

        # 5. Unir features temporales al dataset de entrenamiento
        data_ml = ventas_semanales.dropna().merge(df_stock, on="codigo_producto", how="left").fillna(0)
        data_ml = data_ml.merge(lead_time_map, on="proveedor_id", how="left").fillna(7)
        data_ml = data_ml.merge(df_features_temporales, on="codigo_producto", how="left").fillna(0)

        # 6. Entrenamiento con features extendidas
        features = [
            # Features originales
            "cantidad_vendida", "venta_semana_anterior", "venta_hace_2_semanas",
            "tendencia", "mes", "stock_minimo", "lead_time_dias",
            # Nuevas features temporales
            "dias_desde_ultima_venta",
            "semanas_sin_venta_consecutivas",
            "frecuencia_venta_pct",
            "velocidad_venta_activa",
            "cv_ventas",
            "tendencia_reciente",
        ]

        X = data_ml[features]
        y = data_ml["demanda_proxima_semana"]

        modelo = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        modelo.fit(X, y)
        mae = mean_absolute_error(y, modelo.predict(X))

        # Importancia de features para debugging/transparencia
        importancias = dict(zip(features, modelo.feature_importances_.round(4)))

        # 7. Predicción para la última semana
        ultima_foto = ventas_semanales.groupby('codigo_producto').last().reset_index()
        ultima_foto = ultima_foto.merge(df_stock, on="codigo_producto", how="left").fillna(0)
        ultima_foto = ultima_foto.merge(lead_time_map, on="proveedor_id", how="left").fillna(7)
        ultima_foto = ultima_foto.merge(df_features_temporales, on="codigo_producto", how="left").fillna(0)

        X_predict = ultima_foto[features]
        ultima_foto["demanda_predicha_7dias"] = modelo.predict(X_predict)
        ultima_foto["cantidad_a_comprar"] = ultima_foto.apply(
            lambda r: sugerir_compra_profesional(r, mae), axis=1
        )

        # 8. Preparar respuesta con contexto de actividad del producto
        def clasificar_actividad(row):
            if row['dias_desde_ultima_venta'] > 90:
                return "inactivo"
            elif row['dias_desde_ultima_venta'] > 30:
                return "poco_activo"
            elif row['frecuencia_venta_pct'] >= 0.75:
                return "alta_rotacion"
            else:
                return "rotacion_normal"

        ultima_foto['estado_rotacion'] = ultima_foto.apply(clasificar_actividad, axis=1)

        resultado = ultima_foto[[
            "codigo_producto",
            "demanda_predicha_7dias",
            "cantidad_a_comprar",
            "dias_desde_ultima_venta",
            "frecuencia_venta_pct",
            "estado_rotacion"
        ]].copy()

        resultado["demanda_predicha_7dias"] = resultado["demanda_predicha_7dias"].round(2)
        resultado["frecuencia_venta_pct"] = (resultado["frecuencia_venta_pct"] * 100).round(1)

        resultado = resultado.rename(columns={"frecuencia_venta_pct": "frecuencia_venta_%"})

        return jsonify({
            "status": "success",
            "mae": round(mae, 2),
            "importancia_features": importancias,
            "recomendaciones": resultado.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)