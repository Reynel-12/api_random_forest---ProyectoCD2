from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidores
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import io
import base64
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN DE ESTILO PARA GRÁFICAS ---
COLORS = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#8B5CF6',    # Purple
    'success': '#10B981',      # Emerald
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'info': '#06B6D4',         # Cyan
    'dark_bg': '#0F172A',      # Slate dark
    'card_bg': '#1E293B',      # Slate
    'text': '#F8FAFC',         # Light text
    'text_muted': '#94A3B8',   # Muted text
    'grid': '#334155',         # Grid lines
    'accent_gradient': ['#6366F1', '#8B5CF6', '#A78BFA'],  # Indigo gradient
}

def configurar_estilo_grafica():
    """Configura el estilo premium para todas las gráficas."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': COLORS['dark_bg'],
        'axes.facecolor': COLORS['card_bg'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['text'],
        'axes.grid': True,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.3,
        'text.color': COLORS['text'],
        'xtick.color': COLORS['text_muted'],
        'ytick.color': COLORS['text_muted'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
    })

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
    Penalización por inactividad progresiva:
      - 0–5 días sin venta:   sin penalización (compra normal)
      - 5–15 días sin venta:  penalización leve   (hasta -50%)
      - 15–30 días sin venta: penalización fuerte  (hasta -80%)
      - +30 días sin venta:   penalización máxima  (compra 0)
    """
    factor_espera = row["lead_time_dias"] / 7
    demanda_critica = row["demanda_predicha_7dias"] * factor_espera
    seguridad = (demanda_critica * 0.20) + (mae * factor_espera)
    punto_reorden = demanda_critica + seguridad

    dias_inactivo = row.get('dias_desde_ultima_venta', 0)

    # Penalización más agresiva por inactividad:
    #   0–5  días:   sin penalización
    #   6–10 días:   penalización moderada
    #   11–20 días:  penalización fuerte
    #   21–30 días:  penalización casi total
    #   >30 días:    sin compra
    if dias_inactivo <= 5:
        factor_penalizacion = 1.0
    elif dias_inactivo <= 10:
        factor_penalizacion = 0.7
    elif dias_inactivo <= 20:
        factor_penalizacion = 0.4
    elif dias_inactivo <= 30:
        factor_penalizacion = 0.1
    else:
        factor_penalizacion = 0.0

    cantidad_sin_penalizar = 0
    cantidad_final = 0

    if row["stock_actual"] < punto_reorden:
        cantidad_sin_penalizar = ((punto_reorden * 1.2) - row["stock_actual"])
        cantidad_final = max(0, int(round(cantidad_sin_penalizar * factor_penalizacion)))

    # --- BLOQUE DEBUG ---
    print(f"""
    [{row['codigo_producto']}]
    dias_inactivo          = {dias_inactivo}
    factor_penalizacion    = {factor_penalizacion:.2f}
    demanda_predicha_7dias = {row['demanda_predicha_7dias']:.2f}
    punto_reorden          = {punto_reorden:.2f}
    stock_actual           = {row['stock_actual']}
    cantidad_sin_penalizar = {cantidad_sin_penalizar:.2f}
    cantidad_final         = {cantidad_final}
    """)
    # --- FIN DEBUG ---

    return cantidad_final


# --- LÓGICA PRINCIPAL DE PREDICCIÓN (reutilizable) ---

def ejecutar_prediccion(data):
    """
    Ejecuta toda la lógica de predicción y devuelve los datos necesarios
    tanto para la respuesta JSON como para las gráficas.
    Retorna un diccionario con todos los objetos intermedios.
    """
    # --- VALIDACIÓN DE CAMPOS ---
    campos_requeridos = ['ventas', 'stock', 'proveedores']
    for campo in campos_requeridos:
        if campo not in data or not data[campo]:
            raise ValueError(f"Falta el campo requerido o está vacío: {campo}")

    cols_ventas = ['codigo_producto', 'fecha', 'cantidad_vendida']
    if not all(col in data['ventas'][0] for col in cols_ventas):
        raise ValueError("Estructura de 'ventas' incorrecta")

    # 1. Convertir JSON a DataFrames
    df_ventas = pd.DataFrame(data['ventas'])
    df_stock = pd.DataFrame(data['stock'])
    df_proveedores = pd.DataFrame(data['proveedores'])

    df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])

    # 2. Procesar proveedores y lead time
    df_proveedores['lead_time_dias'] = df_proveedores['dias_reposicion'].apply(calcular_lead_time_real)
    lead_time_map = df_proveedores.groupby('proveedor_id')['lead_time_dias'].mean().reset_index()

    # 3. Feature Engineering semanal
    ventas_semanales = df_ventas.groupby(
        ['codigo_producto', pd.Grouper(key='fecha', freq='W')]
    ).agg({'cantidad_vendida': 'sum'}).reset_index().sort_values(['codigo_producto', 'fecha'])

    ventas_semanales['venta_semana_anterior'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(1)
    ventas_semanales['venta_hace_2_semanas'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(2)
    ventas_semanales['tendencia'] = ventas_semanales['cantidad_vendida'] - ventas_semanales['venta_semana_anterior']
    ventas_semanales['mes'] = ventas_semanales['fecha'].dt.month
    ventas_semanales['demanda_proxima_semana'] = ventas_semanales.groupby('codigo_producto')['cantidad_vendida'].shift(-1)

    # 4. Calcular features temporales por producto
    fecha_referencia_raw = data.get("fecha_referencia")
    if fecha_referencia_raw:
        fecha_referencia = pd.to_datetime(fecha_referencia_raw)
    else:
        fecha_referencia = pd.Timestamp.today().normalize()
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
        "cantidad_vendida", "venta_semana_anterior", "venta_hace_2_semanas",
        "tendencia", "mes", "stock_minimo", "lead_time_dias",
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
    y_pred_train = modelo.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    mse = mean_squared_error(y, y_pred_train)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred_train)

    importancias = dict(zip(features, modelo.feature_importances_.round(4)))

    # 7. Predicción para la última semana
    ultima_foto = ventas_semanales.groupby('codigo_producto').last().reset_index()

    for df in [df_ventas, df_stock, df_features_temporales, ultima_foto]:
        df['codigo_producto'] = df['codigo_producto'].astype(str).str.strip().str.upper()

    ultima_foto = ultima_foto.merge(df_stock, on="codigo_producto", how="left").fillna(0)
    ultima_foto = ultima_foto.merge(lead_time_map, on="proveedor_id", how="left").fillna(7)
    ultima_foto = ultima_foto.merge(df_features_temporales, on="codigo_producto", how="left").fillna(0)

    X_predict = ultima_foto[features]
    ultima_foto["demanda_predicha_7dias"] = modelo.predict(X_predict)
    ultima_foto["cantidad_a_comprar"] = ultima_foto.apply(
        lambda r: sugerir_compra_profesional(r, mae), axis=1
    )

    # 8. Clasificar actividad
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

    return {
        'modelo': modelo,
        'features': features,
        'importancias': importancias,
        'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
        'y_real': y, 'y_pred_train': y_pred_train,
        'data_ml': data_ml,
        'ultima_foto': ultima_foto,
        'resultado': resultado,
        'ventas_semanales': ventas_semanales,
        'df_ventas': df_ventas,
        'df_stock': df_stock,
    }


# --- GENERACIÓN DE GRÁFICAS ---

def fig_a_base64(fig):
    """Convierte una figura matplotlib a string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def grafica_demanda_real_vs_predicha(ctx):
    """
    Gráfica 1: Demanda Real vs Predicha (Scatter + línea ideal).
    Muestra qué tan bien el modelo predice la demanda semanal.
    """
    configurar_estilo_grafica()
    y_real = ctx['y_real'].values
    y_pred = ctx['y_pred_train']

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter con gradiente de color basado en error
    errores = np.abs(y_real - y_pred)
    scatter = ax.scatter(y_real, y_pred, c=errores, cmap='RdYlGn_r',
                         alpha=0.7, s=60, edgecolors=COLORS['grid'], linewidth=0.5,
                         zorder=3)

    # Línea de predicción perfecta
    max_val = max(y_real.max(), y_pred.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], '--', color=COLORS['success'],
            linewidth=2, alpha=0.8, label='Predicción Perfecta', zorder=2)

    # Banda de tolerancia (±MAE)
    mae = ctx['mae']
    ax.fill_between([0, max_val], [0 - mae, max_val - mae], [0 + mae, max_val + mae],
                    alpha=0.1, color=COLORS['primary'], label=f'Banda ±MAE ({mae:.1f})')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Error Absoluto', color=COLORS['text_muted'])
    cbar.ax.yaxis.set_tick_params(color=COLORS['text_muted'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS['text_muted'])

    ax.set_xlabel('Demanda Real (unidades/semana)', fontweight='bold')
    ax.set_ylabel('Demanda Predicha (unidades/semana)', fontweight='bold')
    ax.set_title('Demanda Real vs. Predicha por el Modelo',
                 fontsize=16, fontweight='bold', pad=15)

    # Métricas en caja
    metricas_text = (f"R² = {ctx['r2']:.4f}\n"
                     f"MAE = {ctx['mae']:.2f}\n"
                     f"RMSE = {ctx['rmse']:.2f}")
    props = dict(boxstyle='round,pad=0.5', facecolor=COLORS['primary'], alpha=0.3, edgecolor=COLORS['primary'])
    ax.text(0.05, 0.95, metricas_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    ax.legend(loc='lower right', framealpha=0.3)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)

    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_stock_vs_reorden(ctx):
    """
    Gráfica 2: Stock Actual vs Punto de Reorden por producto.
    Visualiza qué productos necesitan reposición urgente.
    """
    configurar_estilo_grafica()
    df = ctx['ultima_foto'].copy()

    # Calcular punto de reorden
    mae = ctx['mae']
    df['punto_reorden'] = df.apply(lambda r: (
        r['demanda_predicha_7dias'] * (r['lead_time_dias'] / 7) * 1.2 +
        mae * (r['lead_time_dias'] / 7)
    ), axis=1)

    # Ordenar por urgencia (ratio stock/punto_reorden)
    df['ratio_urgencia'] = df['stock_actual'] / df['punto_reorden'].replace(0, 1)
    df = df.sort_values('ratio_urgencia')

    # Limitar a top 20 para legibilidad
    df_top = df.head(20)

    fig, ax = plt.subplots(figsize=(12, max(6, len(df_top) * 0.4)))

    y_pos = range(len(df_top))
    bar_height = 0.35

    # Barras de stock actual
    bars1 = ax.barh([y - bar_height/2 for y in y_pos], df_top['stock_actual'],
                     bar_height, label='Stock Actual', color=COLORS['info'], alpha=0.85,
                     edgecolor='none', zorder=3)

    # Barras de punto de reorden
    bars2 = ax.barh([y + bar_height/2 for y in y_pos], df_top['punto_reorden'],
                     bar_height, label='Punto de Reorden', color=COLORS['warning'], alpha=0.85,
                     edgecolor='none', zorder=3)

    # Marcar productos críticos
    for i, (_, row) in enumerate(df_top.iterrows()):
        if row['stock_actual'] < row['punto_reorden']:
            ax.annotate('⚠️', xy=(max(row['stock_actual'], row['punto_reorden']) + 0.5, i),
                       fontsize=14, va='center')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_top['codigo_producto'], fontsize=9)
    ax.set_xlabel('Unidades', fontweight='bold')
    ax.set_title('Stock Actual vs. Punto de Reorden',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.3)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_importancia_features(ctx):
    """
    Gráfica 3: Importancia de Features del modelo Random Forest.
    Transparencia sobre qué factores influyen más en la predicción.
    """
    configurar_estilo_grafica()
    importancias = ctx['importancias']

    # Nombres legibles para las features
    nombres_legibles = {
        'cantidad_vendida': 'Ventas Actuales',
        'venta_semana_anterior': 'Ventas Sem. Anterior',
        'venta_hace_2_semanas': 'Ventas Hace 2 Sem.',
        'tendencia': 'Tendencia',
        'mes': 'Mes del Año',
        'stock_minimo': 'Stock Mínimo',
        'lead_time_dias': 'Tiempo de Entrega',
        'dias_desde_ultima_venta': 'Días Sin Venta',
        'semanas_sin_venta_consecutivas': 'Sem. Inactivas Consec.',
        'frecuencia_venta_pct': 'Frecuencia de Venta',
        'velocidad_venta_activa': 'Velocidad Venta Activa',
        'cv_ventas': 'Variabilidad Demanda',
        'tendencia_reciente': 'Tendencia Reciente',
    }

    features_sorted = sorted(importancias.items(), key=lambda x: x[1], reverse=True)
    nombres = [nombres_legibles.get(f, f) for f, _ in features_sorted]
    valores = [v for _, v in features_sorted]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Gradiente de colores
    n = len(valores)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))[::-1]  # Verde → Rojo
    # Revertir para que el más importante sea verde
    colors = plt.cm.viridis(np.linspace(0.3, 0.95, n))[::-1]

    bars = ax.barh(range(n), valores, color=colors, edgecolor='none', alpha=0.9, zorder=3)

    # Agregar valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars, valores)):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=10, color=COLORS['text'])

    ax.set_yticks(range(n))
    ax.set_yticklabels(nombres, fontsize=10)
    ax.set_xlabel('Importancia Relativa', fontweight='bold')
    ax.set_title('¿Qué Factores Influyen Más en la Predicción?',
                 fontsize=16, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_distribucion_errores(ctx):
    """
    Gráfica 4: Distribución de errores del modelo.
    Muestra la confiabilidad de las predicciones.
    """
    configurar_estilo_grafica()
    y_real = ctx['y_real'].values
    y_pred = ctx['y_pred_train']
    errores = y_real - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel izquierdo: Histograma de errores
    ax1 = axes[0]
    ax1.hist(errores, bins=30, color=COLORS['primary'], alpha=0.7,
             edgecolor=COLORS['dark_bg'], linewidth=0.5, zorder=3)
    ax1.axvline(x=0, color=COLORS['success'], linestyle='--', linewidth=2,
                label='Error = 0', zorder=4)
    ax1.axvline(x=np.mean(errores), color=COLORS['warning'], linestyle='-', linewidth=2,
                label=f'Media = {np.mean(errores):.2f}', zorder=4)

    ax1.set_xlabel('Error (Real - Predicho)', fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontweight='bold')
    ax1.set_title('Distribución de Errores', fontsize=13, fontweight='bold')
    ax1.legend(framealpha=0.3)

    # Panel derecho: Error absoluto por percentil
    ax2 = axes[1]
    errores_abs = np.abs(errores)
    percentiles = [50, 75, 90, 95, 99]
    valores_pct = [np.percentile(errores_abs, p) for p in percentiles]

    bars = ax2.bar(range(len(percentiles)), valores_pct,
                   color=[COLORS['success'], COLORS['info'], COLORS['primary'],
                          COLORS['warning'], COLORS['danger']],
                   alpha=0.85, edgecolor='none', zorder=3)

    for bar, val in zip(bars, valores_pct):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}', ha='center', fontsize=11, fontweight='bold',
                color=COLORS['text'])

    ax2.set_xticks(range(len(percentiles)))
    ax2.set_xticklabels([f'P{p}' for p in percentiles])
    ax2.set_xlabel('Percentil', fontweight='bold')
    ax2.set_ylabel('Error Absoluto', fontweight='bold')
    ax2.set_title('Error por Percentil', fontsize=13, fontweight='bold')

    fig.suptitle('Análisis de Confiabilidad del Modelo',
                 fontsize=16, fontweight='bold', y=1.02, color=COLORS['text'])
    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_estado_rotacion(ctx):
    """
    Gráfica 5: Resumen de estado de rotación de productos.
    Vista ejecutiva de la salud del inventario.
    """
    configurar_estilo_grafica()
    df = ctx['ultima_foto']

    # Contar por estado
    conteo = df['estado_rotacion'].value_counts()

    colores_estado = {
        'alta_rotacion': COLORS['success'],
        'rotacion_normal': COLORS['info'],
        'poco_activo': COLORS['warning'],
        'inactivo': COLORS['danger'],
    }
    etiquetas_estado = {
        'alta_rotacion': 'Alta Rotación',
        'rotacion_normal': 'Rotación Normal',
        'poco_activo': 'Poco Activo',
        'inactivo': 'Inactivo',
    }

    labels = [etiquetas_estado.get(e, e) for e in conteo.index]
    sizes = conteo.values
    colors_pie = [colores_estado.get(e, COLORS['text_muted']) for e in conteo.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1.3]})

    # Panel izquierdo: Donut chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors_pie,
             autopct='%1.1f%%', startangle=90, pctdistance=0.78,
             wedgeprops=dict(width=0.45, edgecolor=COLORS['dark_bg'], linewidth=2))

    for text in texts:
        text.set_color(COLORS['text'])
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color(COLORS['text'])
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')

    # Centro del donut
    ax1.text(0, 0, f'{len(df)}\nProductos', ha='center', va='center',
             fontsize=16, fontweight='bold', color=COLORS['text'])
    ax1.set_title('Estado de Rotación', fontsize=13, fontweight='bold', pad=15)

    # Panel derecho: Barras con cantidad a comprar por estado
    ax2 = axes[1]
    estados_unicos = df['estado_rotacion'].unique()
    compra_por_estado = []
    for estado in ['alta_rotacion', 'rotacion_normal', 'poco_activo', 'inactivo']:
        if estado in estados_unicos:
            total = df[df['estado_rotacion'] == estado]['cantidad_a_comprar'].sum()
            compra_por_estado.append((etiquetas_estado.get(estado, estado), total,
                                     colores_estado.get(estado, COLORS['text_muted'])))

    if compra_por_estado:
        labels_bar = [x[0] for x in compra_por_estado]
        vals_bar = [x[1] for x in compra_por_estado]
        colors_bar = [x[2] for x in compra_por_estado]

        bars = ax2.bar(labels_bar, vals_bar, color=colors_bar, alpha=0.85,
                       edgecolor='none', zorder=3, width=0.6)

        for bar, val in zip(bars, vals_bar):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{int(val)}', ha='center', fontsize=11, fontweight='bold',
                        color=COLORS['text'])

    ax2.set_ylabel('Unidades a Comprar', fontweight='bold')
    ax2.set_title('Compra Sugerida por Estado', fontsize=13, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=15)

    fig.suptitle('Clasificación de Productos por Rotación',
                 fontsize=16, fontweight='bold', y=1.02, color=COLORS['text'])
    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_top_productos_comprar(ctx):
    """
    Gráfica 6: Top productos que más necesitan reposición.
    Accionable y directamente útil para decisiones de compra.
    """
    configurar_estilo_grafica()
    df = ctx['ultima_foto'].copy()

    # Filtrar solo productos con compra > 0
    df_compra = df[df['cantidad_a_comprar'] > 0].sort_values('cantidad_a_comprar', ascending=False).head(15)

    if df_compra.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, 'No hay productos que necesiten reposición urgente',
                ha='center', va='center', fontsize=14, color=COLORS['success'],
                transform=ax.transAxes)
        ax.set_title('🛒 Top Productos a Comprar', fontsize=16, fontweight='bold')
        fig.tight_layout()
        return fig_a_base64(fig)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df_compra) * 0.45)))

    y_pos = range(len(df_compra))

    # Gradiente de urgencia
    n = len(df_compra)
    gradient_colors = plt.cm.YlOrRd(np.linspace(0.3, 0.85, n))

    bars = ax.barh(y_pos, df_compra['cantidad_a_comprar'].values,
                   color=gradient_colors, edgecolor='none', alpha=0.9, zorder=3)

    # Agregar info adicional sobre cada barra
    for i, (_, row) in enumerate(df_compra.iterrows()):
        cantidad = int(row['cantidad_a_comprar'])
        demanda = row['demanda_predicha_7dias']
        stock = int(row['stock_actual'])

        ax.text(row['cantidad_a_comprar'] + 0.3, i,
                f'  {cantidad} uds  │  Stock: {stock}  │  Demanda: {demanda:.1f}/sem',
                va='center', fontsize=9, color=COLORS['text_muted'])

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df_compra['codigo_producto'].values, fontsize=10)
    ax.set_xlabel('Cantidad a Comprar (unidades)', fontweight='bold')
    ax.set_title('Top Productos que Necesitan Reposición',
                 fontsize=16, fontweight='bold', pad=15)
    ax.invert_yaxis()

    # Ajustar xlim para que quepa el texto
    ax.set_xlim(0, df_compra['cantidad_a_comprar'].max() * 2.2)

    fig.tight_layout()
    return fig_a_base64(fig)


def grafica_tendencia_ventas_semanal(ctx):
    """
    Gráfica 7: Tendencia de ventas semanales agregadas.
    Muestra la evolución general de las ventas para contexto macro.
    """
    configurar_estilo_grafica()
    vs = ctx['ventas_semanales'].copy()

    # Agregar ventas totales por semana
    ventas_totales = vs.groupby('fecha')['cantidad_vendida'].sum().reset_index()
    ventas_totales = ventas_totales.sort_values('fecha')

    fig, ax = plt.subplots(figsize=(12, 5))

    # Área bajo la curva
    ax.fill_between(ventas_totales['fecha'], ventas_totales['cantidad_vendida'],
                    alpha=0.15, color=COLORS['primary'], zorder=2)

    # Línea principal
    ax.plot(ventas_totales['fecha'], ventas_totales['cantidad_vendida'],
            color=COLORS['primary'], linewidth=2.5, marker='o', markersize=4,
            markerfacecolor=COLORS['secondary'], markeredgecolor='none',
            zorder=3, label='Ventas Semanales')

    # Media móvil
    if len(ventas_totales) >= 4:
        ventas_totales['media_movil'] = ventas_totales['cantidad_vendida'].rolling(4, min_periods=2).mean()
        ax.plot(ventas_totales['fecha'], ventas_totales['media_movil'],
                color=COLORS['warning'], linewidth=2, linestyle='--',
                alpha=0.8, zorder=3, label='Media Móvil (4 sem.)')

    ax.set_xlabel('Fecha', fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontweight='bold')
    ax.set_title('Tendencia de Ventas Semanales (Todos los productos)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig_a_base64(fig)


# --- ENDPOINTS ---

@app.route('/predecir_compra', methods=['POST'])
def predecir_compra():
    """Endpoint original: solo retorna predicciones en JSON."""
    try:
        data = request.get_json()
        ctx = ejecutar_prediccion(data)

        return jsonify({
            "status": "success",
            "mae": round(ctx['mae'], 2),
            "mse": round(ctx['mse'], 2),
            "rmse": round(ctx['rmse'], 2),
            "r2": round(ctx['r2'], 2),
            "importancia_features": ctx['importancias'],
            "recomendaciones": ctx['resultado'].to_dict(orient='records')
        })

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/predecir_compra_con_graficas', methods=['POST'])
def predecir_compra_con_graficas():
    """
    Endpoint extendido: retorna predicciones + gráficas en base64.
    Las gráficas se devuelven como PNGs codificados en base64
    dentro del campo 'graficas' de la respuesta JSON.
    """
    try:
        data = request.get_json()
        ctx = ejecutar_prediccion(data)

        # Generar todas las gráficas
        print("Generando gráficas...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graficas = {
            "demanda_real_vs_predicha": {
                "titulo": "Demanda Real vs. Predicha",
                "descripcion": "Comparación entre la demanda real y la predicción del modelo. Los puntos cercanos a la línea diagonal indican buenas predicciones.",
                "imagen_base64": grafica_demanda_real_vs_predicha(ctx),
                "archivo": f"demanda_real_vs_predicha_{timestamp}.png"
            },
            "stock_vs_reorden": {
                "titulo": "Stock Actual vs. Punto de Reorden",
                "descripcion": "Muestra qué productos están por debajo de su punto de reorden óptimo y necesitan reposición.",
                "imagen_base64": grafica_stock_vs_reorden(ctx),
                "archivo": f"stock_vs_reorden_{timestamp}.png"
            },
            "importancia_features": {
                "titulo": "Factores que Influyen en la Predicción",
                "descripcion": "Importancia relativa de cada variable en el modelo Random Forest. Muestra qué datos son más relevantes para predecir la demanda.",
                "imagen_base64": grafica_importancia_features(ctx),
                "archivo": f"importancia_features_{timestamp}.png"
            },
            "distribucion_errores": {
                "titulo": "Análisis de Confiabilidad",
                "descripcion": "Distribución de errores del modelo y percentiles de error absoluto. Permite evaluar la confiabilidad de las predicciones.",
                "imagen_base64": grafica_distribucion_errores(ctx),
                "archivo": f"distribucion_errores_{timestamp}.png"
            },
            "estado_rotacion": {
                "titulo": "Clasificación por Rotación",
                "descripcion": "Distribución de productos según su nivel de actividad y la recomendación de compra agrupada por estado.",
                "imagen_base64": grafica_estado_rotacion(ctx),
                "archivo": f"estado_rotacion_{timestamp}.png"
            },
            "top_productos_comprar": {
                "titulo": "Top Productos a Comprar",
                "descripcion": "Productos con mayor necesidad de reposición según el modelo, con detalle de stock actual y demanda predicha.",
                "imagen_base64": grafica_top_productos_comprar(ctx),
                "archivo": f"top_productos_comprar_{timestamp}.png"
            },
            "tendencia_ventas": {
                "titulo": "Tendencia de Ventas Semanales",
                "descripcion": "Evolución agregada de las ventas semanales con media móvil. Permite identificar estacionalidad y tendencias generales.",
                "imagen_base64": grafica_tendencia_ventas_semanal(ctx),
                "archivo": f"tendencia_ventas_{timestamp}.png"
            },
        }
        print("Gráficas generadas exitosamente.")

         # Guardar gráficas como imágenes
        output_dir = os.path.join(os.path.dirname(__file__), "graficas_output")
        os.makedirs(output_dir, exist_ok=True)

        print(f"📊 Gráficas generadas: {len(graficas)}")
        for nombre, info in graficas.items():
            img_data = base64.b64decode(info["imagen_base64"])
            filepath = os.path.join(output_dir, f"{nombre}.png")
            with open(filepath, "wb") as f:
                f.write(img_data)
            size_kb = len(img_data) / 1024
            print(f"   ✓ {info['titulo']} → {filepath} ({size_kb:.0f} KB)")

        print(f"\n🎉 Todas las gráficas guardadas en: {output_dir}")

        return jsonify({
            "status": "success",
            "mae": round(ctx['mae'], 2),
            "mse": round(ctx['mse'], 2),
            "rmse": round(ctx['rmse'], 2),
            "r2": round(ctx['r2'], 2),
            "importancia_features": ctx['importancias'],
            "recomendaciones": ctx['resultado'].to_dict(orient='records'),
            "graficas": graficas
        })

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/graficas_disponibles', methods=['GET'])
def graficas_disponibles():
    """Lista las gráficas disponibles con sus descripciones."""
    return jsonify({
        "status": "success",
        "graficas": {
            "demanda_real_vs_predicha": "Scatter plot comparando demanda real vs predicha con métricas R², MAE, RMSE",
            "stock_vs_reorden": "Barras horizontales de stock actual vs punto de reorden por producto",
            "importancia_features": "Ranking de importancia de variables del modelo Random Forest",
            "distribucion_errores": "Histograma de errores y análisis por percentiles",
            "estado_rotacion": "Donut chart de clasificación de productos + compra por estado",
            "top_productos_comprar": "Top 15 productos con mayor necesidad de reposición",
            "tendencia_ventas": "Serie temporal de ventas semanales con media móvil"
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)