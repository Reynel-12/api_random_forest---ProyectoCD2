"""
Script de prueba para verificar que las gráficas se generan correctamente.
Simula una solicitud al endpoint /predecir_compra_con_graficas y guarda
las imágenes localmente para visualización.
"""
import json
import base64
import os
import requests
from datetime import datetime, timedelta
import random

# --- Generar datos de prueba realistas ---
random.seed(42)

productos = [f"PROD-{str(i).zfill(3)}" for i in range(1, 21)]
proveedores = [{"proveedor_id": 1, "dias_reposicion": "lunes,jueves"},
               {"proveedor_id": 2, "dias_reposicion": "martes,viernes"},
               {"proveedor_id": 3, "dias_reposicion": "miercoles"}]

# Generar ventas para los últimos 3 meses
ventas = []
fecha_inicio = datetime(2026, 1, 1)
for prod in productos:
    # Cada producto tiene un patrón de ventas diferente
    base = random.randint(2, 25)
    for dia in range(90):
        fecha = fecha_inicio + timedelta(days=dia)
        # No vender todos los días
        if random.random() < 0.6:
            cantidad = max(0, int(base + random.gauss(0, base * 0.3)))
            if cantidad > 0:
                ventas.append({
                    "codigo_producto": prod,
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "cantidad_vendida": cantidad
                })

# Stock actual (algunos productos con poco stock)
stock = []
for prod in productos:
    stock.append({
        "codigo_producto": prod,
        "stock_actual": random.randint(5, 80),
        "stock_minimo": random.randint(3, 15),
        "proveedor_id": random.choice([1, 2, 3])
    })

payload = {
    "ventas": ventas,
    "stock": stock,
    "proveedores": proveedores,
    "fecha_referencia": "2026-03-17"
}

print(f"📦 Datos de prueba:")
print(f"   Productos: {len(productos)}")
print(f"   Registros de ventas: {len(ventas)}")
print(f"   Registros de stock: {len(stock)}")
print()

# --- Enviar al endpoint ---
url = "http://localhost:5000/predecir_compra_con_graficas"
print(f"🚀 Enviando request a {url}...")

try:
    response = requests.post(url, json=payload, timeout=60)
    data = response.json()

    if data["status"] == "success":
        print(f"✅ Respuesta exitosa!")
        print(f"   MAE: {data['mae']}")
        print(f"   R²:  {data['r2']}")
        print(f"   Recomendaciones: {len(data['recomendaciones'])} productos")
        print()

        # Guardar gráficas como imágenes
        output_dir = os.path.join(os.path.dirname(__file__), "graficas_output")
        os.makedirs(output_dir, exist_ok=True)

        graficas = data.get("graficas", {})
        print(f"📊 Gráficas generadas: {len(graficas)}")
        for nombre, info in graficas.items():
            img_data = base64.b64decode(info["imagen_base64"])
            filepath = os.path.join(output_dir, f"{nombre}.png")
            with open(filepath, "wb") as f:
                f.write(img_data)
            size_kb = len(img_data) / 1024
            print(f"   ✓ {info['titulo']} → {filepath} ({size_kb:.0f} KB)")

        print(f"\n🎉 Todas las gráficas guardadas en: {output_dir}")
    else:
        print(f"❌ Error: {data['message']}")

except requests.exceptions.ConnectionError:
    print("❌ No se pudo conectar al servidor. Asegúrate de que api.py esté corriendo:")
    print("   python api.py")
except Exception as e:
    print(f"❌ Error: {e}")
