# API Random Forest - Proyecto CD2

Este proyecto implementa una API REST utilizando Flask para predecir las necesidades de compra de productos basadas en un modelo de regresión Random Forest.

## Descripción
La aplicación toma datos de ventas, stock y proveedores, realiza un proceso de ingeniería de características, entrena un modelo de Random Forest "on the fly" y devuelve una recomendación de cantidad a comprar por producto.

Incluye funciones para calcular lead time real basado en días de reposición, estimar demanda crítica y sugerir compras con un margen de seguridad.

## Requisitos

- Python 3.8 o superior

### Librerías

Las dependencias están listadas en `requirements.txt`. Se pueden instalar con:

```bash
pip install -r requirements.txt
```

## Uso

1. Colocar el proyecto en una carpeta de trabajo.
2. Instalar dependencias.
3. Ejecutar la API:

```bash
python api.py
```

4. Hacer una solicitud POST al endpoint `/predecir_compra` con un JSON que contenga:
   - `ventas`: lista de ventas con `codigo_producto`, `fecha` y `cantidad_vendida`.
   - `stock`: lista con `codigo_producto`, `stock_actual` y `stock_minimo`.
   - `proveedores`: lista con `proveedor_id` y `dias_reposicion`.

Ejemplo de cuerpo:
```json
{
  "ventas": [...],
  "stock": [...],
  "proveedores": [...]
}
```

La respuesta incluirá el MAE del modelo y las recomendaciones de compra.

## Estructura de archivos

- `api.py`: archivo principal con la lógica de la API y el modelo.

## Notas

- La API corre en `localhost:5000` en modo debug.
- Asegúrate de que las fechas en `ventas` estén en formato ISO (`YYYY-MM-DD`).

## Contacto

Este proyecto fue desarrollado como parte del Proyecto CD2. Para más información, consulte al equipo de desarrolladores.