# Dashboard de criptomonedas con Streamlit

Proyecto de analitica de criptomercados y mineria de datos no supervisada implementado 100% con Streamlit y arquitectura dividida por pantallas.

## Implementacion incluida

- Dashboard de segmentacion con K-Means (no supervisado)
- Pantallas separadas: Dataset, Mineria y Exploracion
- Dataset por defecto desde `C:/Users/perez/Downloads/cryptocurrencies.csv`
- Fallback automatico a dataset demo con 730 registros y 11+ variables
- Importacion de CSV en web (equivalente a `read_csv`)
- Validaciones de dataset: dimensiones, columnas, tipos, nulos y duplicados
- Estadistica descriptiva por variable numerica
- Deteccion de anomalias por distancia estandarizada
- Minimo 3 graficas exploratorias:
  - Serie temporal
  - Histograma
  - Dispersion por cluster
- Plantilla de informe/entrega (puntos 2-8) en ENTREGA_REPORTE.md

## Requisitos

- Python 3.10+
- `pip`

## Comandos

1. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Ejecutar el dashboard:

   ```bash
   python -m streamlit run streamlit_app.py
   ```

## Dataset por defecto

- La app intenta cargar automaticamente `C:/Users/perez/Downloads/cryptocurrencies.csv` al iniciar.
- Si el archivo no existe o no se puede procesar, usa un dataset demo y muestra aviso en la pantalla Dataset.
- Puedes reemplazar el dataset activo desde la pantalla Dataset con cualquier CSV.

## Estructura principal

- `streamlit_app.py`: entrypoint y navegacion por pantallas
- `app/state.py`: estado de sesion y dataset inicial
- `app/data.py`: construccion de dataset demo, parseo CSV y enriquecimiento
- `app/analytics.py`: estadistica descriptiva, calidad de datos y K-Means
- `app/screens/dataset_screen.py`: fuente, importacion y calidad del dataset
- `app/screens/mining_screen.py`: configuracion y resultados de mineria no supervisada
- `app/screens/exploration_screen.py`: graficas exploratorias y estadistica descriptiva
