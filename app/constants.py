MIN_REQUIRED_ROWS = 500
MIN_REQUIRED_COLUMNS = 5
MAX_CLUSTER_COUNT = 6

INITIAL_CLUSTER_COUNT = 4
INITIAL_ANOMALY_THRESHOLD = 2.2

DEFAULT_LOCAL_CSV_PATH = r'C:\Users\perez\Downloads\cryptocurrencies.csv'

DEFAULT_DATASET_NAME = 'cryptocurrencies.csv'
DEFAULT_SOURCE_NOTE = (
  'Fuente de datos: CSV local de criptomonedas cargado automaticamente '
  f'({DEFAULT_LOCAL_CSV_PATH}).'
)

FALLBACK_DATASET_NAME = 'btc_usd_demo_730.csv'
FALLBACK_SOURCE_NOTE = (
  'Fuente de datos: serie diaria BTC/USD de demostracion (formato OHLCV), '
  'preparada para analisis de mineria no supervisada.'
)

FEATURE_LABELS = {
  'rank': 'Ranking',
  'name': 'Nombre',
  'symbol': 'Ticker',
  'coin_id': 'Coin ID',
  'price_usd': 'Precio (USD)',
  'market_cap_usd': 'Market cap (USD)',
  'volume_24h_usd': 'Volumen 24h (USD)',
  'price_change_24h_pct': 'Cambio 24h (%)',
  'price_change_7d_pct': 'Cambio 7d (%)',
  'price_change_30d_pct': 'Cambio 30d (%)',
  'as_of': 'Fecha de corte',
  'cryptocurrency': 'Criptomoneda',
  'year': 'Fecha de referencia',
  'length_of_text': 'Longitud de whitepaper',
  'length_of_dictionary': 'Riqueza lexica',
  'compressed_size': 'Tamano comprimido',
  'current_usd_value': 'Valor actual (USD)',
  'current_24h_volume': 'Volumen 24h (USD)',
  'open': 'Precio de apertura',
  'high': 'Precio maximo',
  'low': 'Precio minimo',
  'close': 'Precio de cierre',
  'volume': 'Volumen',
  'market_cap': 'Capitalizacion de mercado',
  'trades': 'Numero de transacciones',
  'return_pct': 'Retorno diario (%)',
  'volatility_pct': 'Volatilidad diaria (%)',
  'range_pct': 'Rango intradia (%)',
}

PREFERRED_FEATURE_ORDER = [
  'close',
  'volume',
  'market_cap_usd',
  'price_change_24h_pct',
  'price_change_7d_pct',
  'price_change_30d_pct',
  'volatility_pct',
  'return_pct',
  'range_pct',
  'open',
  'high',
  'low',
  'trades',
  'market_cap',
]
