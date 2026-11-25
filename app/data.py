from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from app.constants import (
  DEFAULT_DATASET_NAME,
  DEFAULT_LOCAL_CSV_PATH,
  FALLBACK_DATASET_NAME,
  FALLBACK_SOURCE_NOTE,
)


def round_number(value: float, decimals: int = 2) -> float:
  scale = 10**decimals
  return round(value * scale) / scale


def build_bitcoin_dataset(total_rows: int = 730) -> pd.DataFrame:
  random = np.random.default_rng(20261122)
  first_date = pd.Timestamp('2024-01-01')
  records: list[dict] = []
  previous_close = 41950.0

  for row_index in range(total_rows):
    trend = 0.00085
    seasonal_component = np.sin(row_index / 35) * 0.0108
    random_shock = (random.random() - 0.5) * 0.043

    open_value = max(3500, previous_close * (1 + (random.random() - 0.5) * 0.014))
    close_value = max(
      3500,
      previous_close * (1 + trend + seasonal_component + random_shock),
    )

    upper_base = max(open_value, close_value)
    lower_base = min(open_value, close_value)
    high_value = upper_base * (1 + random.random() * 0.022)
    low_value = lower_base * (1 - random.random() * 0.022)

    return_pct = ((close_value - open_value) / open_value) * 100
    volatility_pct = ((high_value - low_value) / open_value) * 100
    range_pct = ((high_value - low_value) / max(close_value, 1)) * 100

    volume_value = 12000 + random.random() * 42000 + abs(return_pct) * 300 + volatility_pct * 460
    market_cap_value = close_value * 19_500_000
    trades_value = int(round(130000 + volume_value * 1.55 + random.random() * 14000))

    current_date = first_date + pd.Timedelta(days=row_index)

    records.append(
      {
        'row_id': row_index + 1,
        'date': current_date.strftime('%Y-%m-%d'),
        'open': round_number(open_value, 2),
        'high': round_number(high_value, 2),
        'low': round_number(low_value, 2),
        'close': round_number(close_value, 2),
        'volume': int(round(volume_value)),
        'market_cap': int(round(market_cap_value)),
        'trades': trades_value,
        'return_pct': round_number(return_pct, 4),
        'volatility_pct': round_number(volatility_pct, 4),
        'range_pct': round_number(range_pct, 4),
      }
    )

    previous_close = close_value

  return pd.DataFrame.from_records(records)


def _normalize_header(header: str) -> str:
  normalized = ''.join(character.lower() if character.isalnum() else '_' for character in header.strip())
  while '__' in normalized:
    normalized = normalized.replace('__', '_')
  return normalized.strip('_')


def _read_csv_content(file_content: bytes) -> pd.DataFrame:
  utf8_buffer = BytesIO(file_content)
  latin_buffer = BytesIO(file_content)

  try:
    return pd.read_csv(utf8_buffer, sep=None, engine='python', encoding='utf-8')
  except UnicodeDecodeError:
    return pd.read_csv(latin_buffer, sep=None, engine='python', encoding='latin1')


def parse_uploaded_csv(file_content: bytes) -> pd.DataFrame:
  parsed_df = _read_csv_content(file_content)

  if parsed_df.empty:
    raise ValueError('El CSV no contiene filas para procesar.')

  parsed_df.columns = [_normalize_header(str(column)) for column in parsed_df.columns]
  if any(not column for column in parsed_df.columns):
    raise ValueError('Se detectaron encabezados vacios en el CSV.')

  parsed_df = parsed_df.dropna(how='all').reset_index(drop=True)
  return parsed_df


def _coerce_object_columns(dataset_df: pd.DataFrame) -> pd.DataFrame:
  adjusted_df = dataset_df.copy()
  for column in adjusted_df.columns:
    if adjusted_df[column].dtype != 'object':
      continue

    as_text = adjusted_df[column].astype(str).str.replace(',', '', regex=False)
    numeric_series = pd.to_numeric(as_text, errors='coerce')
    non_empty = adjusted_df[column].notna().sum()
    if non_empty == 0:
      continue

    numeric_ratio = numeric_series.notna().sum() / non_empty
    if numeric_ratio >= 0.85:
      adjusted_df[column] = numeric_series

  return adjusted_df


def _find_first_numeric_column(dataset_df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
  for candidate in candidates:
    if candidate not in dataset_df.columns:
      continue

    numeric_series = pd.to_numeric(dataset_df[candidate], errors='coerce')
    if numeric_series.notna().any():
      return numeric_series

  return None


def _pick_fallback_series(*series_candidates: pd.Series | None) -> pd.Series | None:
  for candidate in series_candidates:
    if candidate is None:
      continue
    if candidate.notna().any():
      return candidate

  return None


def _resolve_numeric_series(
  dataset_df: pd.DataFrame,
  target_column: str,
  candidates: list[str],
) -> pd.Series | None:
  if target_column in dataset_df.columns:
    own_series = pd.to_numeric(dataset_df[target_column], errors='coerce')
    if own_series.notna().any():
      return own_series

  return _find_first_numeric_column(dataset_df, candidates)


def enrich_dataset(dataset_df: pd.DataFrame) -> pd.DataFrame:
  if dataset_df.empty:
    raise ValueError('El dataset no contiene filas.')

  adjusted_df = _coerce_object_columns(dataset_df)
  adjusted_df = adjusted_df.copy()
  if 'row_id' in adjusted_df.columns:
    adjusted_df = adjusted_df.drop(columns=['row_id'])
  adjusted_df.insert(0, 'row_id', np.arange(1, len(adjusted_df) + 1))

  date_candidates = ['date', 'timestamp', 'fecha', 'year', 'datetime', 'as_of', 'ath_date', 'atl_date']
  date_column = next((candidate for candidate in date_candidates if candidate in adjusted_df.columns), None)

  if date_column:
    adjusted_df['date'] = adjusted_df[date_column].astype(str)
  else:
    adjusted_df['date'] = [f'row-{index}' for index in range(1, len(adjusted_df) + 1)]

  open_series = _find_first_numeric_column(
    adjusted_df,
    ['open', 'apertura', 'btc_open', 'current_usd_value', 'usd_value', 'price_usd'],
  )
  close_series = _find_first_numeric_column(
    adjusted_df,
    [
      'close',
      'cierre',
      'btc_close',
      'btc_market_price',
      'market_price',
      'price',
      'current_usd_value',
      'usd_value',
      'price_usd',
    ],
  )
  high_series = _find_first_numeric_column(adjusted_df, ['high', 'maximo', 'btc_high', 'price_usd'])
  low_series = _find_first_numeric_column(adjusted_df, ['low', 'minimo', 'btc_low', 'price_usd'])

  fallback_series = _pick_fallback_series(close_series, open_series, high_series, low_series)

  if open_series is None and close_series is not None:
    open_series = close_series.shift(1)
    open_series = open_series.fillna(close_series)

  if open_series is None:
    open_series = fallback_series
  if close_series is None and open_series is not None:
    close_series = open_series
  if close_series is None:
    close_series = fallback_series
  if high_series is None and open_series is not None and close_series is not None:
    high_series = pd.concat([open_series, close_series], axis=1).max(axis=1)
  if low_series is None and open_series is not None and close_series is not None:
    low_series = pd.concat([open_series, close_series], axis=1).min(axis=1)

  if open_series is None or close_series is None or high_series is None or low_series is None:
    raise ValueError(
      'No fue posible derivar columnas de precio (open/high/low/close). '
      'Incluye al menos una columna de precios numerica.'
    )

  adjusted_df['open'] = open_series.round(2)
  adjusted_df['close'] = close_series.round(2)
  adjusted_df['high'] = high_series.round(2)
  adjusted_df['low'] = low_series.round(2)

  safe_open = adjusted_df['open'].replace(0, np.nan)
  safe_close = adjusted_df['close'].replace(0, np.nan)
  price_diff = adjusted_df['close'] - adjusted_df['open']
  price_range = adjusted_df['high'] - adjusted_df['low']

  volume_series = _resolve_numeric_series(
    adjusted_df,
    'volume',
    [
      'volume',
      'btc_trade_volume',
      'trade_volume',
      'btc_estimated_transaction_volume_usd',
      'btc_output_volume',
      'current_24h_volume',
      'volume_24h',
      'volume_24h_usd',
    ],
  )
  if volume_series is not None:
    adjusted_df['volume'] = volume_series.round(2)

  market_cap_series = _resolve_numeric_series(
    adjusted_df,
    'market_cap',
    ['market_cap', 'btc_market_cap', 'market_cap_usd', 'fully_diluted_valuation_usd'],
  )
  if market_cap_series is not None:
    adjusted_df['market_cap'] = market_cap_series.round(2)

  trades_series = _resolve_numeric_series(
    adjusted_df,
    'trades',
    [
      'trades',
      'btc_n_transactions',
      'btc_n_transactions_per_block',
      'btc_n_unique_addresses',
    ],
  )
  if trades_series is not None:
    adjusted_df['trades'] = trades_series.round().astype('Int64')

  if 'return_pct' not in adjusted_df.columns:
    adjusted_df['return_pct'] = np.where(
      safe_open.notna(),
      (price_diff / safe_open) * 100,
      np.where(price_diff.eq(0), 0.0, np.nan),
    )
  else:
    adjusted_df['return_pct'] = pd.to_numeric(adjusted_df['return_pct'], errors='coerce')

  if 'volatility_pct' not in adjusted_df.columns:
    adjusted_df['volatility_pct'] = np.where(
      safe_open.notna(),
      (price_range / safe_open) * 100,
      np.where(price_range.eq(0), 0.0, np.nan),
    )
  else:
    adjusted_df['volatility_pct'] = pd.to_numeric(adjusted_df['volatility_pct'], errors='coerce')

  if 'range_pct' not in adjusted_df.columns:
    adjusted_df['range_pct'] = np.where(
      safe_close.notna(),
      (price_range / safe_close) * 100,
      np.where(price_range.eq(0), 0.0, np.nan),
    )
  else:
    adjusted_df['range_pct'] = pd.to_numeric(adjusted_df['range_pct'], errors='coerce')

  for column in ['return_pct', 'volatility_pct', 'range_pct']:
    adjusted_df[column] = adjusted_df[column].replace([np.inf, -np.inf], np.nan).round(4)

  return adjusted_df


def load_dataset_from_csv_path(csv_path: str | Path) -> pd.DataFrame:
  source_path = Path(csv_path)
  if not source_path.exists():
    raise FileNotFoundError(f'No se encontro el archivo CSV en: {source_path}')

  parsed_df = parse_uploaded_csv(source_path.read_bytes())
  return enrich_dataset(parsed_df)


def load_default_dataset() -> tuple[pd.DataFrame, str, str, str]:
  default_path = Path(DEFAULT_LOCAL_CSV_PATH)

  if default_path.exists():
    try:
      loaded_df = load_dataset_from_csv_path(default_path)
      source_note = (
        'Fuente de datos: CSV local de criptomonedas cargado automaticamente '
        f'({default_path}).'
      )
      return loaded_df, DEFAULT_DATASET_NAME, source_note, ''
    except (ValueError, FileNotFoundError) as error:
      fallback_df = build_bitcoin_dataset()
      return (
        fallback_df,
        FALLBACK_DATASET_NAME,
        FALLBACK_SOURCE_NOTE,
        f'No fue posible usar el CSV por defecto: {error}',
      )

  fallback_df = build_bitcoin_dataset()
  return (
    fallback_df,
    FALLBACK_DATASET_NAME,
    FALLBACK_SOURCE_NOTE,
    f'No se encontro el CSV por defecto en {default_path}. Se cargo dataset demo.',
  )
