from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from app.constants import FEATURE_LABELS, PREFERRED_FEATURE_ORDER


MIN_NUMERIC_COVERAGE_RATIO = 0.2


def format_value(value) -> str:
  if value is None or (isinstance(value, float) and np.isnan(value)):
    return '-'

  try:
    numeric = float(value)
    if abs(numeric) >= 1_000_000:
      return f'{numeric:,.2f}'
    return f'{numeric:,.2f}'
  except (TypeError, ValueError):
    return str(value)


def infer_column_type(dataset_df: pd.DataFrame, column_name: str) -> str:
  column = dataset_df[column_name]
  non_empty = column.dropna()
  if non_empty.empty:
    return 'vacio'

  numeric_ratio = pd.to_numeric(non_empty.astype(str).str.replace(',', '', regex=False), errors='coerce').notna().mean()
  if numeric_ratio >= 0.85:
    return 'numerico'

  datetime_ratio = pd.to_datetime(
    non_empty.astype(str),
    errors='coerce',
    format='mixed',
    dayfirst=True,
  ).notna().mean()
  if datetime_ratio >= 0.85:
    return 'fecha'

  return 'texto'


def get_visible_columns(dataset_df: pd.DataFrame) -> list[str]:
  return [column for column in dataset_df.columns if column != 'row_id']


def get_numeric_columns(dataset_df: pd.DataFrame, visible_columns: list[str]) -> list[str]:
  numeric_columns: list[str] = []
  for column in visible_columns:
    numeric_series = pd.to_numeric(dataset_df[column], errors='coerce')
    valid_ratio = numeric_series.notna().mean()
    if valid_ratio >= MIN_NUMERIC_COVERAGE_RATIO:
      numeric_columns.append(column)
  return numeric_columns


def get_column_types(dataset_df: pd.DataFrame, visible_columns: list[str]) -> pd.DataFrame:
  return pd.DataFrame(
    {
      'columna': visible_columns,
      'tipo_inferido': [infer_column_type(dataset_df, column) for column in visible_columns],
    }
  )


def get_data_quality(dataset_df: pd.DataFrame, visible_columns: list[str]) -> dict[str, int]:
  null_count = int(dataset_df[visible_columns].isna().sum().sum())
  duplicate_rows = int(dataset_df.duplicated(subset=visible_columns).sum())
  return {'null_count': null_count, 'duplicate_rows': duplicate_rows}


def compute_descriptive_stats(dataset_df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
  stats_rows: list[dict] = []
  for column in numeric_columns[:8]:
    numeric_values = pd.to_numeric(dataset_df[column], errors='coerce').dropna()
    if numeric_values.empty:
      continue

    stats_rows.append(
      {
        'variable': FEATURE_LABELS.get(column, column),
        'n': int(numeric_values.count()),
        'media': float(numeric_values.mean()),
        'mediana': float(numeric_values.median()),
        'desv_std': float(numeric_values.std(ddof=0)),
        'min': float(numeric_values.min()),
        'max': float(numeric_values.max()),
      }
    )

  return pd.DataFrame(stats_rows)


def pick_default_features(numeric_columns: list[str]) -> list[str]:
  selected: list[str] = []

  for preferred in PREFERRED_FEATURE_ORDER:
    if preferred in numeric_columns and preferred not in selected:
      selected.append(preferred)

  for column in numeric_columns:
    if column not in selected:
      selected.append(column)

  return selected[: min(5, len(selected))]


def run_kmeans(
  dataset_df: pd.DataFrame,
  selected_features: list[str],
  cluster_count: int,
  anomaly_threshold: float,
) -> dict:
  if len(selected_features) < 3:
    return {
      'error': 'Selecciona al menos 3 variables numericas para ejecutar K-Means.',
      'rows': dataset_df.copy(),
      'modeled_rows': pd.DataFrame(),
      'cluster_counts': [],
      'anomaly_count': 0,
      'centroids': pd.DataFrame(),
      'silhouette_score': None,
      'davies_bouldin': None,
      'calinski_harabasz': None,
      'inertia': None,
    }

  model_df = dataset_df.copy()
  for feature in selected_features:
    model_df[feature] = pd.to_numeric(model_df[feature], errors='coerce')

  model_df[selected_features] = model_df[selected_features].replace([np.inf, -np.inf], np.nan)
  model_df = model_df.dropna(subset=selected_features).copy()
  if len(model_df) < cluster_count * 4:
    return {
      'error': (
        'No hay suficientes filas validas para el modelo con las variables elegidas. '
        'Revisa nulos o cambia seleccion.'
      ),
      'rows': dataset_df.copy(),
      'modeled_rows': pd.DataFrame(),
      'cluster_counts': [],
      'anomaly_count': 0,
      'centroids': pd.DataFrame(),
      'silhouette_score': None,
      'davies_bouldin': None,
      'calinski_harabasz': None,
      'inertia': None,
    }

  features_matrix = model_df[selected_features].astype(float).to_numpy()
  scaler = StandardScaler()
  scaled = scaler.fit_transform(features_matrix)

  kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
  labels = kmeans.fit_predict(scaled) + 1
  distances = np.linalg.norm(scaled - kmeans.cluster_centers_[labels - 1], axis=1)

  silhouette_value = None
  davies_value = None
  calinski_value = None
  unique_cluster_count = np.unique(labels).size

  if unique_cluster_count > 1 and len(model_df) > unique_cluster_count:
    silhouette_value = float(silhouette_score(scaled, labels))
    davies_value = float(davies_bouldin_score(scaled, labels))
    calinski_value = float(calinski_harabasz_score(scaled, labels))

  model_df['cluster'] = labels
  model_df['anomaly_score'] = np.round(distances, 4)
  model_df['is_anomaly'] = distances >= anomaly_threshold

  result_rows = dataset_df.copy()
  result_rows['cluster'] = pd.NA
  result_rows['anomaly_score'] = np.nan
  result_rows['is_anomaly'] = False
  result_rows.loc[model_df.index, 'cluster'] = model_df['cluster'].astype('Int64')
  result_rows.loc[model_df.index, 'anomaly_score'] = model_df['anomaly_score']
  result_rows.loc[model_df.index, 'is_anomaly'] = model_df['is_anomaly']

  cluster_counts = (
    model_df['cluster']
    .value_counts()
    .reindex(range(1, cluster_count + 1), fill_value=0)
    .astype(int)
    .tolist()
  )

  centroids = scaler.inverse_transform(kmeans.cluster_centers_)
  centroid_df = pd.DataFrame(centroids, columns=selected_features)
  centroid_df.insert(0, 'cluster', np.arange(1, cluster_count + 1))
  centroid_df.insert(1, 'registros', cluster_counts)
  centroid_df = centroid_df.round(3)

  anomaly_count = int(model_df['is_anomaly'].sum())

  return {
    'error': '',
    'rows': result_rows,
    'modeled_rows': model_df,
    'cluster_counts': cluster_counts,
    'anomaly_count': anomaly_count,
    'centroids': centroid_df,
    'silhouette_score': silhouette_value,
    'davies_bouldin': davies_value,
    'calinski_harabasz': calinski_value,
    'inertia': float(kmeans.inertia_),
  }


def build_histogram(values: pd.Series, bins: int = 12) -> pd.DataFrame:
  clean_values = pd.to_numeric(values, errors='coerce').dropna()
  if clean_values.empty:
    return pd.DataFrame(columns=['label', 'count'])

  hist_counts, bin_edges = np.histogram(clean_values.to_numpy(), bins=bins)
  labels = [f'{bin_edges[index]:.2f}' for index in range(len(bin_edges) - 1)]

  return pd.DataFrame({'label': labels, 'count': hist_counts})


def _is_informative_distribution(values: pd.Series) -> bool:
  clean_values = pd.to_numeric(values, errors='coerce').dropna()
  if clean_values.empty:
    return False

  unique_values = clean_values.nunique(dropna=True)
  return unique_values > 1


def pick_histogram_feature(rows_df: pd.DataFrame, numeric_columns: list[str], selected_features: list[str]) -> str | None:
  priority_candidates = [
    'return_pct',
    'price_change_24h_pct',
    'price_change_7d_pct',
    'price_change_30d_pct',
    'volatility_pct',
    'range_pct',
  ]

  ordered_candidates: list[str] = []
  for candidate in [*priority_candidates, *selected_features, *numeric_columns]:
    if candidate in ordered_candidates:
      continue
    ordered_candidates.append(candidate)

  for candidate in ordered_candidates:
    if candidate not in rows_df.columns:
      continue
    if _is_informative_distribution(rows_df[candidate]):
      return candidate

  for candidate in ordered_candidates:
    if candidate in rows_df.columns:
      return candidate

  return None


def build_scatter_sample(modeled_rows: pd.DataFrame, x_feature: str, y_feature: str) -> pd.DataFrame:
  if modeled_rows.empty:
    return pd.DataFrame(columns=['x', 'y', 'cluster_label'])

  clean_rows = modeled_rows[[x_feature, y_feature, 'cluster']].dropna().copy()
  if clean_rows.empty:
    return pd.DataFrame(columns=['x', 'y', 'cluster_label'])

  step = max(1, len(clean_rows) // 200)
  sampled = clean_rows.iloc[::step].copy()
  sampled['cluster_label'] = sampled['cluster'].astype(int).astype(str)
  sampled = sampled.rename(columns={x_feature: 'x', y_feature: 'y'})
  return sampled


def build_dashboard(
  dataset_df: pd.DataFrame,
  selected_features: list[str],
  cluster_count: int,
  anomaly_threshold: float,
) -> dict:
  visible_columns = get_visible_columns(dataset_df)
  numeric_columns = get_numeric_columns(dataset_df, visible_columns)

  sanitized_features = [feature for feature in selected_features if feature in numeric_columns]
  if len(sanitized_features) < 3:
    defaults = pick_default_features(numeric_columns)
    sanitized_features = defaults if len(defaults) >= 3 else numeric_columns[:3]

  column_types = get_column_types(dataset_df, visible_columns)
  data_quality = get_data_quality(dataset_df, visible_columns)
  descriptive_stats = compute_descriptive_stats(dataset_df, numeric_columns)
  clustering_result = run_kmeans(dataset_df, sanitized_features, cluster_count, anomaly_threshold)

  trend_feature = 'close' if 'close' in numeric_columns else (sanitized_features[0] if sanitized_features else None)
  histogram_feature = pick_histogram_feature(clustering_result['rows'], numeric_columns, sanitized_features)
  scatter_x = sanitized_features[1] if len(sanitized_features) > 1 else (sanitized_features[0] if sanitized_features else None)
  scatter_y = sanitized_features[2] if len(sanitized_features) > 2 else scatter_x

  histogram = (
    build_histogram(clustering_result['rows'][histogram_feature]) if histogram_feature else pd.DataFrame(columns=['label', 'count'])
  )
  scatter = (
    build_scatter_sample(clustering_result['modeled_rows'], scatter_x, scatter_y)
    if scatter_x and scatter_y
    else pd.DataFrame(columns=['x', 'y', 'cluster_label'])
  )

  return {
    'dataset_rows': dataset_df,
    'visible_columns': visible_columns,
    'numeric_columns': numeric_columns,
    'column_types': column_types,
    'data_quality': data_quality,
    'descriptive_stats': descriptive_stats,
    'clustering_result': clustering_result,
    'selected_features': sanitized_features,
    'trend_feature': trend_feature,
    'histogram_feature': histogram_feature,
    'scatter_x_feature': scatter_x,
    'scatter_y_feature': scatter_y,
    'histogram': histogram,
    'scatter': scatter,
    'dimensions_text': f'{len(dataset_df)} filas x {len(visible_columns)} columnas',
  }
