from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics import format_value
from app.constants import FEATURE_LABELS, MAX_CLUSTER_COUNT


def _feature_label(feature_name: str) -> str:
  return FEATURE_LABELS.get(feature_name, feature_name)


def _sanitize_features(selected_features: list[str], numeric_columns: list[str]) -> list[str]:
  unique_features: list[str] = []
  for feature in selected_features:
    if feature not in numeric_columns:
      continue
    if feature in unique_features:
      continue
    unique_features.append(feature)
  return unique_features[:5]


def _build_feature_justification(selected_features: list[str]) -> str:
  if not selected_features:
    return 'Sin seleccion activa.'

  labels = ', '.join(_feature_label(feature) for feature in selected_features)
  return (
    f'{labels}. Estas variables cubren nivel de precio, actividad de mercado y '
    'variabilidad diaria para identificar patrones no supervisados.'
  )


def _metric_text(value: float | int | None, digits: int = 2) -> str:
  if value is None:
    return 'N/A'
  return f'{float(value):,.{digits}f}'


def _build_cluster_profiles(modeled_rows: pd.DataFrame) -> pd.DataFrame:
  if modeled_rows.empty or 'cluster' not in modeled_rows.columns:
    return pd.DataFrame()

  clean_rows = modeled_rows[modeled_rows['cluster'].notna()].copy()
  if clean_rows.empty:
    return pd.DataFrame()

  aggregations = {
    'registros': ('cluster', 'size'),
  }
  if 'return_pct' in clean_rows.columns:
    aggregations['retorno_medio'] = ('return_pct', 'mean')
  if 'volatility_pct' in clean_rows.columns:
    aggregations['volatilidad_media'] = ('volatility_pct', 'mean')
  if 'anomaly_score' in clean_rows.columns:
    aggregations['score_anomalia'] = ('anomaly_score', 'mean')

  profile_df = clean_rows.groupby('cluster').agg(**aggregations).reset_index()
  profile_df['cluster'] = profile_df['cluster'].astype(int)

  total_rows = int(profile_df['registros'].sum())
  profile_df['participacion'] = (profile_df['registros'] / total_rows * 100).round(1) if total_rows else 0.0

  profile_df['regimen'] = profile_df['cluster'].apply(lambda cluster_id: f'Regimen {cluster_id}')
  profile_df['css_tag'] = 'profile-neutral'

  if 'retorno_medio' in profile_df.columns and len(profile_df) >= 2:
    top_cluster = int(profile_df.sort_values('retorno_medio', ascending=False)['cluster'].iloc[0])
    bottom_cluster = int(profile_df.sort_values('retorno_medio', ascending=True)['cluster'].iloc[0])

    profile_df.loc[profile_df['cluster'] == top_cluster, 'regimen'] = 'Expansion / Rally alcista'
    profile_df.loc[profile_df['cluster'] == top_cluster, 'css_tag'] = 'profile-positive'

    profile_df.loc[profile_df['cluster'] == bottom_cluster, 'regimen'] = 'Correccion y estres de liquidez'
    profile_df.loc[profile_df['cluster'] == bottom_cluster, 'css_tag'] = 'profile-negative'

  neutral_mask = profile_df['css_tag'] == 'profile-neutral'
  profile_df.loc[neutral_mask, 'regimen'] = 'Mercado estable / acumulacion'
  profile_df.loc[neutral_mask, 'css_tag'] = 'profile-stable'

  return profile_df.sort_values('participacion', ascending=False).reset_index(drop=True)


def _build_transition_matrix(rows_df: pd.DataFrame) -> pd.DataFrame:
  if rows_df.empty or 'cluster' not in rows_df.columns:
    return pd.DataFrame()

  transition_df = rows_df.copy()
  transition_df = transition_df[transition_df['cluster'].notna()].copy()
  if transition_df.empty:
    return pd.DataFrame()

  transition_df['cluster'] = pd.to_numeric(transition_df['cluster'], errors='coerce')
  transition_df = transition_df.dropna(subset=['cluster']).copy()
  transition_df['cluster'] = transition_df['cluster'].astype(int)

  if 'date' in transition_df.columns:
    transition_df['date_parsed'] = pd.to_datetime(transition_df['date'], errors='coerce', dayfirst=True)
    if transition_df['date_parsed'].notna().any():
      transition_df = transition_df.sort_values('date_parsed')

  transition_df['next_cluster'] = transition_df['cluster'].shift(-1)
  transition_df = transition_df.dropna(subset=['next_cluster']).copy()
  if transition_df.empty:
    return pd.DataFrame()

  transition_df['next_cluster'] = transition_df['next_cluster'].astype(int)
  matrix = pd.crosstab(transition_df['cluster'], transition_df['next_cluster'], normalize='index').round(3)
  matrix.index = [f'R{int(value)}' for value in matrix.index]
  matrix.columns = [f'R{int(value)}' for value in matrix.columns]
  return matrix.reset_index().rename(columns={'index': 'Estado t'})


def _build_cluster_comparison_table(profile_df: pd.DataFrame) -> pd.DataFrame:
  columns = [
    'Cluster',
    'Nombre del cluster',
    'Registros',
    'Participacion (%)',
    'Retorno medio (%)',
    'Volatilidad media (%)',
    'Score anomalia',
  ]
  if profile_df.empty:
    return pd.DataFrame(columns=columns)

  comparison_df = pd.DataFrame()
  comparison_df['Cluster'] = profile_df['cluster'].map(lambda value: f'R{int(value)}')
  comparison_df['Nombre del cluster'] = profile_df['regimen']
  comparison_df['Registros'] = profile_df['registros'].astype(int)
  comparison_df['Participacion (%)'] = profile_df['participacion'].astype(float).round(1)
  comparison_df['Retorno medio (%)'] = profile_df.get('retorno_medio', pd.Series(index=profile_df.index))
  comparison_df['Volatilidad media (%)'] = profile_df.get('volatilidad_media', pd.Series(index=profile_df.index))
  comparison_df['Score anomalia'] = profile_df.get('score_anomalia', pd.Series(index=profile_df.index))

  for column in ['Retorno medio (%)', 'Volatilidad media (%)', 'Score anomalia']:
    comparison_df[column] = pd.to_numeric(comparison_df[column], errors='coerce').round(3)

  return comparison_df[columns]


def _build_group_descriptive_stats(modeled_rows: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
  columns = ['Cluster', 'Variable', 'N', 'Media', 'Mediana', 'Desv. estandar', 'Min', 'Max']
  if modeled_rows.empty or 'cluster' not in modeled_rows.columns:
    return pd.DataFrame(columns=columns)

  available_features = [feature for feature in selected_features if feature in modeled_rows.columns]
  if not available_features:
    return pd.DataFrame(columns=columns)

  source_df = modeled_rows[['cluster', *available_features]].copy()
  source_df['cluster'] = pd.to_numeric(source_df['cluster'], errors='coerce')
  source_df = source_df.dropna(subset=['cluster'])
  if source_df.empty:
    return pd.DataFrame(columns=columns)

  source_df['cluster'] = source_df['cluster'].astype(int)
  grouped_frames: list[pd.DataFrame] = []

  for feature in available_features:
    feature_df = pd.DataFrame(
      {
        'cluster': source_df['cluster'],
        'value': pd.to_numeric(source_df[feature], errors='coerce'),
      }
    ).dropna(subset=['value'])
    if feature_df.empty:
      continue

    stats_df = feature_df.groupby('cluster')['value'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
    stats_df['std'] = stats_df['std'].fillna(0.0)
    stats_df['cluster'] = stats_df['cluster'].map(lambda value: f'R{int(value)}')
    stats_df = stats_df.rename(
      columns={
        'cluster': 'Cluster',
        'count': 'N',
        'mean': 'Media',
        'median': 'Mediana',
        'std': 'Desv. estandar',
        'min': 'Min',
        'max': 'Max',
      }
    )
    stats_df.insert(1, 'Variable', _feature_label(feature))
    grouped_frames.append(stats_df)

  if not grouped_frames:
    return pd.DataFrame(columns=columns)

  output_df = pd.concat(grouped_frames, ignore_index=True)
  output_df['N'] = output_df['N'].astype(int)
  for column in ['Media', 'Mediana', 'Desv. estandar', 'Min', 'Max']:
    output_df[column] = pd.to_numeric(output_df[column], errors='coerce').round(4)
  return output_df[columns]


def _build_centroid_interpretations(
  centroid_df: pd.DataFrame,
  modeled_rows: pd.DataFrame,
  selected_features: list[str],
  profile_df: pd.DataFrame,
) -> list[str]:
  if centroid_df.empty or modeled_rows.empty:
    return []

  available_features = [
    feature for feature in selected_features if feature in centroid_df.columns and feature in modeled_rows.columns
  ]
  if not available_features:
    return []

  feature_means: dict[str, float] = {}
  for feature in available_features:
    values = pd.to_numeric(modeled_rows[feature], errors='coerce').dropna()
    if values.empty:
      continue
    feature_means[feature] = float(values.mean())
  if not feature_means:
    return []

  cluster_name_map: dict[int, str] = {}
  if not profile_df.empty and {'cluster', 'regimen'}.issubset(profile_df.columns):
    cluster_name_map = {
      int(cluster_id): str(regimen)
      for cluster_id, regimen in zip(profile_df['cluster'], profile_df['regimen'], strict=False)
    }

  interpretations: list[str] = []
  ordered_centroids = centroid_df.copy()
  ordered_centroids['cluster'] = pd.to_numeric(ordered_centroids['cluster'], errors='coerce')
  ordered_centroids = ordered_centroids.dropna(subset=['cluster']).sort_values('cluster')

  for _, row in ordered_centroids.iterrows():
    cluster_id = int(row['cluster'])
    deviations: list[tuple[str, float]] = []

    for feature, reference in feature_means.items():
      centroid_value = pd.to_numeric(pd.Series([row.get(feature)]), errors='coerce').iloc[0]
      if pd.isna(centroid_value):
        continue
      if abs(reference) > 1e-9:
        deviation_pct = ((float(centroid_value) - reference) / abs(reference)) * 100
      else:
        deviation_pct = (float(centroid_value) - reference) * 100
      deviations.append((feature, deviation_pct))

    if not deviations:
      continue

    deviations.sort(key=lambda item: abs(item[1]), reverse=True)
    focus_features = deviations[:2]
    clauses: list[str] = []
    for feature, deviation_pct in focus_features:
      direction = 'por encima' if deviation_pct >= 0 else 'por debajo'
      clauses.append(f'{_feature_label(feature)} {direction} de la media ({abs(deviation_pct):.1f}%)')

    regime_name = cluster_name_map.get(cluster_id, f'Regimen {cluster_id}')
    interpretations.append(f'R{cluster_id} ({regime_name}): ' + '; '.join(clauses) + '.')

  return interpretations


def _build_similarity_table(centroid_df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
  columns = ['Comparacion', 'Distancia centroides', 'Lectura']
  if centroid_df.empty:
    return pd.DataFrame(columns=columns)

  available_features = [feature for feature in selected_features if feature in centroid_df.columns]
  if not available_features:
    return pd.DataFrame(columns=columns)

  source_df = centroid_df[['cluster', *available_features]].copy()
  source_df['cluster'] = pd.to_numeric(source_df['cluster'], errors='coerce')
  source_df = source_df.dropna(subset=['cluster']).copy()
  if len(source_df) < 2:
    return pd.DataFrame(columns=columns)

  for feature in available_features:
    source_df[feature] = pd.to_numeric(source_df[feature], errors='coerce')

  rows: list[dict] = []
  for left_index in range(len(source_df)):
    for right_index in range(left_index + 1, len(source_df)):
      left_row = source_df.iloc[left_index]
      right_row = source_df.iloc[right_index]
      diff_series = (left_row[available_features] - right_row[available_features]).astype(float)
      distance = float((diff_series.pow(2).sum()) ** 0.5)
      rows.append(
        {
          'Comparacion': f"R{int(left_row['cluster'])} vs R{int(right_row['cluster'])}",
          'Distancia centroides': distance,
        }
      )

  if not rows:
    return pd.DataFrame(columns=columns)

  similarity_df = pd.DataFrame(rows)
  max_distance = float(similarity_df['Distancia centroides'].max())
  if max_distance <= 1e-9:
    similarity_df['Lectura'] = 'Alta similitud'
  else:
    normalized_distance = similarity_df['Distancia centroides'] / max_distance
    similarity_df['Lectura'] = normalized_distance.map(
      lambda value: 'Alta similitud' if value <= 0.33 else ('Similitud moderada' if value <= 0.66 else 'Diferencia marcada')
    )

  similarity_df['Distancia centroides'] = similarity_df['Distancia centroides'].round(3)
  return similarity_df[columns]


def _build_key_findings(
  profile_df: pd.DataFrame,
  transition_df: pd.DataFrame,
  similarity_df: pd.DataFrame,
  clustering_result: dict,
) -> list[str]:
  findings: list[str] = []

  if not profile_df.empty and {'cluster', 'regimen', 'participacion'}.issubset(profile_df.columns):
    dominant = profile_df.sort_values('participacion', ascending=False).iloc[0]
    findings.append(
      f"El cluster R{int(dominant['cluster'])} ({dominant['regimen']}) concentra {float(dominant['participacion']):.1f}% de las observaciones modeladas."
    )

  if not profile_df.empty and {'cluster', 'retorno_medio'}.issubset(profile_df.columns):
    top_cluster = profile_df.sort_values('retorno_medio', ascending=False).iloc[0]
    bottom_cluster = profile_df.sort_values('retorno_medio', ascending=True).iloc[0]
    spread = float(top_cluster['retorno_medio']) - float(bottom_cluster['retorno_medio'])
    findings.append(
      f"La brecha de retorno medio entre R{int(top_cluster['cluster'])} y R{int(bottom_cluster['cluster'])} es de {spread:.2f} puntos porcentuales."
    )

  if not transition_df.empty and 'Estado t' in transition_df.columns:
    transition_work = transition_df.set_index('Estado t')
    persistence_scores: dict[str, float] = {}
    for state in transition_work.index:
      if state in transition_work.columns:
        persistence_scores[state] = float(transition_work.loc[state, state])
    if persistence_scores:
      stable_state = max(persistence_scores, key=persistence_scores.get)
      findings.append(
        f'La mayor persistencia temporal ocurre en {stable_state}, con continuidad de {persistence_scores[stable_state] * 100:.1f}%.'
      )

  if not similarity_df.empty:
    furthest_pair = similarity_df.sort_values('Distancia centroides', ascending=False).iloc[0]
    findings.append(
      f"La comparacion {furthest_pair['Comparacion']} registra la mayor distancia entre centroides ({float(furthest_pair['Distancia centroides']):.3f})."
    )

  anomaly_count = int(clustering_result.get('anomaly_count', 0))
  modeled_rows = clustering_result.get('modeled_rows', pd.DataFrame())
  modeled_count = len(modeled_rows) if hasattr(modeled_rows, '__len__') else 0
  if modeled_count > 0:
    findings.append(
      f'El modelo marca {anomaly_count} observaciones atipicas, equivalentes al {anomaly_count / modeled_count * 100:.1f}% de los registros modelados.'
    )

  fallback_findings = [
    'La segmentacion evidencia comportamientos diferenciados entre fases de mercado.',
    'La dispersion entre centroides respalda que existen patrones no supervisados distinguibles.',
    'La composicion por cluster permite priorizar decisiones con base en evidencia cuantitativa.',
  ]
  while len(findings) < 3:
    findings.append(fallback_findings[len(findings)])

  return findings[:5]


def _build_recommendations(profile_df: pd.DataFrame, transition_df: pd.DataFrame, clustering_result: dict) -> list[str]:
  recommendations: list[str] = []

  if not profile_df.empty and {'cluster', 'regimen', 'participacion'}.issubset(profile_df.columns):
    dominant = profile_df.sort_values('participacion', ascending=False).iloc[0]
    recommendations.append(
      f"Prioriza monitoreo operativo para R{int(dominant['cluster'])} ({dominant['regimen']}), porque explica {float(dominant['participacion']):.1f}% de los casos."
    )

  if not transition_df.empty and 'Estado t' in transition_df.columns:
    transition_work = transition_df.set_index('Estado t')
    persistence_scores: dict[str, float] = {}
    for state in transition_work.index:
      if state in transition_work.columns:
        persistence_scores[state] = float(transition_work.loc[state, state])
    if persistence_scores:
      stable_state = max(persistence_scores, key=persistence_scores.get)
      stable_value = persistence_scores[stable_state]
      if stable_value >= 0.6:
        recommendations.append(
          f'Cuando el sistema entre en {stable_state}, considera estrategias de continuidad de corto plazo: su persistencia es {stable_value * 100:.1f}%. '
        )
      else:
        recommendations.append(
          f'Incrementa la frecuencia de rebalanceo; la persistencia maxima es {stable_state} con {stable_value * 100:.1f}%, lo que sugiere cambios de fase frecuentes.'
        )

  anomaly_count = int(clustering_result.get('anomaly_count', 0))
  modeled_rows = clustering_result.get('modeled_rows', pd.DataFrame())
  modeled_count = len(modeled_rows) if hasattr(modeled_rows, '__len__') else 0
  if modeled_count > 0:
    anomaly_ratio = anomaly_count / modeled_count * 100
    recommendations.append(
      f'Define alertas de riesgo cuando el score de anomalia supere el umbral actual; hoy el modelo detecta {anomaly_ratio:.1f}% de casos atipicos.'
    )

  fallback_recommendations = [
    'Reentrena K-Means en ventanas temporales moviles para capturar cambios estructurales del mercado.',
    'Contrasta las etiquetas de cluster con eventos externos (tasas, noticias, regulacion) antes de tomar decisiones de asignacion.',
    'Valida el numero de clusters con analisis de estabilidad para reducir sobreajuste por particion puntual.',
  ]
  while len(recommendations) < 3:
    recommendations.append(fallback_recommendations[len(recommendations)])

  return recommendations[:5]


def _build_model_limitations() -> list[str]:
  return [
    'K-Means asume grupos de forma aproximadamente esferica y distancia euclidiana, por lo que puede perder estructuras no lineales.',
    'La segmentacion depende de las variables seleccionadas y de la estandarizacion aplicada; cambios en estos supuestos alteran los clusters.',
    'La deteccion de anomalias usa un umbral fijo sobre distancia al centroide, sensible a periodos extremos o cambios de volatilidad.',
    'El modelo es descriptivo y no causal: no incorpora factores exogenos como noticias, politica monetaria o liquidez global.',
  ]


def render_mining_screen(dashboard: dict) -> None:
  numeric_columns = dashboard['numeric_columns']
  if len(numeric_columns) < 3:
    st.error('Se requieren al menos 3 variables numericas para ejecutar mineria no supervisada.')
    return

  left_col, right_col = st.columns([1.08, 1.62], vertical_alignment='top')

  with left_col:
    with st.container(border=True):
      st.markdown('#### Hiperparametros del algoritmo')
      st.caption('Estandarizacion: Z-score · distancia euclidiana')

      next_cluster_count = st.slider(
        'Numero de clusters (k)',
        min_value=2,
        max_value=MAX_CLUSTER_COUNT,
        value=int(st.session_state.cluster_count),
        step=1,
      )
      next_anomaly_threshold = st.slider(
        'Umbral de anomalia (distancia estandarizada)',
        min_value=1.1,
        max_value=4.0,
        value=float(st.session_state.anomaly_threshold),
        step=0.1,
      )

      if next_cluster_count != st.session_state.cluster_count:
        st.session_state.cluster_count = next_cluster_count
        st.rerun()

      if abs(next_anomaly_threshold - st.session_state.anomaly_threshold) > 1e-9:
        st.session_state.anomaly_threshold = next_anomaly_threshold
        st.rerun()

    default_features = _sanitize_features(st.session_state.selected_features, numeric_columns)
    if len(default_features) < 3:
      default_features = dashboard['selected_features']

    with st.container(border=True):
      st.markdown('#### Espacio de caracteristicas')
      selected_input = st.multiselect(
        'Variables para agrupamiento (minimo 3, maximo 5)',
        options=numeric_columns,
        default=default_features,
        format_func=_feature_label,
      )

      selected_features = _sanitize_features(selected_input, numeric_columns)
      if len(selected_input) > 5:
        st.warning('Solo se consideran las primeras 5 variables seleccionadas.')

      if len(selected_features) < 3:
        st.warning('Selecciona al menos 3 variables para ejecutar K-Means.')
        selected_features = _sanitize_features(st.session_state.selected_features, numeric_columns)

      if selected_features != st.session_state.selected_features:
        st.session_state.selected_features = selected_features
        st.rerun()

      st.caption(f'Activas: {len(st.session_state.selected_features)} variables')

    with st.container(border=True):
      st.markdown('#### Metodo de inicializacion')
      init_cols = st.columns(2)
      init_cols[0].metric('Max iteraciones', '300')
      init_cols[1].metric('Semilla random', '42')
      init_cols[0].metric('Early stop', 'True')
      init_cols[1].metric('Tolerancia', '1e-4')

      if st.button('Ejecutar calibracion K-Means', type='primary', width='stretch'):
        st.toast('Calibracion aplicada con los hiperparametros actuales.')

    with st.container(border=True):
      st.markdown('#### Fundamento teorico')
      st.write(
        'La mineria no supervisada con K-Means segmenta fases del mercado en funcion de retorno, '
        'volatilidad y liquidez. El enfoque permite detectar cambios estructurales de regimen.'
      )
      st.caption(f'Justificacion de variables: {_build_feature_justification(st.session_state.selected_features)}')

  clustering_result = dashboard['clustering_result']
  if clustering_result['error']:
    with right_col:
      st.error(clustering_result['error'])
    return

  with right_col:
    with st.container(border=True):
      st.markdown('#### Metricas globales de validacion de clusters')

      metric_cols_top = st.columns(3)
      metric_cols_top[0].metric('Coeficiente de silueta', _metric_text(clustering_result.get('silhouette_score'), 2))
      metric_cols_top[1].metric('Inercia WCSS', _metric_text(clustering_result.get('inertia'), 1))
      metric_cols_top[2].metric('Calinski-Harabasz', _metric_text(clustering_result.get('calinski_harabasz'), 1))

      metric_cols_bottom = st.columns(3)
      metric_cols_bottom[0].metric('Davies-Bouldin', _metric_text(clustering_result.get('davies_bouldin'), 2))
      metric_cols_bottom[1].metric('Anomalias detectadas', format_value(clustering_result['anomaly_count']))
      metric_cols_bottom[2].metric('Registros modelados', format_value(len(clustering_result['modeled_rows'])))

    with st.container(border=True):
      st.markdown('#### Perfiles de regimenes segmentados')
      profile_df = _build_cluster_profiles(clustering_result['modeled_rows'])

      if profile_df.empty:
        st.info('No hay perfiles disponibles con la configuracion actual.')
      else:
        for _, row in profile_df.iterrows():
          retorno = _metric_text(row.get('retorno_medio'), 2) if 'retorno_medio' in row else 'N/A'
          volatilidad = _metric_text(row.get('volatilidad_media'), 2) if 'volatilidad_media' in row else 'N/A'
          score = _metric_text(row.get('score_anomalia'), 2) if 'score_anomalia' in row else 'N/A'

          st.markdown(
            f'''
            <section class="profile-card {row['css_tag']}">
              <div class="profile-header">
                <span class="profile-id">R{int(row['cluster'])}</span>
                <span class="profile-name">{row['regimen']}</span>
                <span class="profile-share">{row['participacion']:.1f}% de las observaciones</span>
              </div>
              <div class="profile-grid">
                <span>Retorno medio: <strong>{retorno}%</strong></span>
                <span>Volatilidad: <strong>{volatilidad}%</strong></span>
                <span>Score anomalia: <strong>{score}</strong></span>
                <span>Registros: <strong>{int(row['registros'])}</strong></span>
              </div>
            </section>
            ''',
            unsafe_allow_html=True,
          )

    with st.container(border=True):
      st.markdown('#### Matriz empirica de transicion markoviana')
      transition_df = _build_transition_matrix(clustering_result['rows'])

      if transition_df.empty:
        st.info('No hay suficientes observaciones para estimar transiciones entre regimenes.')
      else:
        st.dataframe(transition_df, hide_index=True, width='stretch')

    action_cols = st.columns(2)

    centroid_df = clustering_result['centroids']
    centroid_bytes = centroid_df.to_csv(index=False).encode('utf-8') if not centroid_df.empty else b''

    action_cols[0].download_button(
      'Exportar matriz de centroides (.csv)',
      data=centroid_bytes,
      file_name='centroides_kmeans.csv',
      mime='text/csv',
      disabled=centroid_df.empty,
      width='stretch',
    )

    if action_cols[1].button('Ver graficos en exploracion empirica', type='primary', width='stretch'):
      st.session_state.active_screen = 'Exploracion'
      st.rerun()

    summary_tab, centroid_tab, detail_tab = st.tabs(['Resumen', 'Centroides', 'Asignacion'])

    with summary_tab:
      if clustering_result['cluster_counts']:
        summary_df = pd.DataFrame(
          {
            'cluster': [f'C{index + 1}' for index in range(len(clustering_result['cluster_counts']))],
            'registros': clustering_result['cluster_counts'],
          }
        )
        modeled_count = len(clustering_result['modeled_rows'])
        summary_df['porcentaje'] = (summary_df['registros'] / modeled_count * 100).round(2) if modeled_count else 0.0

        st.bar_chart(summary_df.set_index('cluster')['registros'])
        st.dataframe(summary_df, hide_index=True, width='stretch')
      else:
        st.info('Aun no hay conteo de clusters disponible.')

    with centroid_tab:
      if centroid_df.empty:
        st.info('No fue posible calcular centroides con la configuracion actual.')
      else:
        renamed_columns = {feature: _feature_label(feature) for feature in st.session_state.selected_features}
        st.dataframe(centroid_df.rename(columns=renamed_columns), hide_index=True, width='stretch')

    with detail_tab:
      preview_columns = ['date', 'close', 'volume', 'return_pct', 'cluster', 'anomaly_score', 'is_anomaly']
      available_columns = [column for column in preview_columns if column in clustering_result['rows'].columns]
      if not available_columns:
        st.info('No hay columnas disponibles para mostrar la asignacion de cluster.')
      else:
        preview_source = clustering_result['rows']
        if 'cluster' in preview_source.columns:
          modeled_preview = preview_source[preview_source['cluster'].notna()].copy()
          if not modeled_preview.empty:
            preview_source = modeled_preview

        preview_df = preview_source[available_columns].head(20).copy()
        if 'cluster' in preview_df.columns:
          preview_df['cluster'] = preview_df['cluster'].fillna('N/A')

        if 'is_anomaly' in preview_df.columns:
          preview_df['Etiqueta'] = preview_df['is_anomaly'].map(lambda value: 'Atipico' if bool(value) else 'Normal')
          preview_df = preview_df.drop(columns=['is_anomaly'])

        preview_df = preview_df.rename(
          columns={
            'date': 'Fecha',
            'close': 'Cierre',
            'volume': 'Volumen',
            'return_pct': 'Retorno (%)',
            'cluster': 'Cluster',
            'anomaly_score': 'Score anomalia',
          }
        )
        st.dataframe(preview_df, hide_index=True, width='stretch')

    comparison_df = _build_cluster_comparison_table(profile_df)
    grouped_stats_df = _build_group_descriptive_stats(clustering_result['modeled_rows'], st.session_state.selected_features)
    centroid_interpretations = _build_centroid_interpretations(
      centroid_df,
      clustering_result['modeled_rows'],
      st.session_state.selected_features,
      profile_df,
    )
    similarity_df = _build_similarity_table(centroid_df, st.session_state.selected_features)
    key_findings = _build_key_findings(profile_df, transition_df, similarity_df, clustering_result)
    recommendations = _build_recommendations(profile_df, transition_df, clustering_result)
    limitations = _build_model_limitations()

    with st.container(border=True):
      st.markdown('#### Analisis e interpretacion de resultados')

      st.markdown('##### Tabla comparativa de los grupos')
      if comparison_df.empty:
        st.info('No hay informacion suficiente para construir la tabla comparativa de clusters.')
      else:
        st.dataframe(comparison_df, hide_index=True, width='stretch')

      st.markdown('##### Estadistica descriptiva por grupo')
      if grouped_stats_df.empty:
        st.info('No hay estadisticas descriptivas por cluster con la configuracion actual.')
      else:
        st.dataframe(grouped_stats_df, hide_index=True, width='stretch')

      st.markdown('##### Interpretacion de centroides')
      if not centroid_interpretations:
        st.info('No fue posible interpretar centroides con las variables activas.')
      else:
        for interpretation in centroid_interpretations:
          st.markdown(f'- {interpretation}')

      st.markdown('##### Diferencias y similitudes entre agrupamientos')
      if similarity_df.empty:
        st.info('No hay suficientes clusters para comparar distancias entre centroides.')
      else:
        st.dataframe(similarity_df, hide_index=True, width='stretch')

      st.markdown('##### Hallazgos relevantes')
      for index, finding in enumerate(key_findings, start=1):
        st.markdown(f'{index}. {finding}')

      st.markdown('##### Recomendaciones basadas en resultados')
      for index, recommendation in enumerate(recommendations, start=1):
        st.markdown(f'{index}. {recommendation}')

      st.markdown('##### Limitaciones del modelo desarrollado')
      for index, limitation in enumerate(limitations, start=1):
        st.markdown(f'{index}. {limitation}')
