from __future__ import annotations

import pandas as pd
import streamlit as st

from app.constants import FEATURE_LABELS


def _feature_label(feature_name: str | None) -> str:
  if feature_name is None:
    return 'N/A'
  return FEATURE_LABELS.get(feature_name, feature_name)


def _build_trend_frame(rows_df: pd.DataFrame, feature_name: str | None) -> pd.DataFrame:
  name_candidates = ['name', 'cryptocurrency', 'coin_name', 'coin_id', 'symbol']
  name_column = next((candidate for candidate in name_candidates if candidate in rows_df.columns), None)

  output_columns = ['x', 'nombre_cripto', 'valor']
  if feature_name is None or feature_name not in rows_df.columns:
    return pd.DataFrame(columns=output_columns)

  source_columns = [feature_name]
  if 'date' in rows_df.columns:
    source_columns.insert(0, 'date')
  if name_column:
    source_columns.append(name_column)

  trend_df = rows_df[source_columns].copy()
  if name_column:
    trend_df['nombre_cripto'] = trend_df[name_column].astype(str).str.strip()
    if name_column == 'coin_id':
      trend_df['nombre_cripto'] = (
        trend_df['nombre_cripto'].str.replace('_', ' ', regex=False).str.replace('-', ' ', regex=False).str.title()
      )

    invalid_name_mask = trend_df['nombre_cripto'].eq('') | trend_df['nombre_cripto'].str.lower().eq('nan')
    trend_df.loc[invalid_name_mask, 'nombre_cripto'] = 'Cripto desconocida'
  else:
    trend_df['nombre_cripto'] = 'Cripto desconocida'

  trend_df['valor'] = pd.to_numeric(trend_df[feature_name], errors='coerce')
  trend_df = trend_df.dropna(subset=['valor']).reset_index(drop=True)

  if trend_df.empty:
    return pd.DataFrame(columns=output_columns)

  if 'date' in trend_df.columns:
    trend_df['date_parsed'] = pd.to_datetime(trend_df['date'], errors='coerce', dayfirst=True)
    if trend_df['date_parsed'].notna().any():
      trend_df = trend_df.sort_values('date_parsed').reset_index(drop=True)
      trend_df['x'] = trend_df['date_parsed']
      return trend_df[output_columns]

  trend_df['x'] = range(1, len(trend_df) + 1)
  return trend_df[output_columns]


def _build_insights(trend_df: pd.DataFrame, scatter_df: pd.DataFrame) -> list[str]:
  insights: list[str] = []

  if not trend_df.empty:
    first_value = float(trend_df['valor'].iloc[0])
    last_value = float(trend_df['valor'].iloc[-1])
    if abs(first_value) > 1e-9:
      variation = ((last_value - first_value) / first_value) * 100
      insights.append(f'Dinamica principal: variacion acumulada de {variation:+.2f}% en la serie base.')

    rolling_std = trend_df['valor'].rolling(window=min(30, len(trend_df)), min_periods=5).std().dropna()
    if not rolling_std.empty:
      insights.append(f'Riesgo local: volatilidad movil aproximada de {rolling_std.iloc[-1]:,.2f} unidades.')

  if not scatter_df.empty and 'cluster_label' in scatter_df.columns:
    cluster_distribution = scatter_df['cluster_label'].value_counts(normalize=True).mul(100).round(1)
    dominant_cluster = cluster_distribution.index[0]
    dominant_share = cluster_distribution.iloc[0]
    insights.append(
      f'Espacio de estados: el cluster C{dominant_cluster} domina con {dominant_share:.1f}% de prevalencia muestral.'
    )

  if not insights:
    insights.append('No hay suficiente evidencia estadistica para generar lectura automatica de transiciones.')

  return insights


def _build_validation_table(trend_df: pd.DataFrame) -> pd.DataFrame:
  if trend_df.empty:
    return pd.DataFrame(columns=['Momento', 'Valor'])

  values = trend_df['valor']
  return pd.DataFrame(
    {
      'Momento': [
        'Media',
        'Desv. estandar',
        'Sesgo (skewness)',
        'Curtosis (Fisher)',
        'Minimo',
        'Maximo',
      ],
      'Valor': [
        f'{values.mean():,.4f}',
        f'{values.std(ddof=0):,.4f}',
        f'{values.skew():,.4f}',
        f'{values.kurtosis():,.4f}',
        f'{values.min():,.4f}',
        f'{values.max():,.4f}',
      ],
    }
  )


def render_exploration_screen(dashboard: dict) -> None:
  clustering_rows = dashboard['clustering_result']['rows']
  trend_feature = dashboard['trend_feature']
  histogram_feature = dashboard['histogram_feature']
  scatter_x_feature = dashboard['scatter_x_feature']
  scatter_y_feature = dashboard['scatter_y_feature']

  with st.container(border=True):
    st.markdown('#### Rango temporal y parametros de lectura')
    summary_cols = st.columns(3)
    summary_cols[0].caption('Intervalo: Diario (UTC 08:00)')
    summary_cols[1].caption(f'Variable base: {_feature_label(trend_feature)}')
    summary_cols[2].caption(f'Scatter: {_feature_label(scatter_x_feature)} vs {_feature_label(scatter_y_feature)}')

  trend_df = _build_trend_frame(clustering_rows, trend_feature)
  histogram_df = dashboard['histogram'].copy()
  scatter_df = dashboard['scatter'].copy()

  chart_col, insight_col = st.columns([2.05, 1], vertical_alignment='top')

  with chart_col:
    with st.container(border=True):
      st.markdown('#### Serie temporal con regimenes inferidos')
      if trend_df.empty:
        st.info('No hay datos suficientes para dibujar la serie temporal.')
      else:
        st.line_chart(trend_df, x='x', y='valor')
        st.caption(
          f'Serie sobre {_feature_label(trend_feature)} con enfoque en continuidad temporal y eventos de cambio de fase.'
        )

  with insight_col:
    with st.container(border=True):
      st.markdown('#### Lectura academica y transiciones')
      insights = _build_insights(trend_df, scatter_df)
      for index, insight in enumerate(insights, start=1):
        st.markdown(f'{index}. {insight}')

      st.info('Validacion operativa recomendada: contrastar estos hallazgos con ventana movil y stress periods.')

  left_chart_col, right_chart_col = st.columns(2, vertical_alignment='top')

  with left_chart_col:
    with st.container(border=True):
      st.markdown(f'#### Distribucion empirica ({_feature_label(histogram_feature)})')
      if histogram_df.empty:
        st.info('No hay datos suficientes para construir el histograma.')
      else:
        st.bar_chart(histogram_df.set_index('label')['count'])
        histogram_summary = pd.DataFrame(
          {
            'Metrica': ['Frecuencia total', 'Pico maximo'],
            'Valor': [
              int(histogram_df['count'].sum()),
              int(histogram_df['count'].max()),
            ],
          }
        )
        st.dataframe(histogram_summary, hide_index=True, width='stretch')

  with right_chart_col:
    with st.container(border=True):
      st.markdown('#### Espacio de estados: retorno vs volatilidad')
      if scatter_df.empty:
        st.info('No hay datos suficientes para la grafica de dispersion.')
      else:
        x_axis_label = _feature_label(scatter_x_feature)
        y_axis_label = _feature_label(scatter_y_feature)

        scatter_plot_df = scatter_df.rename(columns={'x': x_axis_label, 'y': y_axis_label})
        scatter_plot_df['cluster_label'] = scatter_plot_df['cluster_label'].astype(str)

        st.scatter_chart(scatter_plot_df, x=x_axis_label, y=y_axis_label, color='cluster_label')

        cluster_share_df = (
          scatter_plot_df['cluster_label']
          .value_counts(normalize=True)
          .mul(100)
          .round(2)
          .rename_axis('Cluster')
          .reset_index(name='Participacion (%)')
        )
        st.dataframe(cluster_share_df, hide_index=True, width='stretch')

  with st.container(border=True):
    st.markdown('#### Validacion formal de hipotesis')
    validation_df = _build_validation_table(trend_df)
    if validation_df.empty:
      st.info('No hay datos para validar momentos estadisticos.')
    else:
      st.dataframe(validation_df, hide_index=True, width='stretch')

    with st.expander('Tabla descriptiva completa'):
      descriptive_df = dashboard['descriptive_stats'].copy()
      if descriptive_df.empty:
        st.info('No hay variables numericas suficientes para estadistica descriptiva.')
      else:
        descriptive_df = descriptive_df.rename(
          columns={
            'variable': 'Variable',
            'n': 'N',
            'media': 'Media',
            'mediana': 'Mediana',
            'desv_std': 'Desv. estandar',
            'min': 'Min',
            'max': 'Max',
          }
        )
        st.dataframe(descriptive_df, hide_index=True, width='stretch')

  action_cols = st.columns(2)

  if trend_df.empty:
    series_bytes = b''
  else:
    series_bytes = trend_df.to_csv(index=False).encode('utf-8')

  action_cols[0].download_button(
    'Exportar serie analitica (.csv)',
    data=series_bytes,
    file_name='serie_regimenes.csv',
    mime='text/csv',
    disabled=not bool(series_bytes),
    width='stretch',
  )

  report_text = (
    'Reporte sintetico de exploracion\n'
    f'- Serie base: {_feature_label(trend_feature)}\n'
    f'- Histograma base: {_feature_label(histogram_feature)}\n'
    f'- Eje X scatter: {_feature_label(scatter_x_feature)}\n'
    f'- Eje Y scatter: {_feature_label(scatter_y_feature)}\n'
    f'- Muestras analizadas: {len(clustering_rows):,}\n'
  )

  action_cols[1].download_button(
    'Descargar informe academico (.txt)',
    data=report_text.encode('utf-8'),
    file_name='informe_exploracion.txt',
    mime='text/plain',
    width='stretch',
  )
