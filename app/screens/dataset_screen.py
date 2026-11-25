from __future__ import annotations

import pandas as pd
import streamlit as st

from app.analytics import get_numeric_columns, get_visible_columns, pick_default_features
from app.constants import MIN_REQUIRED_COLUMNS, MIN_REQUIRED_ROWS
from app.data import enrich_dataset, load_default_dataset, parse_uploaded_csv


def _render_requirement(label: str, ok: bool) -> None:
  status_class = 'ok' if ok else 'alert'
  status_text = 'OK' if ok else 'Alerta'
  st.markdown(f'<div class="req-pill {status_class}">{status_text}: {label}</div>', unsafe_allow_html=True)


def _sync_selected_features() -> None:
  dataset_df = st.session_state.dataset_df
  visible_columns = get_visible_columns(dataset_df)
  numeric_columns = get_numeric_columns(dataset_df, visible_columns)
  default_features = pick_default_features(numeric_columns)

  if len(default_features) >= 3:
    st.session_state.selected_features = default_features
  else:
    st.session_state.selected_features = numeric_columns[:3]


def _build_structural_series(dataset_df: pd.DataFrame) -> pd.DataFrame:
  if dataset_df.empty:
    return pd.DataFrame(columns=['x', 'precio_cierre', 'volumen_normalizado'])

  if 'close' not in dataset_df.columns:
    return pd.DataFrame(columns=['x', 'precio_cierre', 'volumen_normalizado'])

  frame = pd.DataFrame()
  if 'date' in dataset_df.columns:
    parsed_dates = pd.to_datetime(dataset_df['date'], errors='coerce', dayfirst=True)
    if parsed_dates.notna().any():
      frame['x'] = parsed_dates
    else:
      frame['x'] = range(1, len(dataset_df) + 1)
  else:
    frame['x'] = range(1, len(dataset_df) + 1)

  frame['precio_cierre'] = pd.to_numeric(dataset_df['close'], errors='coerce')

  if 'volume' in dataset_df.columns:
    volume_series = pd.to_numeric(dataset_df['volume'], errors='coerce')
    max_volume = volume_series.max()
    if pd.notna(max_volume) and max_volume > 0:
      frame['volumen_normalizado'] = (volume_series / max_volume) * frame['precio_cierre'].max()

  frame = frame.dropna(subset=['precio_cierre']).reset_index(drop=True)
  return frame


def _build_sample_table(dataset_df: pd.DataFrame, visible_columns: list[str]) -> pd.DataFrame:
  sample_columns = [column for column in ['date', 'close', 'return_pct', 'volatility_pct', 'volume'] if column in visible_columns]
  if not sample_columns:
    sample_columns = visible_columns[:5]

  sample_df = dataset_df[sample_columns].head(8).copy()
  if sample_df.empty:
    return sample_df

  sample_df['integridad'] = sample_df.isna().any(axis=1).map(lambda has_nulls: 'Revision' if has_nulls else 'Validado')
  renamed = {
    'date': 'Timestamp UTC',
    'close': 'Precio cierre',
    'return_pct': 'Retorno diario (%)',
    'volatility_pct': 'Volatilidad (%)',
    'volume': 'Volumen 24h',
    'integridad': 'Estado integridad',
  }
  return sample_df.rename(columns={key: value for key, value in renamed.items() if key in sample_df.columns})


def render_dataset_screen(dashboard: dict) -> None:
  with st.container(border=True):
    st.markdown('#### Control de carga y fuente de datos')
    st.caption(
      f"{st.session_state.source_note} "
      'Importacion web equivalente a `read_csv`: carga un CSV con al menos 5 columnas y 500 filas.'
    )

    uploaded_file = st.file_uploader('Cargar archivo CSV local', type=['csv'])

    controls = st.columns([1, 1, 1.25])
    process_clicked = controls[0].button(
      'Procesar CSV',
      type='primary',
      disabled=uploaded_file is None,
      width='stretch',
    )
    reset_clicked = controls[1].button('Restaurar predeterminado', width='stretch')

    export_bytes = st.session_state.dataset_df.to_csv(index=False).encode('utf-8')
    controls[2].download_button(
      'Exportar limpio (.csv)',
      data=export_bytes,
      file_name='dataset_limpio.csv',
      mime='text/csv',
      width='stretch',
    )

  if process_clicked and uploaded_file is not None:
    try:
      parsed_df = parse_uploaded_csv(uploaded_file.getvalue())
      enriched_df = enrich_dataset(parsed_df)

      st.session_state.dataset_df = enriched_df
      st.session_state.dataset_name = uploaded_file.name
      st.session_state.source_note = f'Fuente de datos: archivo subido por usuario ({uploaded_file.name}).'
      st.session_state.data_error = ''
      _sync_selected_features()
      st.rerun()
    except ValueError as error:
      st.session_state.data_error = str(error)

  if reset_clicked:
    default_df, dataset_name, source_note, data_error = load_default_dataset()
    st.session_state.dataset_df = default_df
    st.session_state.dataset_name = dataset_name
    st.session_state.source_note = source_note
    st.session_state.data_error = data_error
    _sync_selected_features()
    st.rerun()

  if st.session_state.data_error:
    st.error(st.session_state.data_error)

  main_col, side_col = st.columns([2.15, 1], vertical_alignment='top')

  with main_col:
    with st.container(border=True):
      st.markdown('#### Matriz estadistica descriptiva univariada')
      descriptive_df = dashboard['descriptive_stats'].copy()
      if descriptive_df.empty:
        st.info('No hay variables numericas suficientes para construir la matriz descriptiva.')
      else:
        descriptive_df = descriptive_df.rename(
          columns={
            'variable': 'Metrica',
            'n': 'N',
            'media': 'Media',
            'mediana': 'Mediana',
            'desv_std': 'Desv. estandar',
            'min': 'Minimo',
            'max': 'Maximo',
          }
        )
        st.dataframe(descriptive_df, hide_index=True, width='stretch')

    with st.container(border=True):
      st.markdown('#### Co-movimiento estructural de series (2020-2024)')
      trend_df = _build_structural_series(dashboard['dataset_rows'])
      if trend_df.empty:
        st.info('No hay datos suficientes para trazar la serie estructural.')
      else:
        chart_source = trend_df[['x', 'precio_cierre']].copy()
        if 'volumen_normalizado' in trend_df.columns:
          chart_source['volumen_normalizado'] = trend_df['volumen_normalizado']
        st.line_chart(chart_source, x='x')

        st.caption(
          'La linea de volumen se normaliza al rango de precio para comparar ciclos y puntos de inflexion.'
        )

    with st.container(border=True):
      st.markdown('#### Inspeccion de registros muestrales')
      sample_df = _build_sample_table(dashboard['dataset_rows'], dashboard['visible_columns'])
      if sample_df.empty:
        st.info('No hay columnas disponibles para la inspeccion rapida.')
      else:
        st.dataframe(sample_df, hide_index=True, width='stretch')

    with st.expander('Estructura de columnas y muestra rapida'):
      st.dataframe(dashboard['column_types'], hide_index=True, width='stretch')

      sample_columns = dashboard['visible_columns'][:10]
      if sample_columns:
        st.dataframe(
          dashboard['dataset_rows'][sample_columns].head(10),
          hide_index=True,
          width='stretch',
        )

  with side_col:
    with st.container(border=True):
      st.markdown('#### Diagnostico de integridad')
      rows_ok = len(dashboard['dataset_rows']) >= MIN_REQUIRED_ROWS
      columns_ok = len(dashboard['visible_columns']) >= MIN_REQUIRED_COLUMNS

      total_cells = len(dashboard['dataset_rows']) * max(1, len(dashboard['visible_columns']))
      valid_cells = max(0, total_cells - int(dashboard['data_quality']['null_count']))
      structural_integrity = (valid_cells / total_cells * 100) if total_cells else 0

      st.metric('Integridad estructural', f'{structural_integrity:.1f}%')
      st.metric('Nulos', dashboard['data_quality']['null_count'])
      st.metric('Duplicados', dashboard['data_quality']['duplicate_rows'])

      _render_requirement(
        f'Registros: {len(dashboard["dataset_rows"])} (minimo {MIN_REQUIRED_ROWS})',
        rows_ok,
      )
      _render_requirement(
        f'Variables: {len(dashboard["visible_columns"])} (minimo {MIN_REQUIRED_COLUMNS})',
        columns_ok,
      )
      _render_requirement(f'Dimensiones actuales: {dashboard["dimensions_text"]}', True)

    with st.container(border=True):
      st.markdown('#### Pipeline de preprocesamiento')
      st.caption('Ajustes operativos para limpiar colas y estabilizar dispersion antes de K-Means.')

      st.selectbox(
        'Tecnica de normalizacion',
        options=['Z-score (estandar)', 'Min-max [0,1]', 'Robusta (IQR)'],
        index=0,
        key='dataset_normalization_mode',
      )
      st.toggle('Winsorizacion de colas (1%-99%)', value=True, key='dataset_winsorize_toggle')
      st.slider('Ventana de volatilidad historica', min_value=7, max_value=90, value=30, key='dataset_volatility_window')

    with st.container(border=True):
      st.markdown('#### Fundamento metodologico')
      st.write(
        'La estandarizacion multivariante (Z-score) mantiene comparabilidad entre precio, retorno y volumen '
        'antes de ejecutar segmentacion K-Means en espacio euclidiano.'
      )
      st.code('z = (x - media) / desv_std', language='text')

      if st.button('Proceder a mineria y segmentacion', type='primary', width='stretch'):
        st.session_state.active_screen = 'Mineria'
        st.rerun()
