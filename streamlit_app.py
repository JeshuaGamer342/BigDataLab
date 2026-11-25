from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from app.analytics import build_dashboard
from app.screens.dataset_screen import render_dataset_screen
from app.screens.exploration_screen import render_exploration_screen
from app.screens.mining_screen import render_mining_screen
from app.state import initialize_state


SCREENS = {
  'Dataset': {
    'nav_label': 'Dataset',
    'description': 'Ingesta, estructura y diagnostico de calidad del dataset.',
    'title': 'Gestion de dataset & diagnostico de calidad',
    'subtitle': 'Ingesta empirica de criptomonedas y pipeline de preprocesamiento con validacion de robustez.',
    'kicker': 'Dataset activo',
  },
  'Mineria': {
    'nav_label': 'Mineria',
    'description': 'Calibracion de K-Means y perfiles de regimenes segmentados.',
    'title': 'Modelado no supervisado: K-Means clustering',
    'subtitle': 'Calibracion multivariante de regimenes de mercado y validacion de clusters.',
    'kicker': 'Modelo activo',
  },
  'Exploracion': {
    'nav_label': 'Exploracion',
    'description': 'Validacion visual de regimenes y estadistica descriptiva del espacio empirico.',
    'title': 'Exploracion visual de regimenes de mercado',
    'subtitle': 'Analisis grafico multivariante para interpretar transiciones y dispersion de fases.',
    'kicker': 'Documento de trabajo',
  },
}


def _inject_styles() -> None:
  st.markdown(
    '''
    <style>
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f6f8ff 0%, #f9fbff 55%, #f6f9ff 100%);
        color: #0f172a;
      }

      [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.86);
        border-bottom: 1px solid #dbe5ff;
        backdrop-filter: blur(6px);
      }

      .main .block-container {
        max-width: 1460px;
        padding-top: 0.55rem;
        padding-bottom: 1.45rem;
      }

      .lab-topbar {
        align-items: center;
        background: #ffffff;
        border: 1px solid #dbe5ff;
        border-radius: 14px;
        display: flex;
        gap: 0.9rem;
        justify-content: space-between;
        margin-bottom: 0.6rem;
        padding: 0.65rem 0.85rem;
      }

      .lab-brand {
        align-items: center;
        color: #0f172a;
        display: inline-flex;
        font-weight: 700;
        gap: 0.45rem;
        letter-spacing: 0.01em;
      }

      .lab-brand-badge {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        border-radius: 6px;
        height: 1.08rem;
        position: relative;
        width: 1.08rem;
      }

      .lab-brand-badge::after {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 999px;
        content: "";
        height: 0.2rem;
        left: 0.43rem;
        position: absolute;
        top: 0.44rem;
        width: 0.2rem;
      }

      .lab-context {
        color: #334155;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }

      .lab-meta {
        color: #334155;
        font-size: 0.76rem;
        text-align: right;
      }

      .lab-chip {
        background: #ffffff;
        border: 1px solid #dbe5ff;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        min-height: 4.2rem;
        padding: 0.62rem 0.72rem;
      }

      .lab-chip-label {
        color: #64748b;
        display: block;
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
      }

      .lab-chip-value {
        color: #0f172a;
        display: block;
        font-size: 0.96rem;
        font-weight: 600;
        line-height: 1.2;
      }

      .screen-nav-label {
        color: #334155;
        font-size: 0.79rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 0.42rem;
        text-transform: uppercase;
      }

      .screen-description {
        color: #475569;
        font-size: 0.86rem;
        margin-top: 0.2rem;
      }

      .req-pill {
        border-radius: 10px;
        font-size: 0.79rem;
        font-weight: 600;
        margin: 0.22rem 0;
        padding: 0.34rem 0.48rem;
      }

      .req-pill.ok {
        background: rgba(34, 197, 94, 0.09);
        border: 1px solid rgba(22, 163, 74, 0.25);
        color: #166534;
      }

      .req-pill.alert {
        background: rgba(249, 115, 22, 0.09);
        border: 1px solid rgba(234, 88, 12, 0.25);
        color: #9a3412;
      }

      .profile-card {
        background: #ffffff;
        border: 1px solid #dbe5ff;
        border-left-width: 5px;
        border-radius: 12px;
        margin: 0.42rem 0;
        padding: 0.55rem 0.7rem;
      }

      .profile-positive {
        border-left-color: #16a34a;
      }

      .profile-negative {
        border-left-color: #dc2626;
      }

      .profile-stable {
        border-left-color: #2563eb;
      }

      .profile-neutral {
        border-left-color: #64748b;
      }

      .profile-header {
        align-items: baseline;
        display: flex;
        gap: 0.4rem;
      }

      .profile-id {
        color: #1d4ed8;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.02em;
      }

      .profile-name {
        color: #0f172a;
        font-size: 0.95rem;
        font-weight: 700;
        margin-right: auto;
      }

      .profile-share {
        color: #334155;
        font-size: 0.75rem;
        font-weight: 600;
      }

      .profile-grid {
        color: #334155;
        display: grid;
        font-size: 0.8rem;
        gap: 0.24rem 0.9rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin-top: 0.38rem;
      }

      .screen-banner {
        background: #ffffff;
        border: 1px solid #dbe5ff;
        border-radius: 14px;
        margin-bottom: 0.58rem;
        padding: 0.8rem 0.95rem;
      }

      .screen-kicker {
        color: #1d4ed8;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin: 0;
        text-transform: uppercase;
      }

      .screen-title {
        color: #0f172a;
        font-size: 2.08rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.08;
        margin: 0.24rem 0 0.26rem;
      }

      .screen-subtitle {
        color: #334155;
        font-size: 0.96rem;
        margin: 0;
      }

      .screen-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.34rem;
        margin-top: 0.52rem;
      }

      .screen-meta-pill {
        background: #f8faff;
        border: 1px solid #dbe5ff;
        border-radius: 999px;
        color: #334155;
        font-size: 0.73rem;
        font-weight: 600;
        padding: 0.2rem 0.56rem;
      }

      [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #dbe5ff;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
      }

      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid #dbe5ff;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        min-height: 6.2rem;
        padding: 0.6rem 0.72rem;
      }

      div[data-testid="stMetricLabel"] p {
        color: #475569;
        font-size: 0.76rem;
        font-weight: 700;
      }

      div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-weight: 800;
      }

      div[role="radiogroup"] {
        gap: 0.48rem;
      }

      div[role="radiogroup"] label {
        background: #f8faff;
        border: 1px solid #d5e2ff;
        border-radius: 999px;
        padding: 0.16rem 0.7rem;
      }

      div[role="radiogroup"] label p,
      div[role="radiogroup"] label span {
        color: #1e293b !important;
        font-size: 0.81rem;
        font-weight: 600;
      }

      div[role="radiogroup"] label:has(input:checked) {
        background: #1954e8;
        border-color: #1954e8;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.28);
      }

      div[role="radiogroup"] label:has(input:checked) p,
      div[role="radiogroup"] label:has(input:checked) span {
        color: #ffffff !important;
      }

      button[kind="primary"] {
        background: linear-gradient(120deg, #1752e6 0%, #1d4ed8 100%);
        border: none;
        color: #ffffff !important;
      }

      button[kind="secondary"] {
        background: #ffffff;
        border: 1px solid #cddbf9;
        color: #0f172a !important;
      }

      div[data-testid="stAlert"] [data-testid^="stAlertContent"] {
        background: #f8faff !important;
        border: 1px solid #dbe5ff !important;
        color: #0f172a !important;
      }

      [data-testid="stFileUploaderDropzone"] {
        background: #f8fbff;
        border: 1px dashed #b8ccf8;
        border-radius: 12px;
      }

      [data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
      }

      .lab-footer {
        color: #64748b;
        font-size: 0.74rem;
        letter-spacing: 0.03em;
        margin-top: 0.7rem;
        text-align: center;
      }
    </style>
    ''',
    unsafe_allow_html=True,
  )


def _render_chip(column, label: str, value: str) -> None:
  column.markdown(
    f'''
    <div class="lab-chip">
      <span class="lab-chip-label">{label}</span>
      <span class="lab-chip-value">{value}</span>
    </div>
    ''',
    unsafe_allow_html=True,
  )


def _date_range_text(dataset_df: pd.DataFrame) -> str:
  if 'date' not in dataset_df.columns:
    return 'Sin rango temporal'

  parsed_dates = pd.to_datetime(dataset_df['date'], errors='coerce', dayfirst=True).dropna()
  if parsed_dates.empty:
    return 'Sin rango temporal'

  return f'{parsed_dates.min().date()} - {parsed_dates.max().date()}'


def _build_header_metrics(active_screen: str, dashboard: dict) -> list[tuple[str, str]]:
  data_quality = dashboard['data_quality']
  clustering_result = dashboard['clustering_result']

  if active_screen == 'Dataset':
    total_cells = len(dashboard['dataset_rows']) * max(1, len(dashboard['visible_columns']))
    valid_cells = max(0, total_cells - int(data_quality['null_count']))
    integrity = (valid_cells / total_cells * 100) if total_cells else 0
    return [
      ('Filas muestreadas', f'{len(dashboard["dataset_rows"]):,}'),
      ('Variables analiticas', f'{len(dashboard["visible_columns"])}'),
      ('Diagnostico integridad', f'{integrity:.1f}%'),
      ('Outliers (Tukey)', f"{int(clustering_result['anomaly_count'])}"),
    ]

  if active_screen == 'Mineria':
    silhouette = clustering_result.get('silhouette_score')
    inertia = clustering_result.get('inertia')
    return [
      ('Registros modelados', f"{len(clustering_result['modeled_rows']):,}"),
      ('Coeficiente de silueta', 'N/A' if silhouette is None else f'{silhouette:.2f}'),
      ('Inercia WCSS', 'N/A' if inertia is None else f'{inertia:,.1f}'),
      ('Anomalias detectadas', f"{int(clustering_result['anomaly_count'])}"),
    ]

  return [
    ('Serie base', dashboard['trend_feature'] or 'N/A'),
    ('Bins histograma', f"{len(dashboard['histogram'])}"),
    ('Puntos dispersion', f"{len(dashboard['scatter'])}"),
    ('Clusters activos', f"{len(clustering_result['cluster_counts'])}"),
  ]


def _render_top_bar(dashboard: dict) -> None:
  st.markdown(
    '''
    <section class="lab-topbar">
      <div class="lab-brand">
        <span class="lab-brand-badge"></span>
        <span>BitData Lab</span>
      </div>
      <div class="lab-context">2024-II · Finanzas computacionales · K-Means regimes</div>
      <div class="lab-meta">Investigacion aplicada · Cripto x 24/7</div>
    </section>
    ''',
    unsafe_allow_html=True,
  )

  chip_cols = st.columns(6)
  _render_chip(chip_cols[0], 'Dataset activo', st.session_state.dataset_name)
  _render_chip(chip_cols[1], 'Fecha de corrida', datetime.now().strftime('%d/%m/%Y'))
  _render_chip(chip_cols[2], 'Ventana temporal', _date_range_text(dashboard['dataset_rows']))
  _render_chip(chip_cols[3], 'Variables k-means', f"{len(st.session_state.selected_features)} seleccionadas")
  _render_chip(chip_cols[4], 'Clusters activos', str(st.session_state.cluster_count))
  _render_chip(chip_cols[5], 'Umbral de anomalia', f'{st.session_state.anomaly_threshold:.1f} sigma')


def _render_navigation() -> str:
  screen_names = list(SCREENS.keys())
  current_screen = st.session_state.active_screen if st.session_state.active_screen in SCREENS else screen_names[0]
  current_index = screen_names.index(current_screen)

  with st.container(border=True):
    st.markdown('<p class="screen-nav-label">Navegacion principal</p>', unsafe_allow_html=True)

    selected_screen = st.radio(
      'Pantallas',
      options=screen_names,
      index=current_index,
      format_func=lambda screen_name: SCREENS[screen_name]['nav_label'],
      horizontal=True,
      label_visibility='collapsed',
    )

    st.markdown(
      f'<div class="screen-description">{SCREENS[selected_screen]["description"]}</div>',
      unsafe_allow_html=True,
    )

  st.session_state.active_screen = selected_screen
  return selected_screen


def _render_screen_header(active_screen: str, dashboard: dict) -> None:
  screen_data = SCREENS[active_screen]
  date_range = _date_range_text(dashboard['dataset_rows'])

  st.markdown(
    f'''
    <section class="screen-banner">
      <p class="screen-kicker">{screen_data['kicker']}</p>
      <h1 class="screen-title">{screen_data['title']}</h1>
      <p class="screen-subtitle">{screen_data['subtitle']}</p>
      <div class="screen-meta-row">
        <span class="screen-meta-pill">Rango: {date_range}</span>
        <span class="screen-meta-pill">Muestras: {len(dashboard['dataset_rows']):,}</span>
        <span class="screen-meta-pill">Frecuencia: segun dataset</span>
      </div>
    </section>
    ''',
    unsafe_allow_html=True,
  )

  metric_cols = st.columns(4)
  for index, (label, value) in enumerate(_build_header_metrics(active_screen, dashboard)):
    metric_cols[index].metric(label, value)


def _render_footer() -> None:
  st.markdown(
    '<div class="lab-footer">BitData Lab · Semestre 2024-II · Laboratorio de criptoactivos y modelacion estadistica</div>',
    unsafe_allow_html=True,
  )


def main() -> None:
  st.set_page_config(page_title='BitData Lab · Crypto regimes', page_icon='📊', layout='wide')
  _inject_styles()
  initialize_state()

  dashboard = build_dashboard(
    st.session_state.dataset_df,
    st.session_state.selected_features,
    st.session_state.cluster_count,
    st.session_state.anomaly_threshold,
  )

  if dashboard['selected_features'] != st.session_state.selected_features:
    st.session_state.selected_features = dashboard['selected_features']
    st.rerun()

  _render_top_bar(dashboard)
  active_screen = _render_navigation()
  _render_screen_header(active_screen, dashboard)

  if active_screen == 'Dataset':
    render_dataset_screen(dashboard)
  elif active_screen == 'Mineria':
    render_mining_screen(dashboard)
  else:
    render_exploration_screen(dashboard)

  _render_footer()


if __name__ == '__main__':
  main()
