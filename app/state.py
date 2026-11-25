from __future__ import annotations

import streamlit as st

from app.analytics import get_numeric_columns, get_visible_columns, pick_default_features
from app.constants import (
  INITIAL_ANOMALY_THRESHOLD,
  INITIAL_CLUSTER_COUNT,
)
from app.data import load_default_dataset


def initialize_state() -> None:
  if 'dataset_df' not in st.session_state:
    default_df, dataset_name, source_note, data_error = load_default_dataset()
    st.session_state.dataset_df = default_df
    st.session_state.dataset_name = dataset_name
    st.session_state.source_note = source_note
    st.session_state.data_error = data_error

    visible_columns = get_visible_columns(default_df)
    numeric_columns = get_numeric_columns(default_df, visible_columns)
    st.session_state.selected_features = pick_default_features(numeric_columns)

  if 'cluster_count' not in st.session_state:
    st.session_state.cluster_count = INITIAL_CLUSTER_COUNT

  if 'anomaly_threshold' not in st.session_state:
    st.session_state.anomaly_threshold = INITIAL_ANOMALY_THRESHOLD

  if 'data_error' not in st.session_state:
    st.session_state.data_error = ''

  if 'active_screen' not in st.session_state:
    st.session_state.active_screen = 'Dataset'
