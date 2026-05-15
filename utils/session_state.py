"""Session-state helpers for the PyPIELM Streamlit app.

Provides typed accessors for the shared session state so every page works
with the same keys without hardcoding string literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Keys (single source of truth)
# ---------------------------------------------------------------------------

_KEY_MODEL        = "trained_model"
_KEY_DATASET      = "dataset"
_KEY_CONFIG       = "model_config"
_KEY_METRICS      = "last_metrics"
_KEY_SOURCE_NAME  = "source_filename"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model() -> Any | None:
    return st.session_state.get(_KEY_MODEL)


def set_model(model: Any) -> None:
    st.session_state[_KEY_MODEL] = model


def get_dataset() -> Any | None:
    return st.session_state.get(_KEY_DATASET)


def set_dataset(ds: Any) -> None:
    st.session_state[_KEY_DATASET] = ds


def get_config() -> dict:
    return st.session_state.get(_KEY_CONFIG, {})


def set_config(cfg: dict) -> None:
    st.session_state[_KEY_CONFIG] = cfg


def get_metrics() -> dict:
    return st.session_state.get(_KEY_METRICS, {})


def set_metrics(m: dict) -> None:
    st.session_state[_KEY_METRICS] = m


def get_source_name() -> str:
    return st.session_state.get(_KEY_SOURCE_NAME, "unknown")


def set_source_name(name: str) -> None:
    st.session_state[_KEY_SOURCE_NAME] = name


def clear_model() -> None:
    for k in (_KEY_MODEL, _KEY_DATASET, _KEY_CONFIG, _KEY_METRICS, _KEY_SOURCE_NAME):
        st.session_state.pop(k, None)
