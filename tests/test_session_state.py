"""Unit tests for utils/session_state.py.

Streamlit's ``st.session_state`` is a ``SessionStateProxy`` that behaves like
a dict. We replace it with a plain ``dict`` subclass during tests so no
Streamlit server is needed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixture: mock st.session_state with an ordinary dict
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_session_state():
    """Replace st.session_state with a plain dict for every test."""
    state: dict = {}

    # session_state.py does  `import streamlit as st`  then accesses
    # `st.session_state`.  We patch the `streamlit` module attribute so that
    # all calls in the module under test use our plain dict.
    import types
    import streamlit as _real_st  # noqa: F401 — imported to ensure module exists

    fake_st = types.SimpleNamespace(session_state=state)
    with patch("utils.session_state.st", fake_st):
        yield state


# ---------------------------------------------------------------------------
# Import the module AFTER patching (import-time code runs on first import;
# subsequent imports use the cached module, so patch must be in place first).
# ---------------------------------------------------------------------------

import utils.session_state as ss  # noqa: E402 — after sys.path is set by conftest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSessionStateGettersDefaultsNone:
    def test_model_default_none(self, mock_session_state):
        assert ss.get_model() is None

    def test_dataset_default_none(self, mock_session_state):
        assert ss.get_dataset() is None

    def test_config_default_empty_dict(self, mock_session_state):
        assert ss.get_config() == {}

    def test_metrics_default_empty_dict(self, mock_session_state):
        assert ss.get_metrics() == {}

    def test_source_name_default_unknown(self, mock_session_state):
        assert ss.get_source_name() == "unknown"


class TestSessionStateSetAndGet:
    def test_set_and_get_model(self, mock_session_state):
        sentinel = object()
        ss.set_model(sentinel)
        assert ss.get_model() is sentinel

    def test_set_and_get_dataset(self, mock_session_state):
        sentinel = object()
        ss.set_dataset(sentinel)
        assert ss.get_dataset() is sentinel

    def test_set_and_get_config(self, mock_session_state):
        cfg = {"hidden_dim": 100, "activation": "tanh"}
        ss.set_config(cfg)
        assert ss.get_config() == cfg

    def test_set_and_get_metrics(self, mock_session_state):
        m = {"rmse": 0.01, "r2": 0.99}
        ss.set_metrics(m)
        assert ss.get_metrics() == m

    def test_set_and_get_source_name(self, mock_session_state):
        ss.set_source_name("poisson_1d.csv")
        assert ss.get_source_name() == "poisson_1d.csv"

    def test_overwrite_model(self, mock_session_state):
        ss.set_model("first")
        ss.set_model("second")
        assert ss.get_model() == "second"

    def test_overwrite_config(self, mock_session_state):
        ss.set_config({"a": 1})
        ss.set_config({"b": 2})
        assert ss.get_config() == {"b": 2}


class TestSessionStateClear:
    def test_clear_removes_model(self, mock_session_state):
        ss.set_model("x")
        ss.clear_model()
        assert ss.get_model() is None

    def test_clear_removes_dataset(self, mock_session_state):
        ss.set_dataset("ds")
        ss.clear_model()
        assert ss.get_dataset() is None

    def test_clear_removes_config(self, mock_session_state):
        ss.set_config({"a": 1})
        ss.clear_model()
        assert ss.get_config() == {}

    def test_clear_removes_metrics(self, mock_session_state):
        ss.set_metrics({"rmse": 0.1})
        ss.clear_model()
        assert ss.get_metrics() == {}

    def test_clear_removes_source_name(self, mock_session_state):
        ss.set_source_name("file.csv")
        ss.clear_model()
        assert ss.get_source_name() == "unknown"

    def test_clear_is_idempotent(self, mock_session_state):
        """Calling clear on an already-empty state should not raise."""
        ss.clear_model()
        ss.clear_model()
        assert ss.get_model() is None

    def test_clear_does_not_affect_unrelated_keys(self, mock_session_state):
        """Keys outside the module's namespace must survive a clear."""
        mock_session_state["__my_custom_key"] = "keep_me"
        ss.set_model("x")
        ss.clear_model()
        assert mock_session_state.get("__my_custom_key") == "keep_me"
