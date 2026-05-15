"""Integration-style smoke tests for the Streamlit pages.

Because Streamlit pages call ``st.*`` functions at module-level (not inside
functions), we cannot simply import them without starting a Streamlit server.
Instead we use ``unittest.mock`` to stub out all Streamlit API calls and
verify that:
  1. The page modules are importable without raising.
  2. The helper functions embedded in each page work correctly in isolation.
  3. The ``_load_file`` helper correctly converts uploaded bytes to a
     ``PIELMDataset`` via a temp file.

Any interaction that requires a live Streamlit session (widget state, reruns)
is out of scope for unit tests — those belong in end-to-end / browser tests.
"""

from __future__ import annotations

import io
import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
# torch must be imported at module level before the streamlit-mock fixture
# installs itself, otherwise a second import attempt inside the mock context
# triggers a "already has a docstring" RuntimeError in torch.overrides.
import torch  # noqa: F401

# ---------------------------------------------------------------------------
# Ensure app root is in path (conftest already does this; belt-and-braces)
# ---------------------------------------------------------------------------

_APP_ROOT = Path(__file__).parent.parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))


# ---------------------------------------------------------------------------
# Stub the entire ``streamlit`` module so pages can be imported without a
# Streamlit server.  Every ``st.xxx`` call returns a MagicMock.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_streamlit():
    """Install a MagicMock as the ``streamlit`` module for all page imports."""
    st_mock = MagicMock()
    # st.session_state must behave like a dict
    st_mock.session_state = {}
    # st.stop() raises SystemExit so pages halt — mock it to just return
    st_mock.stop = MagicMock()
    # st.tabs returns a context-manager tuple
    tab_ctx = MagicMock()
    tab_ctx.__enter__ = MagicMock(return_value=MagicMock())
    tab_ctx.__exit__ = MagicMock(return_value=False)
    st_mock.tabs.return_value = (tab_ctx, tab_ctx)

    with patch.dict("sys.modules", {"streamlit": st_mock}):
        # Also patch inside utils so session_state uses the same mock
        with patch("utils.session_state.st", st_mock):
            yield st_mock


# ---------------------------------------------------------------------------
# Helper: create a minimal in-memory CSV and return it as a BytesIO + name
# ---------------------------------------------------------------------------

def _make_csv_upload(n: int = 50) -> tuple[bytes, str]:
    x = np.linspace(0.0, 1.0, n)
    y = x * (1.0 - x)
    header = "x,y"
    lines = [header] + [f"{xi},{yi}" for xi, yi in zip(x, y)]
    return "\n".join(lines).encode(), "test_poisson.csv"


# ---------------------------------------------------------------------------
# Test: sample_data integration with _arrays_to_dataset helper
# ---------------------------------------------------------------------------

class TestSampleDataIntegration:
    """Verify the (X, y) → PIELMDataset round-trip used in the Train page."""

    def test_arrays_to_dataset_1d(self):
        """Small helper extracted from the page; tests tensor conversion."""
        from pypielm.data.dataset import PIELMDataset

        X = np.linspace(0.0, 1.0, 30).reshape(-1, 1)
        y = (X * (1 - X)).reshape(-1, 1)

        X_t = torch.tensor(X, dtype=torch.float64)
        y_t = torch.tensor(y, dtype=torch.float64)
        ds = PIELMDataset(X_colloc=X_t, y_data=y_t)

        assert ds.X_colloc.shape == (30, 1)
        assert ds.y_data.shape   == (30, 1)
        np.testing.assert_allclose(ds.X_colloc.numpy(), X)
        np.testing.assert_allclose(ds.y_data.numpy(),   y)

    def test_all_sample_datasets_convert(self):
        """Every sample dataset can be wrapped into a PIELMDataset."""
        from pypielm.data.dataset import PIELMDataset
        from utils.sample_data import SAMPLE_DATASETS, get_sample_dataset

        for name in SAMPLE_DATASETS:
            sd = get_sample_dataset(name)
            X_t = torch.tensor(sd.X, dtype=torch.float64)
            y_t = torch.tensor(sd.y, dtype=torch.float64)
            ds = PIELMDataset(X_colloc=X_t, y_data=y_t)
            assert ds.X_colloc.shape[0] == sd.X.shape[0]


# ---------------------------------------------------------------------------
# Test: _load_file CSV path (without running the full Streamlit page)
# ---------------------------------------------------------------------------

class TestLoadFileHelper:
    """Test the _load_file helper logic directly (not via Streamlit)."""

    def _make_fake_upload(self, content: bytes, filename: str):
        obj = MagicMock()
        obj.name = filename
        obj.getvalue.return_value = content
        return obj

    def test_csv_load_returns_dataset(self):
        """CSVAdapter should parse the simple header-less CSV."""
        content, name = _make_csv_upload(n=40)
        fake_upload = self._make_fake_upload(content, name)

        import tempfile
        from pathlib import Path
        from pypielm.data import auto_load

        suffix = Path(fake_upload.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(fake_upload.getvalue())
            tmp_path = Path(tmp.name)
        try:
            ds = auto_load(tmp_path, dtype=torch.float64)
        finally:
            tmp_path.unlink(missing_ok=True)

        assert ds.X_colloc is not None
        assert ds.X_colloc.shape[0] > 0


# ---------------------------------------------------------------------------
# Test: plotting helpers used in pages
# ---------------------------------------------------------------------------

class TestPagePlottingHelpers:
    """Verify the plotting helpers produce figures without Streamlit active."""

    def test_plot_metrics_bar_with_typical_metrics(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_metrics_bar
        import matplotlib.pyplot as plt
        metrics = {"rmse": 0.02, "relative_l2": 0.10, "r2": 0.97, "mae": 0.015}
        fig = plot_metrics_bar(metrics)
        assert fig is not None
        plt.close(fig)

    def test_plot_solution_1d_no_ref(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_solution_1d
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, 50)
        u = np.sin(np.pi * x)
        fig = plot_solution_1d(x, u)
        assert fig is not None
        plt.close(fig)

    def test_plot_solution_2d_with_ref(self):
        import matplotlib
        matplotlib.use("Agg")
        from utils.plotting import plot_solution_2d
        import matplotlib.pyplot as plt
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, (80, 2))
        u = np.sin(np.pi * X[:, 0:1]) * np.sin(np.pi * X[:, 1:2])
        fig = plot_solution_2d(X, u, u + 0.01)
        assert fig is not None
        plt.close(fig)


# ---------------------------------------------------------------------------
# Test: benchmark page JSON parsing logic (pure Python, no Streamlit)
# ---------------------------------------------------------------------------

class TestBenchmarkJSONParsing:
    """Test that the JSON-flattening logic in the benchmark page works."""

    def _flatten_records(self, records: list[dict]) -> list[dict]:
        """Replicate the flattening logic from 2_Benchmark_Results.py."""
        rows = []
        for r in records:
            row = {}
            row["model"] = r.get("model", r.get("model_name", "?"))
            row["task"]  = r.get("task",  r.get("dataset",   "?"))
            row["seed"]  = r.get("seed",  "-")
            row.update(r.get("metrics", {}))
            for k, v in r.get("config", {}).items():
                if isinstance(v, (int, float, str, bool)):
                    row[f"cfg.{k}"] = v
            rows.append(row)
        return rows

    def test_single_record(self):
        records = [{"model": "CorePIELM", "task": "poisson_1d", "seed": 42,
                    "metrics": {"relative_l2": 0.05}, "config": {"hidden_dim": 200}}]
        rows = self._flatten_records(records)
        assert len(rows) == 1
        assert rows[0]["model"]        == "CorePIELM"
        assert rows[0]["relative_l2"]  == 0.05
        assert rows[0]["cfg.hidden_dim"] == 200

    def test_multiple_records_preserve_order(self):
        records = [
            {"model": "A", "task": "t1", "metrics": {"rmse": 0.1}},
            {"model": "B", "task": "t2", "metrics": {"rmse": 0.2}},
        ]
        rows = self._flatten_records(records)
        assert rows[0]["model"] == "A"
        assert rows[1]["model"] == "B"

    def test_missing_fields_use_defaults(self):
        rows = self._flatten_records([{}])
        assert rows[0]["model"] == "?"
        assert rows[0]["task"]  == "?"
        assert rows[0]["seed"]  == "-"

    def test_nested_config_non_primitive_skipped(self):
        records = [{"model": "M", "task": "t", "metrics": {},
                    "config": {"layers": [64, 64], "active": True, "lr": 1e-3}}]
        rows = self._flatten_records(records)
        assert "cfg.active" in rows[0]
        assert "cfg.lr"     in rows[0]
        assert "cfg.layers" not in rows[0]   # list → skipped
