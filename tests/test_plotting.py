"""Unit tests for utils/plotting.py.

Uses a headless matplotlib backend (set in conftest via sys.path) and
verifies that every plot function returns a Figure with the expected number
of panels, correct axis labels, and no exceptions on typical inputs.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from utils.plotting import (
    plot_comparison_bar,
    plot_metrics_bar,
    plot_solution_1d,
    plot_solution_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close(fig: plt.Figure) -> None:
    plt.close(fig)


@pytest.fixture()
def x1():
    return np.linspace(0.0, 1.0, 50)


@pytest.fixture()
def u1(x1):
    return np.sin(math.pi * x1)


@pytest.fixture()
def u1_ref(x1):
    return np.sin(math.pi * x1) + 0.01 * np.random.default_rng(0).standard_normal(len(x1))


@pytest.fixture()
def X2():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, size=(100, 2))


@pytest.fixture()
def u2(X2):
    return (np.sin(math.pi * X2[:, 0]) * np.sin(math.pi * X2[:, 1])).reshape(-1, 1)


# ---------------------------------------------------------------------------
# plot_solution_1d
# ---------------------------------------------------------------------------

class TestPlotSolution1D:
    def test_returns_figure(self, x1, u1):
        fig = plot_solution_1d(x1, u1)
        assert isinstance(fig, plt.Figure)
        _close(fig)

    def test_single_panel_without_ref(self, x1, u1):
        fig = plot_solution_1d(x1, u1)
        assert len(fig.axes) == 1
        _close(fig)

    def test_two_panels_with_ref(self, x1, u1, u1_ref):
        fig = plot_solution_1d(x1, u1, u1_ref)
        assert len(fig.axes) == 2
        _close(fig)

    def test_custom_title_applied(self, x1, u1):
        fig = plot_solution_1d(x1, u1, title="My PDE")
        # suptitle is stored in the figure's _suptitle attribute
        assert fig._suptitle.get_text() == "My PDE"
        _close(fig)

    def test_xlabel_ylabel(self, x1, u1):
        fig = plot_solution_1d(x1, u1, xlabel="time", ylabel="pressure")
        ax = fig.axes[0]
        assert ax.get_xlabel() == "time"
        assert ax.get_ylabel() == "pressure"
        _close(fig)

    def test_no_nan_in_data(self, x1, u1):
        fig = plot_solution_1d(x1, u1)
        for ax in fig.axes:
            for line in ax.get_lines():
                assert not np.any(np.isnan(line.get_ydata()))
        _close(fig)


# ---------------------------------------------------------------------------
# plot_solution_2d
# ---------------------------------------------------------------------------

class TestPlotSolution2D:
    def test_returns_figure(self, X2, u2):
        fig = plot_solution_2d(X2, u2)
        assert isinstance(fig, plt.Figure)
        _close(fig)

    def test_single_panel_no_ref(self, X2, u2):
        fig = plot_solution_2d(X2, u2)
        assert len(fig.axes) >= 1
        _close(fig)

    def test_three_panels_with_ref(self, X2, u2):
        u_ref = u2 + 0.01
        fig = plot_solution_2d(X2, u2, u_ref)
        # 3 scatter axes + 3 colorbar axes = 6 axes total
        assert len(fig.axes) >= 3
        _close(fig)

    def test_custom_title(self, X2, u2):
        fig = plot_solution_2d(X2, u2, title="Wave field")
        assert fig._suptitle.get_text() == "Wave field"
        _close(fig)


# ---------------------------------------------------------------------------
# plot_metrics_bar
# ---------------------------------------------------------------------------

class TestPlotMetricsBar:
    def test_returns_figure(self):
        metrics = {"rmse": 0.01, "relative_l2": 0.05, "r2": 0.98}
        fig = plot_metrics_bar(metrics)
        assert isinstance(fig, plt.Figure)
        _close(fig)

    def test_bar_count(self):
        metrics = {"a": 1.0, "b": 2.0, "c": 3.0}
        fig = plot_metrics_bar(metrics)
        ax = fig.axes[0]
        assert len(ax.patches) == 3
        _close(fig)

    def test_single_metric(self):
        fig = plot_metrics_bar({"rmse": 0.1})
        assert isinstance(fig, plt.Figure)
        _close(fig)

    def test_empty_metrics_does_not_raise(self):
        # empty dict → zero bars, but should not crash
        fig = plot_metrics_bar({})
        assert isinstance(fig, plt.Figure)
        _close(fig)


# ---------------------------------------------------------------------------
# plot_comparison_bar
# ---------------------------------------------------------------------------

class TestPlotComparisonBar:
    @pytest.fixture()
    def model_data(self):
        names  = ["ModelA", "ModelB", "ModelC"]
        values = [0.30, 0.10, 0.20]
        return names, values

    def test_returns_figure(self, model_data):
        names, values = model_data
        fig = plot_comparison_bar(names, values)
        assert isinstance(fig, plt.Figure)
        _close(fig)

    def test_bar_count_equals_models(self, model_data):
        names, values = model_data
        fig = plot_comparison_bar(names, values)
        ax = fig.axes[0]
        assert len(ax.patches) == len(names)
        _close(fig)

    def test_sorted_ascending(self, model_data):
        names, values = model_data
        fig = plot_comparison_bar(names, values, lower_is_better=True)
        ax = fig.axes[0]
        bar_widths = [p.get_width() for p in ax.patches]
        assert bar_widths == sorted(bar_widths), "Bars should be sorted ascending"
        _close(fig)

    def test_higher_is_better_sorting(self, model_data):
        names, values = model_data
        fig = plot_comparison_bar(names, values, lower_is_better=False)
        ax = fig.axes[0]
        bar_widths = [p.get_width() for p in ax.patches]
        assert bar_widths == sorted(bar_widths, reverse=True), "Bars should be sorted descending"
        _close(fig)

    def test_custom_metric_name_in_title(self, model_data):
        names, values = model_data
        fig = plot_comparison_bar(names, values, metric_name="relative_l2")
        ax = fig.axes[0]
        assert "relative_l2" in ax.get_title()
        _close(fig)

    def test_single_model(self):
        fig = plot_comparison_bar(["OnlyModel"], [0.42])
        assert isinstance(fig, plt.Figure)
        _close(fig)
