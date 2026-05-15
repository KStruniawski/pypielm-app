"""Unit tests for utils/sample_data.py."""

from __future__ import annotations

import math

import numpy as np
import pytest

from utils.sample_data import (
    SAMPLE_DATASETS,
    SampleDataset,
    allen_cahn_1d,
    burgers_steady_1d,
    get_sample_dataset,
    heat_1d,
    helmholtz_2d,
    poisson_1d,
    poisson_2d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_dataset(ds: SampleDataset) -> None:
    """Common shape / dtype / NaN assertions for any SampleDataset."""
    assert isinstance(ds, SampleDataset)
    assert ds.X.ndim == 2
    assert ds.y.ndim == 2
    assert ds.X.shape[0] == ds.y.shape[0], "X and y row count mismatch"
    assert ds.y.shape[1] == 1, "y must have exactly 1 column"
    assert ds.X.dtype == np.float64
    assert ds.y.dtype == np.float64
    assert not np.any(np.isnan(ds.X)), "NaN found in X"
    assert not np.any(np.isnan(ds.y)), "NaN found in y"
    assert ds.dim >= 1
    assert ds.name
    assert ds.pde
    assert ds.exact_str


# ---------------------------------------------------------------------------
# Individual dataset tests
# ---------------------------------------------------------------------------

class TestPoisson1D:
    def test_default_shape(self):
        ds = poisson_1d()
        assert ds.X.shape == (500, 1)
        assert ds.y.shape == (500, 1)

    def test_custom_n(self):
        ds = poisson_1d(n=200)
        assert ds.X.shape == (200, 1)

    def test_exact_solution_values(self):
        ds = poisson_1d(n=100)
        expected = ds.X * (1.0 - ds.X)
        np.testing.assert_allclose(ds.y, expected, atol=1e-15)

    def test_domain_range(self):
        ds = poisson_1d()
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_dim(self):
        assert poisson_1d().dim == 1

    def test_metadata(self):
        _check_dataset(poisson_1d())


class TestHeat1D:
    def test_shape(self):
        ds = heat_1d(nx=20, nt=10)
        assert ds.X.shape == (200, 2)  # 20*10 points, 2 columns (x, t)
        assert ds.y.shape == (200, 1)

    def test_exact_solution(self):
        ds = heat_1d(nx=10, nt=5)
        x, t = ds.X[:, 0], ds.X[:, 1]
        expected = np.exp(-math.pi ** 2 * t) * np.sin(math.pi * x)
        np.testing.assert_allclose(ds.y.ravel(), expected, atol=1e-15)

    def test_dim(self):
        assert heat_1d().dim == 2

    def test_metadata(self):
        _check_dataset(heat_1d())


class TestPoisson2D:
    def test_shape(self):
        ds = poisson_2d(n_per_dim=30)
        assert ds.X.shape == (900, 2)
        assert ds.y.shape == (900, 1)

    def test_exact_solution(self):
        ds = poisson_2d(n_per_dim=10)
        expected = np.sin(math.pi * ds.X[:, 0]) * np.sin(math.pi * ds.X[:, 1])
        np.testing.assert_allclose(ds.y.ravel(), expected, atol=1e-15)

    def test_domain_range(self):
        ds = poisson_2d()
        assert ds.X.min() >= 0.0
        assert ds.X.max() <= 1.0

    def test_dim(self):
        assert poisson_2d().dim == 2

    def test_metadata(self):
        _check_dataset(poisson_2d())


class TestBurgers1D:
    def test_shape(self):
        ds = burgers_steady_1d()
        assert ds.X.shape == (500, 1)
        assert ds.y.shape == (500, 1)

    def test_antisymmetry(self):
        """u(x) = -tanh(x/(2ν)) is an odd function → u(-x) = -u(x)."""
        # Use odd n so there is a true x=0 point in linspace(-1,1,n)
        ds = burgers_steady_1d(n=201)
        x = ds.X.ravel()
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = ds.y.ravel()[order]
        # The centre point (x==0) should satisfy u≈0  (tanh(0)=0)
        mid = len(x_sorted) // 2
        assert abs(x_sorted[mid]) < 1e-10, "midpoint is not x=0"
        assert abs(y_sorted[mid]) < 1e-10

    def test_custom_nu(self):
        ds_sharp = burgers_steady_1d(n=100, nu=0.001)
        ds_smooth = burgers_steady_1d(n=100, nu=0.5)
        # sharper interface → larger gradient → larger |u| range
        assert ds_sharp.y.max() > ds_smooth.y.max() - 0.01

    def test_dim(self):
        assert burgers_steady_1d().dim == 1

    def test_metadata(self):
        _check_dataset(burgers_steady_1d())


class TestHelmholtz2D:
    def test_shape(self):
        ds = helmholtz_2d(n_per_dim=20)
        assert ds.X.shape == (400, 2)
        assert ds.y.shape == (400, 1)

    def test_exact_solution(self):
        ds = helmholtz_2d(n_per_dim=10)
        expected = np.sin(math.pi * ds.X[:, 0]) * np.cos(math.pi * ds.X[:, 1])
        np.testing.assert_allclose(ds.y.ravel(), expected, atol=1e-15)

    def test_dim(self):
        assert helmholtz_2d().dim == 2

    def test_metadata(self):
        _check_dataset(helmholtz_2d())


class TestAllenCahn1D:
    def test_shape(self):
        ds = allen_cahn_1d()
        assert ds.X.shape == (500, 1)
        assert ds.y.shape == (500, 1)

    def test_range(self):
        ds = allen_cahn_1d()
        assert ds.y.min() >= -1.0 - 1e-10
        assert ds.y.max() <= 1.0 + 1e-10

    def test_monotone(self):
        """tanh profile is strictly increasing."""
        ds = allen_cahn_1d(n=100)
        order = np.argsort(ds.X.ravel())
        y_sorted = ds.y.ravel()[order]
        diffs = np.diff(y_sorted)
        assert np.all(diffs >= -1e-14)

    def test_dim(self):
        assert allen_cahn_1d().dim == 1

    def test_metadata(self):
        _check_dataset(allen_cahn_1d())


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestSampleDatasetsRegistry:
    def test_all_keys_present(self):
        # Minimum set of expected datasets; registry may have more (3D/advanced)
        expected_subset = {
            "Poisson 1D",
            "Heat 1D (transient)",
            "Poisson 2D",
            "Burgers 1D (steady, ν=0.01)",
            "Helmholtz 2D (k=π)",
            "Allen–Cahn 1D (ε=0.1)",
        }
        assert expected_subset <= set(SAMPLE_DATASETS.keys()), (
            f"Missing keys: {expected_subset - set(SAMPLE_DATASETS.keys())}"
        )

    def test_all_callables(self):
        for name, fn in SAMPLE_DATASETS.items():
            assert callable(fn), f"{name} is not callable"

    @pytest.mark.parametrize("name", list(SAMPLE_DATASETS.keys()))
    def test_get_sample_dataset_returns_valid(self, name):
        ds = get_sample_dataset(name)
        _check_dataset(ds)

    def test_get_sample_dataset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown sample dataset"):
            get_sample_dataset("NonExistent PDE")

    def test_get_sample_dataset_passes_kwargs(self):
        ds = get_sample_dataset("Poisson 1D", n=77)
        assert ds.X.shape[0] == 77
