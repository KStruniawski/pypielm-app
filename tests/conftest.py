"""Pytest configuration for PyPIELM-App tests.

Sets up the Python path so ``utils`` can be imported without installing the
package, and provides shared fixtures (numpy arrays, tiny PIELMDatasets, etc.).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup — make the app root importable
# ---------------------------------------------------------------------------

_APP_ROOT = Path(__file__).parent.parent   # PyPIELM-App/
sys.path.insert(0, str(_APP_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def x_1d() -> np.ndarray:
    return np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(-1, 1)


@pytest.fixture()
def x_2d(rng) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=(200, 2))


@pytest.fixture()
def y_1d(x_1d) -> np.ndarray:
    return (x_1d * (1.0 - x_1d)).reshape(-1, 1)


@pytest.fixture()
def y_2d(x_2d) -> np.ndarray:
    import math
    return (np.sin(math.pi * x_2d[:, 0]) * np.sin(math.pi * x_2d[:, 1])).reshape(-1, 1)
