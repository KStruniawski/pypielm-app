"""Built-in sample datasets for the PyPIELM Streamlit app.

Each dataset is a pure-Python/NumPy function that returns a tuple
``(X, y, metadata)`` where:

* ``X``  — float64 ndarray, shape ``(N, d)``  collocation points
* ``y``  — float64 ndarray, shape ``(N, 1)``  reference solution values
* ``metadata`` — dict with keys: ``name``, ``pde``, ``exact_str``, ``dim``

The datasets are intentionally small (≤ 2 000 points) so they work
comfortably in the browser without a GPU.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class SampleDataset(NamedTuple):
    X: np.ndarray          # shape (N, d), float64
    y: np.ndarray          # shape (N, 1), float64
    name: str
    pde: str
    exact_str: str         # human-readable formula
    dim: int               # spatial dimension


# ---------------------------------------------------------------------------
# 1-D Poisson  u'' = -2,  u(0)=u(1)=0  →  u=x(1-x)
# ---------------------------------------------------------------------------

def poisson_1d(n: int = 500) -> SampleDataset:
    x = np.linspace(0.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    u = x * (1.0 - x)
    return SampleDataset(
        X=x, y=u,
        name="Poisson 1D",
        pde="-u'' = 2,   u(0)=u(1)=0",
        exact_str="u(x) = x(1-x)",
        dim=1,
    )


# ---------------------------------------------------------------------------
# 1-D Heat / diffusion  u_t = u_xx,  t∈[0,1]  periodic IC sin(πx)
#   u(x,t) = exp(-π²t) sin(πx)
# ---------------------------------------------------------------------------

def heat_1d(nx: int = 40, nt: int = 30) -> SampleDataset:
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    XX, TT = np.meshgrid(xs, ts)
    X = np.column_stack([XX.ravel(), TT.ravel()])
    u = np.exp(-math.pi ** 2 * X[:, 1]) * np.sin(math.pi * X[:, 0])
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Heat 1D (transient)",
        pde="u_t = u_xx,  u(x,0)=sin(πx)",
        exact_str="u(x,t) = exp(-π²t) sin(πx)",
        dim=2,
    )


# ---------------------------------------------------------------------------
# 2-D Poisson  -Δu = 2π²sin(πx)sin(πy)  on [0,1]²
#   u(x,y) = sin(πx)sin(πy)
# ---------------------------------------------------------------------------

def poisson_2d(n_per_dim: int = 40) -> SampleDataset:
    xs = np.linspace(0.0, 1.0, n_per_dim, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    X = np.column_stack([XX.ravel(), YY.ravel()])
    u = np.sin(math.pi * X[:, 0]) * np.sin(math.pi * X[:, 1])
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Poisson 2D",
        pde="-Δu = 2π²sin(πx)sin(πy),  [0,1]²",
        exact_str="u(x,y) = sin(πx)sin(πy)",
        dim=2,
    )


# ---------------------------------------------------------------------------
# 1-D Burgers (steady, viscous)
#   -ν u'' + u u' = 0,   ν=0.01
#   Approximate analytical profile: tanh-based travelling wave
# ---------------------------------------------------------------------------

def burgers_steady_1d(n: int = 500, nu: float = 0.01) -> SampleDataset:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    # Steady viscous Burgers: u(x) = -tanh(x / (2ν))  (approximate)
    u = -np.tanh(x / (2.0 * nu))
    return SampleDataset(
        X=x, y=u,
        name="Burgers 1D (steady, ν=0.01)",
        pde="-ν u'' + u u' = 0,  x∈[-1,1]",
        exact_str="u(x) ≈ -tanh(x / 2ν)",
        dim=1,
    )


# ---------------------------------------------------------------------------
# 2-D wave / Helmholtz  -Δu - k²u = f
#   k=π,  exact solution: u(x,y) = sin(πx)cos(πy)
#   f = (π² + k²) sin(πx)cos(πy) — but with k=π → f = 2π² sin(πx)cos(πy)
# ---------------------------------------------------------------------------

def helmholtz_2d(n_per_dim: int = 40, k: float = math.pi) -> SampleDataset:
    xs = np.linspace(0.0, 1.0, n_per_dim, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    X = np.column_stack([XX.ravel(), YY.ravel()])
    u = np.sin(math.pi * X[:, 0]) * np.cos(math.pi * X[:, 1])
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Helmholtz 2D (k=π)",
        pde="-Δu - k²u = f,   k=π,   [0,1]²",
        exact_str="u(x,y) = sin(πx)cos(πy)",
        dim=2,
    )


# ---------------------------------------------------------------------------
# 1-D Allen–Cahn (steady)
#   u'' - u(u²-1)/ε² = 0,  ε=0.1
#   Approximate tanh interface: u(x) = tanh(x / (ε√2))
# ---------------------------------------------------------------------------

def allen_cahn_1d(n: int = 500, eps: float = 0.1) -> SampleDataset:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64).reshape(-1, 1)
    u = np.tanh(x / (eps * math.sqrt(2.0)))
    return SampleDataset(
        X=x, y=u,
        name="Allen–Cahn 1D (ε=0.1)",
        pde="u'' - u(u²-1)/ε² = 0,  x∈[-1,1]",
        exact_str="u(x) ≈ tanh(x / (ε√2))",
        dim=1,
    )


# ---------------------------------------------------------------------------
# 3-D Poisson  -Δu = 3π²sin(πx)sin(πy)sin(πz)  on [0,1]³
#   u(x,y,z) = sin(πx)sin(πy)sin(πz)
# ---------------------------------------------------------------------------

def poisson_3d(n_per_dim: int = 18) -> SampleDataset:
    """3-D Poisson with exact sine solution — visualisable as isosurface."""
    xs = np.linspace(0.0, 1.0, n_per_dim, dtype=np.float64)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs)
    X = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    u = (
        np.sin(math.pi * X[:, 0])
        * np.sin(math.pi * X[:, 1])
        * np.sin(math.pi * X[:, 2])
    )
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Poisson 3D",
        pde="-Δu = 3π²sin(πx)sin(πy)sin(πz),   [0,1]³",
        exact_str="u(x,y,z) = sin(πx)sin(πy)sin(πz)",
        dim=3,
    )


# ---------------------------------------------------------------------------
# 2-D Heat transient  u_t = Δu  on [0,1]²×[0,T]
#   u(x,y,t) = exp(-2π²t) sin(πx) sin(πy)
# ---------------------------------------------------------------------------

def heat_2d_transient(nx: int = 20, nt: int = 20) -> SampleDataset:
    """3-D input (x, y, t) heat equation — good for animated surface plots."""
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    XX, YY, TT = np.meshgrid(xs, xs, ts)
    X = np.column_stack([XX.ravel(), YY.ravel(), TT.ravel()])
    u = (
        np.exp(-2.0 * math.pi ** 2 * X[:, 2])
        * np.sin(math.pi * X[:, 0])
        * np.sin(math.pi * X[:, 1])
    )
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Heat 2D transient",
        pde="u_t = Δu,   u(x,y,0) = sin(πx)sin(πy)",
        exact_str="u(x,y,t) = exp(-2π²t) sin(πx) sin(πy)",
        dim=3,
    )


# ---------------------------------------------------------------------------
# Navier-Stokes — Kovasznay 2-D steady flow  (Re = 40)
#   Exact u-velocity: u(x,y) = 1 - exp(λx)cos(2πy)
#   λ = Re/2 - sqrt(Re²/4 + 4π²)
# ---------------------------------------------------------------------------

def navier_stokes_kovasznay(n_per_dim: int = 50, Re: float = 40.0) -> SampleDataset:
    """Kovasznay exact NS solution — nonlinear, sharp boundary layer at x=0."""
    lam = Re / 2.0 - math.sqrt(Re ** 2 / 4.0 + 4.0 * math.pi ** 2)
    xs = np.linspace(-0.5, 1.5, n_per_dim, dtype=np.float64)
    ys = np.linspace(-0.5, 1.5, n_per_dim, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    X = np.column_stack([XX.ravel(), YY.ravel()])
    u = 1.0 - np.exp(lam * X[:, 0]) * np.cos(2.0 * math.pi * X[:, 1])
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Navier-Stokes Kovasznay (Re=40)",
        pde="(u·∇)u - (1/Re)Δu + ∇p = 0,   ∇·u=0",
        exact_str="u = 1 - exp(λx)cos(2πy),  λ≈-6.24",
        dim=2,
    )


# ---------------------------------------------------------------------------
# Nonlinear Schrödinger  (|u|² focusing soliton)
#   i u_t + u_xx + 2|u|²u = 0
#   1-soliton: |u(x,t)| = sech(x - 4t)  (η=1, v=4)
# ---------------------------------------------------------------------------

def nonlinear_schrodinger_soliton(nx: int = 100, nt: int = 60) -> SampleDataset:
    """NLS bright soliton amplitude — sharp localised feature, highly nonlinear."""
    xs = np.linspace(-8.0, 8.0, nx, dtype=np.float64)
    ts = np.linspace(0.0, 2.0, nt, dtype=np.float64)
    XX, TT = np.meshgrid(xs, ts)
    X = np.column_stack([XX.ravel(), TT.ravel()])
    amp = 1.0 / np.cosh(X[:, 0] - 4.0 * X[:, 1])   # |u| = sech(x-4t)
    return SampleDataset(
        X=X, y=amp.reshape(-1, 1),
        name="Nonlinear Schrödinger Soliton",
        pde="i u_t + u_xx + 2|u|²u = 0",
        exact_str="|u(x,t)| = sech(x - 4t)",
        dim=2,
    )


# ---------------------------------------------------------------------------
# Wave equation 3-D input  (x, y, t)
#   u_tt = Δu,  c = 1
#   u(x,y,t) = sin(πx)sin(πy)cos(π√2 t)
# ---------------------------------------------------------------------------

def wave_2d(nx: int = 20, nt: int = 20) -> SampleDataset:
    """2-D wave equation — 3-D input (x, y, t), oscillatory in time."""
    xs = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    ts = np.linspace(0.0, 2.0, nt, dtype=np.float64)
    XX, YY, TT = np.meshgrid(xs, xs, ts)
    X = np.column_stack([XX.ravel(), YY.ravel(), TT.ravel()])
    u = (
        np.sin(math.pi * X[:, 0])
        * np.sin(math.pi * X[:, 1])
        * np.cos(math.pi * math.sqrt(2.0) * X[:, 2])
    )
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Wave 2D (3-D input)",
        pde="u_tt = Δu,   u(x,y,0) = sin(πx)sin(πy),  u_t(x,y,0) = 0",
        exact_str="u(x,y,t) = sin(πx)sin(πy)cos(π√2 t)",
        dim=3,
    )


# ---------------------------------------------------------------------------
# Nonlinear Reaction-Diffusion (Brusselator, u component)
#   u_t = Du·Δu + A - (B+1)u + u²v,  steady 2-D pattern at u=A, v=B/A + small
#   Use spatial ring pattern: u(x,y) = A + C·cos(2πx)·cos(2πy) (linearised)
#   for a richer nonlinear benchmark use exact manufactured solution with cubic term
# ---------------------------------------------------------------------------

def nonlinear_reaction_diffusion_2d(n_per_dim: int = 45) -> SampleDataset:
    """Nonlinear 2-D diffusion-reaction with cubic nonlinearity — complex pattern."""
    xs = np.linspace(0.0, 1.0, n_per_dim, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)
    X = np.column_stack([XX.ravel(), YY.ravel()])
    # Manufactured exact solution: u = sin(2πx)sin(2πy) + 0.3·sin(6πx)sin(4πy)
    # This has a multi-scale structure challenging for ELMs with fixed frequencies
    u = (
        np.sin(2 * math.pi * X[:, 0]) * np.sin(2 * math.pi * X[:, 1])
        + 0.3 * np.sin(6 * math.pi * X[:, 0]) * np.sin(4 * math.pi * X[:, 1])
        + 0.1 * np.sin(10 * math.pi * X[:, 0]) * np.sin(10 * math.pi * X[:, 1])
    )
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Multi-scale Reaction-Diffusion 2D",
        pde="-DΔu + f(u) = g(x,y)  [multi-frequency manufactured]",
        exact_str="u = sin(2πx)sin(2πy) + 0.3sin(6πx)sin(4πy) + 0.1sin(10πx)sin(10πy)",
        dim=2,
    )


# ---------------------------------------------------------------------------
# Cahn-Hilliard / Phase field 3-D input (x, y, t) — diffuse interface evolution
#   u_t = Δ(-ε²Δu + W'(u)),  W(u)=¼(u²-1)²
#   Use radius-shrinking circle ansatz:
#   u(x,y,t) ≈ tanh((R(t) - r(x,y)) / (ε√2)),  R(t)=√(R₀² - 4t/3)
# ---------------------------------------------------------------------------

def cahn_hilliard_2d(nx: int = 30, nt: int = 15, eps: float = 0.05, R0: float = 0.4) -> SampleDataset:
    """Cahn-Hilliard shrinking circle — 3-D input, sharp diffuse interface, strongly nonlinear."""
    xs = np.linspace(-0.5, 0.5, nx, dtype=np.float64)
    # Use small time range so R(t) stays positive
    ts = np.linspace(0.0, 0.05, nt, dtype=np.float64)
    XX, YY, TT = np.meshgrid(xs, xs, ts)
    X = np.column_stack([XX.ravel(), YY.ravel(), TT.ravel()])
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    Rt = np.sqrt(np.maximum(R0 ** 2 - 4.0 * X[:, 2] / 3.0, 1e-6))
    u = np.tanh((Rt - r) / (eps * math.sqrt(2.0)))
    return SampleDataset(
        X=X, y=u.reshape(-1, 1),
        name="Cahn-Hilliard 2D (shrinking circle)",
        pde="u_t = Δ(-ε²Δu + W'(u)),  W(u)=¼(u²-1)²,  ε=0.05",
        exact_str="u(x,y,t) ≈ tanh((R(t)-r)/(ε√2)),  R(t)=√(0.16-4t/3)",
        dim=3,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Basic datasets (1-D and 2-D)
_BASIC: dict[str, callable] = {
    "Poisson 1D":                        poisson_1d,
    "Heat 1D (transient)":               heat_1d,
    "Poisson 2D":                        poisson_2d,
    "Burgers 1D (steady, ν=0.01)":       burgers_steady_1d,
    "Helmholtz 2D (k=π)":               helmholtz_2d,
    "Allen–Cahn 1D (ε=0.1)":            allen_cahn_1d,
    "Navier-Stokes Kovasznay (Re=40)":   navier_stokes_kovasznay,
    "Nonlinear Schrödinger Soliton":     nonlinear_schrodinger_soliton,
    "Multi-scale Reaction-Diffusion 2D": nonlinear_reaction_diffusion_2d,
}

# 3-D / 4-D datasets (3+ input dimensions)
_ADVANCED: dict[str, callable] = {
    "Poisson 3D":                        poisson_3d,
    "Heat 2D transient (3-D input)":     heat_2d_transient,
    "Wave 2D (3-D input)":               wave_2d,
    "Cahn-Hilliard 2D (shrinking circle)": cahn_hilliard_2d,
}

SAMPLE_DATASETS: dict[str, callable] = {**_BASIC, **_ADVANCED}

# Convenience groupings for the UI
BASIC_DATASETS   = list(_BASIC.keys())
ADVANCED_DATASETS = list(_ADVANCED.keys())


def get_sample_dataset(name: str, **kwargs) -> SampleDataset:
    """Return a ``SampleDataset`` by registry name."""
    if name not in SAMPLE_DATASETS:
        raise KeyError(f"Unknown sample dataset {name!r}. Available: {list(SAMPLE_DATASETS)}")
    return SAMPLE_DATASETS[name](**kwargs)
