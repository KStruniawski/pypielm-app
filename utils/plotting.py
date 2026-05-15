"""Plotting helpers for the PyPIELM Streamlit app.

All matplotlib functions return a ``matplotlib.figure.Figure`` so they work
with ``st.pyplot(fig)``.  Plotly functions return a ``plotly.graph_objects.Figure``
and should be rendered with ``st.plotly_chart(fig, use_container_width=True)``.
"""

from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_C_PRED  = "#2563eb"   # blue
_C_REF   = "#16a34a"   # green
_C_ERR   = "#dc2626"   # red
_C_FILL  = "#dbeafe"   # light blue


# ---------------------------------------------------------------------------
# 1-D helpers
# ---------------------------------------------------------------------------

def plot_solution_1d(
    x: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray | None = None,
    *,
    title: str = "Solution",
    xlabel: str = "x",
    ylabel: str = "u(x)",
) -> plt.Figure:
    """Line plot with optional reference and point-wise error panel."""
    n_panels = 2 if u_ref is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(x, u_pred, color=_C_PRED, lw=2, label="Predicted")
    if u_ref is not None:
        ax.plot(x, u_ref, color=_C_REF, lw=2, ls="--", label="Reference")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(framealpha=0.3)
    ax.grid(True, lw=0.4, alpha=0.5)

    if u_ref is not None:
        err = np.abs(u_pred - u_ref)
        axes[1].semilogy(x, err + 1e-16, color=_C_ERR, lw=1.5)
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel("|error|")
        axes[1].set_title("Point-wise absolute error")
        axes[1].grid(True, lw=0.4, alpha=0.5)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2-D helpers
# ---------------------------------------------------------------------------

def plot_solution_2d(
    X: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray | None = None,
    *,
    title: str = "Solution",
    cmap: str = "viridis",
) -> plt.Figure:
    """Side-by-side scatter heatmaps for prediction and error (2-D domain)."""
    n_panels = 3 if u_ref is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    sc = axes[0].scatter(
        X[:, 0], X[:, 1], c=u_pred.ravel(), cmap=cmap, s=8
    )
    plt.colorbar(sc, ax=axes[0], fraction=0.046)
    axes[0].set_title("Prediction")
    axes[0].set_aspect("equal")

    if u_ref is not None:
        sc2 = axes[1].scatter(
            X[:, 0], X[:, 1], c=u_ref.ravel(), cmap=cmap, s=8
        )
        plt.colorbar(sc2, ax=axes[1], fraction=0.046)
        axes[1].set_title("Reference")
        axes[1].set_aspect("equal")

        err = np.abs(u_pred.ravel() - u_ref.ravel())
        sc3 = axes[2].scatter(
            X[:, 0], X[:, 1], c=err, cmap="Reds", s=8
        )
        plt.colorbar(sc3, ax=axes[2], fraction=0.046)
        axes[2].set_title("Absolute error")
        axes[2].set_aspect("equal")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3-D / 4-D Plotly helpers
# ---------------------------------------------------------------------------

def _try_plotly():
    """Import plotly lazily; return None if not installed."""
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


def plot_solution_3d_scatter(
    X: np.ndarray,
    u_pred: np.ndarray,
    u_ref: np.ndarray | None = None,
    *,
    title: str = "3-D Solution",
    colorscale: str = "Viridis",
    max_points: int = 4000,
):
    """Interactive 3-D scatter coloured by value.

    For 3-D inputs ``X`` has shape ``(N, 3)``.  The colour represents the
    predicted (or reference) scalar field value.  Returns a Plotly Figure or
    ``None`` if plotly is unavailable.
    """
    go = _try_plotly()
    if go is None:
        return None

    # Sub-sample for performance
    n = X.shape[0]
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, max_points, replace=False)
        X, u_pred = X[idx], u_pred[idx]
        if u_ref is not None:
            u_ref = u_ref[idx]

    traces = [
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(size=3, color=u_pred.ravel(), colorscale=colorscale,
                        colorbar=dict(title="u pred", x=0.85), opacity=0.85),
            name="Predicted",
        )
    ]

    if u_ref is not None:
        traces.append(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(size=3, color=u_ref.ravel(), colorscale="Cividis",
                        colorbar=dict(title="u ref", x=1.0), opacity=0.5,
                        symbol="square"),
            name="Reference",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=("x" if X.shape[1] > 0 else ""),
            yaxis_title=("y" if X.shape[1] > 1 else ""),
            zaxis_title=("z / t" if X.shape[1] > 2 else ""),
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_solution_3d_surface(
    X: np.ndarray,
    u: np.ndarray,
    *,
    x_axis: int = 0,
    y_axis: int = 1,
    title: str = "Surface",
    colorscale: str = "RdBu",
):
    """Plotly surface plot for 2-D spatial data (3-D with one axis = value).

    Works when ``X`` has at least 2 columns and the domain is (approximately)
    on a structured grid — it tries to reshape into a 2-D grid.
    Returns a Plotly Figure or ``None``.
    """
    go = _try_plotly()
    if go is None:
        return None

    try:
        xs = np.unique(X[:, x_axis])
        ys = np.unique(X[:, y_axis])
        nx, ny = len(xs), len(ys)
        U = u.ravel().reshape(ny, nx)
        Xg = X[:, x_axis].reshape(ny, nx)
        Yg = X[:, y_axis].reshape(ny, nx)
    except Exception:
        return None  # irregular grid — fall back to scatter

    fig = go.Figure(data=[
        go.Surface(x=Xg, y=Yg, z=U, colorscale=colorscale, opacity=0.9)
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"x{x_axis}",
            yaxis_title=f"x{y_axis}",
            zaxis_title="u",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_isosurface_3d(
    X: np.ndarray,
    u: np.ndarray,
    *,
    n_iso: int = 6,
    title: str = "Isosurface",
    colorscale: str = "RdBu",
    opacity: float = 0.4,
):
    """Plotly volume / isosurface for 3-D spatial data.

    ``X`` must have shape ``(N, 3)``.  ``n_iso`` evenly-spaced iso-levels are
    drawn.  Returns a Plotly Figure or ``None``.
    """
    go = _try_plotly()
    if go is None:
        return None

    u_flat = u.ravel()
    vmin, vmax = float(u_flat.min()), float(u_flat.max())
    isovals = np.linspace(vmin + 0.05 * (vmax - vmin),
                          vmax - 0.05 * (vmax - vmin), n_iso).tolist()

    fig = go.Figure(data=go.Isosurface(
        x=X[:, 0].ravel(),
        y=X[:, 1].ravel(),
        z=X[:, 2].ravel(),
        value=u_flat,
        isomin=vmin,
        isomax=vmax,
        surface_count=n_iso,
        colorscale=colorscale,
        opacity=opacity,
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_xt_heatmap(
    X: np.ndarray,
    u: np.ndarray,
    *,
    x_axis: int = 0,
    t_axis: int = 1,
    title: str = "Space–time heatmap",
    colorscale: str = "RdBu",
):
    """Plotly heatmap for (x, t) data — classic space-time plot.

    Returns a Plotly Figure or ``None``.
    """
    go = _try_plotly()
    if go is None:
        return None

    try:
        xs = np.unique(X[:, x_axis])
        ts = np.unique(X[:, t_axis])
        U = u.ravel().reshape(len(ts), len(xs))
    except Exception:
        return None

    fig = go.Figure(data=go.Heatmap(
        x=xs, y=ts, z=U,
        colorscale=colorscale,
        colorbar=dict(title="u"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title=f"x{x_axis}",
        yaxis_title=f"x{t_axis} (time)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig

def plot_metrics_bar(metrics: dict[str, float]) -> plt.Figure:
    """Horizontal bar chart for a single model's metric dict."""
    labels = list(metrics.keys())
    values = [float(v) for v in metrics.values()]

    fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(labels))))
    bars = ax.barh(labels, values, color=_C_PRED, alpha=0.85)
    ax.bar_label(bars, fmt="%.4g", padding=4, fontsize=9)
    ax.set_xlabel("Value")
    ax.set_title("Model metrics")
    ax.invert_yaxis()
    ax.grid(True, axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    model_names: list[str],
    metric_values: list[float],
    *,
    metric_name: str = "relative_l2",
    lower_is_better: bool = True,
) -> plt.Figure:
    """Grouped bar chart comparing multiple models on one metric."""
    order = np.argsort(metric_values) if lower_is_better else np.argsort(metric_values)[::-1]
    sorted_names  = [model_names[i] for i in order]
    sorted_values = [metric_values[i] for i in order]

    colors = [_C_PRED if i == 0 else "#93c5fd" for i in range(len(sorted_names))]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(sorted_names))))
    bars = ax.barh(sorted_names, sorted_values, color=colors, alpha=0.9)
    ax.bar_label(bars, fmt="%.4g", padding=4, fontsize=9)
    ax.set_xlabel(metric_name)
    ax.set_title(f"Model comparison — {metric_name}")
    ax.invert_yaxis()
    ax.grid(True, axis="x", lw=0.4, alpha=0.5)
    fig.tight_layout()
    return fig
