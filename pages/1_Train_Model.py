"""Train Model page — upload data, configure a PIELM/PINN model, train, visualise.

Flow
----
1. Upload a data file (CSV, .dat, .npz).
2. Preview the data and pick training/validation columns.
3. Select a model from the registry and tune hyper-parameters.
4. Click **Train** — runs fit() and evaluates metrics.
5. Visualise predictions vs. reference and download results.
"""

from __future__ import annotations

import io
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Guard — inform user if pypielm is missing
try:
    import torch
    import pypielm  # noqa: F401
    _PIELM_OK = True
except ImportError as _e:
    _PIELM_OK = False
    _PIELM_ERR = str(_e)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import session_state as ss
from utils.plotting import (
    plot_solution_1d, plot_solution_2d, plot_metrics_bar,
    plot_solution_3d_scatter, plot_solution_3d_surface,
    plot_isosurface_3d, plot_xt_heatmap,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Train Model · PyPIELM", page_icon="🏋️", layout="wide")

st.title("🏋️ Train Model")

if not _PIELM_OK:
    st.error(
        f"**pypielm not found:** {_PIELM_ERR}  \n"
        "Install with `pip install pypielm` or for local dev: `pip install -e ../PyPIELM`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Imports (only after guard)
# ---------------------------------------------------------------------------

from pypielm.data.dataset import PIELMDataset
from pypielm.data import auto_load
from pypielm.models.registry import MODEL_REGISTRY, get_model
from pypielm.metrics.metrics import MetricsBundle
from utils.sample_data import SAMPLE_DATASETS, BASIC_DATASETS, ADVANCED_DATASETS, get_sample_dataset


# ---------------------------------------------------------------------------
# Helper: build PIELMDataset + preview dataframe from (X, y) arrays
# ---------------------------------------------------------------------------

def _arrays_to_dataset(X: np.ndarray, y: np.ndarray) -> PIELMDataset:
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    return PIELMDataset(X_colloc=X_t, y_data=y_t)


def _make_preview(X: np.ndarray, y: np.ndarray | None) -> pd.DataFrame:
    cols = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    if y is not None:
        df["y"] = y.ravel()
    return df


def _load_file(uploaded) -> tuple[PIELMDataset, pd.DataFrame | None]:
    """Save the upload to a temp file and call auto_load."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = Path(tmp.name)
    ds = auto_load(tmp_path, dtype=torch.float64)
    tmp_path.unlink(missing_ok=True)

    preview = None
    try:
        preview = _make_preview(
            ds.X_colloc.numpy(),
            ds.y_data.numpy() if ds.y_data is not None else None,
        )
    except Exception:
        pass
    return ds, preview


# ---------------------------------------------------------------------------
# Step 1 — Data source
# ---------------------------------------------------------------------------

st.markdown("### Step 1 · Load data")

source_tab_labels = ["📦 Sample dataset", "📂 Upload your own file"]
tab_sample, tab_upload = st.tabs(source_tab_labels)

ds: PIELMDataset | None = None
df_preview: pd.DataFrame | None = None

# ---- Tab A: built-in sample datasets ----------------------------------------

with tab_sample:
    st.markdown(
        "Choose one of the built-in analytical PDE benchmark datasets.  \n"
        "All have known exact solutions so you can immediately evaluate accuracy."
    )

    ds_group = st.radio(
        "Dataset group",
        ["🔵 Classic (1D/2D)", "🔴 Advanced (3D/4D, nonlinear)"],
        horizontal=True,
        key="sample_group",
    )
    _group_names = BASIC_DATASETS if "Classic" in ds_group else ADVANCED_DATASETS
    ds_name = st.selectbox("Dataset", _group_names, key="sample_name")

    # Show dataset description
    sample_obj = get_sample_dataset(ds_name)
    info_cols = st.columns(3)
    info_cols[0].metric("PDE", sample_obj.pde.split(",")[0])
    info_cols[1].metric("Points", sample_obj.X.shape[0])
    info_cols[2].metric("Input dim", sample_obj.dim)
    st.caption(f"Exact solution: `{sample_obj.exact_str}`")

    if st.button("Load sample dataset", key="btn_sample"):
        ds = _arrays_to_dataset(sample_obj.X, sample_obj.y)
        df_preview = _make_preview(sample_obj.X, sample_obj.y)
        ss.set_dataset(ds)
        ss.set_source_name(ds_name)
        st.success(f"Loaded **{ds_name}** — {sample_obj.X.shape[0]} points, dim={sample_obj.dim}", icon="✅")

# ---- Tab B: user upload -------------------------------------------------------

with tab_upload:
    st.markdown("Upload any CSV, PINNacle `.dat`, or NumPy `.npz`/`.npy` file.")
    uploaded = st.file_uploader(
        "Supported formats: CSV, .dat, .txt, .npz, .npy",
        type=["csv", "dat", "txt", "npz", "npy"],
        key="file_uploader",
    )
    if uploaded is not None:
        with st.spinner("Loading data…"):
            try:
                ds, df_preview = _load_file(uploaded)
                ss.set_dataset(ds)
                ss.set_source_name(uploaded.name)
                st.success(
                    f"Loaded **{uploaded.name}** — "
                    f"{tuple(ds.X_colloc.shape)} collocation points",
                    icon="✅",
                )
            except Exception as exc:
                st.error(f"Failed to load `{uploaded.name}`: {exc}")
                st.code(traceback.format_exc())

# ---- Retrieve dataset from session state (persists across reruns) ------------

if ds is None:
    ds = ss.get_dataset()
    if ds is not None:
        try:
            df_preview = _make_preview(
                ds.X_colloc.numpy(),
                ds.y_data.numpy() if ds.y_data is not None else None,
            )
        except Exception:
            df_preview = None

if ds is None:
    st.info("Select or upload a dataset above, then click **Load**.", icon="ℹ️")
    st.stop()

if df_preview is not None:
    with st.expander(f"Preview — {ss.get_source_name()} (first 100 rows)"):
        st.dataframe(df_preview.head(100), use_container_width=True)

# ---------------------------------------------------------------------------
# Step 2 — Model selection
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Step 2 · Choose model")

# Filter to models that are likely to work with data-only (no PDE operator required)
_DATA_COMPATIBLE = [
    "vanilla_pielm", "core_pielm", "bayesian_pielm",
    "gff_pielm", "curriculum_pielm",
    "dpielm", "locelm",
]
available = [k for k in _DATA_COMPATIBLE if k in MODEL_REGISTRY]
model_name = st.selectbox(
    "Model",
    options=available,
    format_func=lambda k: MODEL_REGISTRY[k].__name__,
    help="Physics-informed variants (CorePIELM, GFFPIELM) work best when the "
         "data file contains PDE collocation points in x0, x1 … and targets in y.",
)

# ---------------------------------------------------------------------------
# Step 3 — Hyper-parameters
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Step 3 · Hyper-parameters")

# Introspect the selected model's __init__ to know which params it accepts
_model_sig = inspect.signature(MODEL_REGISTRY[model_name].__init__).parameters
_has = lambda p: p in _model_sig  # noqa: E731

col_a, col_b, col_c = st.columns(3)

with col_a:
    hidden_dim = st.number_input("Hidden dim", value=200, min_value=8, step=50)
    if _has("ridge_lambda"):
        ridge_lambda = st.number_input("Ridge λ", value=1e-8, format="%.2e", step=1e-9)
    elif _has("prior_precision"):
        prior_precision = st.number_input(
            "Prior precision", value=1e-4, format="%.2e", step=1e-5,
            help="Bayesian prior precision (1/variance) on the output weights."
        )
with col_b:
    if _has("activation"):
        activation = st.selectbox("Activation", ["tanh", "sin", "relu", "sigmoid"])
    seed = st.number_input("Seed", value=42, min_value=0, step=1)
with col_c:
    dtype_name = st.selectbox("Dtype", ["float64", "float32"])
    dtype = torch.float64 if dtype_name == "float64" else torch.float32

# Variant-specific extras
extras: dict = {}
if model_name == "gff_pielm":
    extras["freq_init"] = st.selectbox("Freq init", ["log_uniform", "uniform"])
elif model_name == "curriculum_pielm":
    extras["n_stages"]      = st.slider("Stages", 1, 10, 3)
    extras["n_collocation"] = st.number_input("Collocation pts", value=200, step=50)

# ---------------------------------------------------------------------------
# Step 4 — Train
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Step 4 · Train")

train_btn = st.button("▶ Train", type="primary", use_container_width=False)

if train_btn:
    ds = ss.get_dataset()
    if ds is None:
        st.error("Dataset disappeared from session. Please re-upload.")
        st.stop()

    try:
        # Build kwargs from the widgets that are relevant to this model
        model_kwargs: dict = {
            "hidden_dim": int(hidden_dim),
            "seed":       int(seed),
            "dtype":      dtype,
        }
        if _has("ridge_lambda"):
            model_kwargs["ridge_lambda"] = float(ridge_lambda)
        if _has("prior_precision"):
            model_kwargs["prior_precision"] = float(prior_precision)
        if _has("activation"):
            model_kwargs["activation"] = activation
        model_kwargs.update(extras)
        model = get_model(model_name, **model_kwargs)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        st.stop()

    log_area = st.empty()
    log_area.info("Training…", icon="⏳")

    try:
        with st.spinner("Fitting model…"):
            model.fit(ds)
        ss.set_model(model)
        config_to_save = {"model": model_name, **model_kwargs}
        config_to_save["dtype"] = dtype_name  # store human-readable name, not torch.dtype
        ss.set_config(config_to_save)
        log_area.success("Training complete!", icon="✅")
    except Exception as exc:
        log_area.error(f"Training failed: {exc}")
        st.code(traceback.format_exc())
        st.stop()

    # Evaluate
    if ds.y_data is not None:
        X_eval = ds.X_colloc
        y_eval = ds.y_data
        try:
            bundle = MetricsBundle(model.predict(X_eval), y_eval)
            metrics = bundle.compute()
            ss.set_metrics(metrics)
        except Exception:
            metrics = {}

# ---------------------------------------------------------------------------
# Step 5 — Results
# ---------------------------------------------------------------------------

model  = ss.get_model()
metrics = ss.get_metrics()

if model is None:
    st.stop()

st.divider()
st.markdown("### Step 5 · Results")

col_r, col_p = st.columns([1, 2], gap="large")

with col_r:
    st.markdown("**Metrics**")
    if metrics:
        m_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Value": [f"{v:.6g}" for v in metrics.values()]}
        )
        st.dataframe(m_df, use_container_width=True, hide_index=True)
        st.pyplot(plot_metrics_bar(metrics), use_container_width=True)
    else:
        st.info("No targets available for metric evaluation.")

    # Download button
    cfg = ss.get_config()
    summary = {"source": ss.get_source_name(), "config": cfg, "metrics": metrics}
    import json
    st.download_button(
        "⬇ Download results.json",
        data=json.dumps(summary, indent=2),
        file_name="pypielm_results.json",
        mime="application/json",
    )

with col_p:
    st.markdown("**Predictions**")
    ds = ss.get_dataset()
    if ds is not None:
        X_np = ds.X_colloc.numpy()
        try:
            y_pred = model.predict(ds.X_colloc).detach().numpy()
        except Exception as exc:
            st.warning(f"Could not generate predictions: {exc}")
            st.stop()

        y_ref = ds.y_data.numpy() if ds.y_data is not None else None

        if X_np.shape[1] == 1:
            order = np.argsort(X_np[:, 0])
            fig = plot_solution_1d(
                X_np[order, 0], y_pred[order, 0],
                y_ref[order, 0] if y_ref is not None else None,
                title=f"{MODEL_REGISTRY[ss.get_config().get('model','')].__name__} — {ss.get_source_name()}",
            )
            import matplotlib.pyplot as plt  # noqa: F811
            st.pyplot(fig, use_container_width=True)

        elif X_np.shape[1] == 2:
            # Check if it looks like (x, t) or purely spatial
            _ds_name = ss.get_source_name() or ""
            _is_xt = any(k in _ds_name for k in ["Heat 1D", "Schrödinger", "Soliton"])
            if _is_xt:
                pfig = plot_xt_heatmap(
                    X_np, y_pred, title=f"Predicted — {_ds_name}")
                rfig = plot_xt_heatmap(
                    X_np, y_ref, title=f"Reference — {_ds_name}") if y_ref is not None else None
            else:
                pfig = plot_solution_3d_surface(
                    X_np, y_pred, title=f"Predicted — {_ds_name}")
                rfig = plot_solution_3d_surface(
                    X_np, y_ref, title=f"Reference — {_ds_name}") if y_ref is not None else None

            if pfig is not None:
                st.plotly_chart(pfig, use_container_width=True)
                if rfig is not None:
                    st.plotly_chart(rfig, use_container_width=True)
            else:
                # Matplotlib fallback
                import matplotlib.pyplot as plt  # noqa: F811
                mfig = plot_solution_2d(X_np, y_pred, y_ref,
                                        title=f"{ss.get_source_name()}")
                st.pyplot(mfig, use_container_width=True)

        elif X_np.shape[1] == 3:
            _ds_name = ss.get_source_name() or ""
            _is_spatial3d = any(k in _ds_name for k in ["Poisson 3D", "Poisson3D"])
            if _is_spatial3d:
                pv_tabs = st.tabs(["Predicted (isosurface)", "Reference (isosurface)", "Scatter"])
                with pv_tabs[0]:
                    ifig = plot_isosurface_3d(X_np, y_pred, title="Prediction isosurface")
                    if ifig:
                        st.plotly_chart(ifig, use_container_width=True)
                with pv_tabs[1]:
                    if y_ref is not None:
                        ifig_r = plot_isosurface_3d(X_np, y_ref, title="Reference isosurface")
                        if ifig_r:
                            st.plotly_chart(ifig_r, use_container_width=True)
                with pv_tabs[2]:
                    sfig = plot_solution_3d_scatter(X_np, y_pred, y_ref, title=_ds_name)
                    if sfig:
                        st.plotly_chart(sfig, use_container_width=True)
            else:
                # (x, y, t) — show scatter + x-t heatmap at a slice
                pv_tabs = st.tabs(["3-D scatter", "x-t heatmap (y slice)"])
                with pv_tabs[0]:
                    sfig = plot_solution_3d_scatter(X_np, y_pred, y_ref, title=_ds_name)
                    if sfig:
                        st.plotly_chart(sfig, use_container_width=True)
                with pv_tabs[1]:
                    # Pick y ≈ median slice
                    y_vals = np.unique(X_np[:, 1])
                    mid_y = y_vals[len(y_vals) // 2]
                    mask = np.abs(X_np[:, 1] - mid_y) < 1e-9
                    if mask.sum() > 4:
                        ht = plot_xt_heatmap(
                            X_np[mask][:, [0, 2]], y_pred[mask],
                            x_axis=0, t_axis=1,
                            title=f"Heatmap at y≈{mid_y:.2f}  |  {_ds_name}",
                        )
                        if ht:
                            st.plotly_chart(ht, use_container_width=True)
                    else:
                        st.info("Slice too thin for heatmap; use scatter tab.")

        else:
            # Higher dim — just show predicted vs reference scatter
            import matplotlib.pyplot as plt  # noqa: F811
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(range(len(y_pred)), y_pred.ravel(), s=4, label="pred", color="#2563eb")
            if y_ref is not None:
                ax.scatter(range(len(y_ref)), y_ref.ravel(), s=4, label="ref", color="#16a34a", alpha=0.6)
            ax.legend()
            ax.set_title("Prediction vs reference (index order)")
            st.pyplot(fig, use_container_width=True)
