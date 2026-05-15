"""Export Model page — save a trained model to TorchScript or ONNX.

Requires a model to have been trained on the Train Model page first
(stored in ``st.session_state``).  Alternatively, load a ``.pt`` checkpoint.
"""

from __future__ import annotations

import io
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

try:
    import torch
    import pypielm  # noqa: F401
    _PIELM_OK = True
except ImportError as _e:
    _PIELM_OK = False
    _PIELM_ERR = str(_e)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import session_state as ss

st.set_page_config(page_title="Export Model · PyPIELM", page_icon="💾", layout="wide")
st.title("💾 Export Model")

if not _PIELM_OK:
    st.error(f"pypielm not found: {_PIELM_ERR}")
    st.stop()

from pypielm.io.checkpoint import save_model, load_model
from pypielm.io.export import to_torchscript

# ---------------------------------------------------------------------------
# Model source — session or upload checkpoint
# ---------------------------------------------------------------------------

st.markdown("### Model source")
src = st.radio("Load model from", ["Current session (trained on Train page)", "Upload .pt checkpoint"], horizontal=True)

model = None

if src.startswith("Current"):
    model = ss.get_model()
    cfg   = ss.get_config()
    if model is None:
        st.warning("No trained model in the current session. Go to **Train Model** first.", icon="⚠️")
    else:
        st.success(
            f"Using **{type(model).__name__}** from session  \n"
            f"Config: `{cfg}`",
            icon="✅",
        )

else:  # upload checkpoint
    ckpt_file = st.file_uploader("Upload a PyPIELM `.pt` checkpoint", type="pt")
    if ckpt_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(ckpt_file.getvalue())
            tmp_path = Path(tmp.name)
        try:
            with st.spinner("Loading checkpoint…"):
                model = load_model(tmp_path)
            tmp_path.unlink(missing_ok=True)
            st.success(f"Loaded `{ckpt_file.name}` → {type(model).__name__}", icon="✅")
        except Exception as exc:
            st.error(f"Failed to load checkpoint: {exc}")
            st.code(traceback.format_exc())

if model is None:
    st.stop()

# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Export options")

col_l, col_r = st.columns(2, gap="large")

# ---- LEFT: Save checkpoint (.pt) -----------------------------------------

with col_l:
    st.markdown("#### PyPIELM checkpoint (.pt)")
    st.markdown(
        "Saves the model's `state_dict` + config in a format that can be "
        "reloaded with `pypielm.io.checkpoint.load_model(path)`."
    )
    include_config = st.checkbox("Include config in checkpoint", value=True)
    if st.button("Save checkpoint", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_model(model, tmp_path, include_config=include_config, overwrite=True)
            with open(tmp_path, "rb") as f:
                data = f.read()
            tmp_path.unlink(missing_ok=True)
            st.download_button(
                f"⬇ Download {type(model).__name__}.pt",
                data=data,
                file_name=f"{type(model).__name__}.pt",
                mime="application/octet-stream",
                key="dl_checkpoint",
            )
            st.success("Checkpoint ready for download.", icon="✅")
        except Exception as exc:
            st.error(f"Export failed: {exc}")
            st.code(traceback.format_exc())

# ---- RIGHT: TorchScript (.ts.pt) -----------------------------------------

with col_r:
    st.markdown("#### TorchScript (.pt)")
    st.markdown(
        "Traces the model's `forward()` method into a standalone "
        "TorchScript module — deployable without the `pypielm` package."
    )
    input_dim = st.number_input(
        "Input dimension (spatial dim of X)",
        value=2, min_value=1, step=1,
        help="Must match the dimension of X used during training.",
    )
    ts_method = st.selectbox("Method", ["trace", "script"])
    if st.button("Export TorchScript", use_container_width=True):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            example = torch.zeros(1, int(input_dim), dtype=torch.float64)
            with st.spinner("Tracing model…"):
                to_torchscript(model, tmp_path, example_input=example, method=ts_method)
            with open(tmp_path, "rb") as f:
                data = f.read()
            tmp_path.unlink(missing_ok=True)
            st.download_button(
                f"⬇ Download {type(model).__name__}_ts.pt",
                data=data,
                file_name=f"{type(model).__name__}_torchscript.pt",
                mime="application/octet-stream",
                key="dl_torchscript",
            )
            st.success("TorchScript module ready. Load it anywhere with `torch.jit.load(path)`.", icon="✅")
        except Exception as exc:
            st.error(f"TorchScript export failed: {exc}")
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Usage snippet
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### Usage")
st.code(
    """\
# Load PyPIELM checkpoint
from pypielm.io.checkpoint import load_model
model = load_model("MyModel.pt")
predictions = model.predict(X)

# Load TorchScript (no pypielm needed at inference)
import torch
scripted = torch.jit.load("MyModel_torchscript.pt")
predictions = scripted(X)
""",
    language="python",
)
