"""PyPIELM App — landing page.

Streamlit multi-page application providing a GUI for the ``pypielm`` library.

Run locally::

    streamlit run app.py

Or via Docker::

    docker compose up
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="PyPIELM",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — library version check
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.shields.io/pypi/v/pypielm?label=pypielm&color=2563eb",
        width=160,
    )
    st.divider()
    try:
        import pypielm
        st.success(f"pypielm {pypielm.__version__} loaded", icon="✅")
    except ImportError:
        st.error(
            "pypielm not installed.  \n"
            "Run `pip install pypielm` or, for local dev:  \n"
            "`pip install -e ../PyPIELM`",
            icon="❌",
        )
    st.divider()
    st.caption("Pages")
    st.page_link("pages/1_Train_Model.py",       label="Train Model",        icon="🏋️")
    st.page_link("pages/2_Benchmark_Results.py", label="Benchmark Results",  icon="🏆")
    st.page_link("pages/3_Export_Model.py",      label="Export Model",       icon="💾")
    st.page_link("pages/4_Battery_Application.py", label="Battery Digital Twin", icon="🔋")

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------

st.title("⚛️ PyPIELM")
st.subheader(
    "A Unified Framework for Physics-Informed Extreme Learning Machines"
)

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.markdown(
        """
PyPIELM provides **26 PIELM and PINN variants** with a scikit-learn-style API:
`fit` → `predict` → `score`.  All models are PyTorch-native, support GPU
acceleration, and solve PDE-constrained regression **analytically** — no
gradient descent required for ELM-based variants.

**Quick start** (Python):
```python
import pypielm
from pypielm.data import auto_load
from pypielm.models import CorePIELM
from pypielm.pde.operators import AnalyticLaplacian

ds = auto_load("poisson.dat")
model = CorePIELM(hidden_dim=300).fit(ds, pde_operator=AnalyticLaplacian())
print(model.score(ds.X_colloc, ds.y_data))   # relative L²
```

Use the sidebar to navigate to:
- **Train Model** — upload your data, pick a model, train, and visualise.
- **Benchmark Results** — compare model accuracy across PDE tasks.
- **Export Model** — save a trained model to TorchScript or ONNX.
        """
    )

with col_right:
    st.markdown("### Available models")
    models_table = {
        "Variant": [
            "VanillaPIELM", "CorePIELM", "BayesianPIELM",
            "GFFPIELM", "CurriculumPIELM", "DPIELM",
            "LocELM", "DDELMCoarse", "NullSpacePIELM",
            "EigPIELM", "LSEELM", "StefanPIELM",
            "VanillaPINN", "AdaptivePINN",
        ],
        "Solver": [
            "Ridge", "Ridge/RRQR", "Bayesian",
            "Ridge/RRQR", "Adaptive", "Ridge",
            "Ridge", "Ridge", "Ridge+Null",
            "Eig", "LSE", "Iterative",
            "Adam/L-BFGS", "Adam/L-BFGS",
        ],
    }
    import pandas as pd
    st.dataframe(
        pd.DataFrame(models_table),
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Feature highlights
# ---------------------------------------------------------------------------

st.divider()
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("PIELM variants", "26+")
with c2:
    st.metric("PDE tasks tested", "5")
with c3:
    st.metric("Test coverage", "90 %")
with c4:
    st.metric("PyTorch min", "2.0")

st.divider()
st.markdown(
    """
<div style='text-align:center; color:#64748b; font-size:0.85rem;'>
PyPIELM · MIT License ·
<a href='https://github.com/kstruniawski/pypielm'>GitHub</a> ·
<a href='https://pypielm.readthedocs.io'>Docs</a> ·
<a href='https://pypi.org/project/pypielm'>PyPI</a>
</div>
""",
    unsafe_allow_html=True,
)
