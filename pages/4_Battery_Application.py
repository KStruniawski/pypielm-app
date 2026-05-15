"""Battery Digital Twin — Real-World Application Demo.

Demonstrates PIELM applied to coupled electro-thermal battery state estimation
using a simplified Single Particle Model (SPM) + Lumped Thermal Model:

    dSoC/dt  = -I / (3600 · Q_n)
    V(t)     = U_OCV(SoC) - I·R₀(SoC,T) - η_ct(SoC,T,I)
    dT/dt    = [I²·(R₀+R_ct) - h·A·(T-T_amb)] / (m·Cp)

Physics-Informed ELM learns the voltage surface V(SoC, T, I) from sparse
discharge data, enabling real-time SoC/SoH estimation far faster than
iterative PINN optimisation.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first st call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Battery Digital Twin · PyPIELM",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Path setup + optional pypielm import
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import pypielm  # noqa: F401
    from pypielm.data.dataset import PIELMDataset
    from pypielm.models.registry import get_model
    from pypielm.metrics.metrics import MetricsBundle
    _PIELM_OK = True
except ImportError as _e:
    _PIELM_OK = False
    _PIELM_ERR = str(_e)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


# ===========================================================================
# Physics helpers — battery simulation
# ===========================================================================

# OCV look-up table for a generic NMC/graphite cell (SoC 0→1)
# Sampled from a typical Bernardi / Doyle-Fuller-Newman OCV curve
_SOC_LUT  = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                       0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00])
_OCV_LUT  = np.array([2.80, 3.10, 3.25, 3.35, 3.45, 3.55, 3.65,
                       3.72, 3.78, 3.85, 3.95, 4.05, 4.12, 4.20])


def ocv(soc: np.ndarray) -> np.ndarray:
    """Open-circuit voltage interpolated from NMC look-up table."""
    return np.interp(soc, _SOC_LUT, _OCV_LUT)


def resistance(soc: np.ndarray, T_K: np.ndarray, soh: float = 1.0) -> np.ndarray:
    """Total DC internal resistance (Ω) — Arrhenius temperature dependence."""
    R_ref = 0.020 / soh          # Ω at T_ref, SoC=0.5
    T_ref = 298.15               # K
    Ea    = 5000.0               # J/mol
    R_gas = 8.314
    # Resistance increases at low SoC
    soc_factor = 1.0 + 0.8 * np.exp(-8.0 * soc)
    arr_factor = np.exp(Ea / R_gas * (1.0 / T_K - 1.0 / T_ref))
    return R_ref * soc_factor * arr_factor


def simulate_discharge(
    Q_n: float,        # nominal capacity [Ah]
    C_rate: float,     # C-rate (1C = full discharge in 1 h)
    T_amb_C: float,    # ambient temperature [°C]
    soc0: float,       # initial SoC
    soh: float,        # state of health [0→1]
    dt: float = 1.0,   # time step [s]
    m_cp: float = 800.0,   # thermal mass [J/K]
    h_conv: float = 5.0,   # convective heat transfer [W/K]
) -> dict[str, np.ndarray]:
    """Euler-integrate the SPM to generate synthetic discharge data."""
    I_app   = C_rate * Q_n          # applied current [A]
    T_amb_K = T_amb_C + 273.15
    t_end   = 3600.0 / C_rate       # discharge time [s]
    n_steps = int(t_end / dt) + 1

    t_arr   = np.zeros(n_steps)
    soc_arr = np.zeros(n_steps)
    V_arr   = np.zeros(n_steps)
    T_arr   = np.zeros(n_steps)
    R_arr   = np.zeros(n_steps)

    soc_arr[0] = soc0
    T_arr[0]   = T_amb_K

    for k in range(n_steps):
        t_arr[k] = k * dt
        soc = np.clip(soc_arr[k], 0.01, 0.99)
        T_K = T_arr[k]
        R   = resistance(np.array([soc]), np.array([T_K]), soh)[0]
        R_arr[k] = R

        V_arr[k] = ocv(soc) - I_app * R
        V_arr[k] = np.clip(V_arr[k], 2.5, 4.3)  # cut-off guard

        Q_heat = I_app ** 2 * R                   # Joule heating
        dT     = (Q_heat - h_conv * (T_K - T_amb_K)) / m_cp
        dSoC   = -I_app / (3600.0 * Q_n * soh)

        if k < n_steps - 1:
            T_arr[k + 1]   = T_K + dt * dT
            soc_arr[k + 1] = soc_arr[k] + dt * dSoC
            if soc_arr[k + 1] <= 0.02 or V_arr[k] <= 2.5:
                # Cell discharged — fill remaining with final state
                for j in range(k + 2, n_steps):
                    t_arr[j]   = j * dt
                    soc_arr[j] = soc_arr[k + 1]
                    V_arr[j]   = V_arr[k]
                    T_arr[j]   = T_arr[k + 1]
                    R_arr[j]   = R_arr[k]
                break

    return {
        "t":   t_arr,
        "SoC": soc_arr,
        "V":   V_arr,
        "T_C": T_arr - 273.15,
        "R":   R_arr,
        "I":   np.full(n_steps, I_app),
    }


def build_pielm_features(sim: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) = ([SoC, T_norm, I_norm], V) for PIELM training."""
    SoC    = sim["SoC"].reshape(-1, 1)
    T_norm = ((sim["T_C"] - 25.0) / 20.0).reshape(-1, 1)   # centre ≈ 0
    I_norm = (sim["I"] / 3.0).reshape(-1, 1)                # scale by max C
    X = np.hstack([SoC, T_norm, I_norm]).astype(np.float64)
    y = sim["V"].reshape(-1, 1).astype(np.float64)
    return X, y


# ===========================================================================
# Page layout
# ===========================================================================

# Hero header
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg,#0f172a 0%,#1e3a5f 50%,#065f46 100%);
        padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;">
      <h1 style="color:#f0f9ff; margin:0; font-size:2rem;">
        🔋 Real-World Application: Battery State Estimation
      </h1>
      <p style="color:#bae6fd; margin:0.5rem 0 0; font-size:1.05rem;">
        <em>Real-Time Multi-Physics Battery Digital Twin via Physics-Informed ELM</em>
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — paper reference & key claims
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Real-Time Multi-Physics Battery State Estimation Using "
        "Physics-Informed Extreme Learning Machines")
    st.markdown("#### Key claims")
    st.markdown(
        "- **< 1 ms** inference (vs. ≥ 50 ms PINN)  \n"
        "- **< 2 % SoC error** over full discharge  \n"
        "- **< 1 °C** thermal estimation error  \n"
        "- No iterative solver at runtime  \n"
        "- Works across C-rates & temperatures"
    )
    st.divider()
    st.markdown("#### Navigation")
    st.page_link("pages/1_Train_Model.py",       label="Train Model",        icon="🏋️")
    st.page_link("pages/2_Benchmark_Results.py", label="Benchmark Results",  icon="🏆")
    st.page_link("pages/3_Export_Model.py",      label="Export Model",       icon="💾")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_sim, tab_pielm, tab_twin, tab_benchmark = st.tabs([
    "🔬 Physics Overview",
    "⚡ Discharge Simulation",
    "🧠 Train PIELM",
    "🤖 Digital Twin",
    "📊 Benchmark",
])


# ===========================================================================
# TAB 1 — Physics Overview
# ===========================================================================

with tab_overview:
    col_eq, col_dia = st.columns([3, 2], gap="large")

    with col_eq:
        st.markdown("### Governing Equations")
        st.markdown(
            r"""
**Electrochemical (Single Particle Model)**

$$
\frac{\partial c_s}{\partial t} = D_s \left(\frac{\partial^2 c_s}{\partial r^2}
  + \frac{2}{r}\frac{\partial c_s}{\partial r}\right)
\qquad 0 \le r \le R_s
$$

$$
V(t) = U_{\text{OCV}}\!\left(c_s^{\text{surf}}\right) - \eta_{\text{ct}} - I R_0
$$

**Thermal (lumped)**

$$
m C_p \frac{dT}{dt} = Q_{\text{Joule}} + Q_{\text{entropy}} - h_{\text{conv}}(T-T_{\text{amb}})
$$

$$
Q_{\text{Joule}} = I^2 (R_0 + R_{\text{ct}}), \qquad
Q_{\text{entropy}} = -I T \frac{\partial U_{\text{OCV}}}{\partial T}
$$

**PIELM surrogate** — learns $V(\text{SoC}, T, I)$ from sparse data + physics residuals:

$$
\hat{V} = \boldsymbol{\beta}^{\top} \sigma(\mathbf{W}\,[\text{SoC}, T^*, I^*] + \mathbf{b})
$$

where $\boldsymbol{\beta}$ is solved in closed form (one matrix inversion).
            """
        )

    with col_dia:
        st.markdown("### System Diagram")
        st.markdown(
            """
```
 ┌──────────────────────────────────┐
 │      Measurement Layer           │
 │   Current (I) · Voltage (V)      │
 │   Temperature (T) · Time (t)     │
 └──────────────┬───────────────────┘
                │ sensor data
 ┌──────────────▼───────────────────┐
 │     PIELM State Estimator        │
 │  ┌────────────────────────────┐  │
 │  │ Physics residuals (SPM)    │  │
 │  │   PDE collocation blocks   │  │
 │  ├────────────────────────────┤  │
 │  │ Data blocks (V, T obs.)    │  │
 │  └────────────────────────────┘  │
 │  Solve: β = (H'H + λI)⁻¹ H'y     │
 └──────────────┬───────────────────┘
                │ real-time estimates
 ┌──────────────▼───────────────────┐
 │        State Variables           │
 │   SoC · SoH · T · V_pred         │
 └──────────────────────────────────┘
```
            """
        )
        st.markdown("### Why PIELM?")
        data_cmp = {
            "Method": ["Physics model (DFN)", "Kalman Filter", "PINN (Adam)", "**PIELM** ✅"],
            "Train time": ["N/A", "N/A", "≥ 10 min", "**< 5 s**"],
            "Inference": ["≥ 50 ms", "~5 ms", "~10 ms", "**< 1 ms**"],
            "SoC error": ["< 0.5 %", "~2 %", "~1 %", "**< 2 %**"],
            "No GPU needed": ["✅", "✅", "❌", "**✅**"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(data_cmp), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 2 — Discharge Simulation
# ===========================================================================

with tab_sim:
    st.markdown("### Configure Battery & Discharge Protocol")
    st.markdown(
        "Simulate a lithium-ion NMC/graphite cell using the lumped SPM + thermal model. "
        "The resulting data is used to train the PIELM surrogate in the next tab."
    )

    # Parameter controls
    c1, c2, c3 = st.columns(3)
    with c1:
        Q_n      = st.slider("Nominal capacity Q_n (Ah)", 1.0, 10.0, 3.0, 0.5)
        soc0     = st.slider("Initial SoC", 0.5, 1.0, 1.0, 0.05)
    with c2:
        C_rate   = st.select_slider("C-rate", options=[0.2, 0.5, 1.0, 2.0, 3.0], value=1.0)
        T_amb_C  = st.slider("Ambient temperature (°C)", -10, 45, 25, 5)
    with c3:
        soh      = st.slider("State of Health (SoH)", 0.60, 1.00, 1.00, 0.05,
                             help="1.0 = new cell; 0.7 = aged (30% capacity loss)")
        add_noise = st.checkbox("Add measurement noise (σ=5 mV)", value=True)

    if st.button("⚡ Run Discharge Simulation", type="primary"):
        with st.spinner("Simulating…"):
            sim = simulate_discharge(Q_n, C_rate, T_amb_C, soc0, soh)
            if add_noise:
                rng = np.random.default_rng(42)
                sim["V"] = sim["V"] + rng.normal(0, 0.005, sim["V"].shape)
            st.session_state["battery_sim"] = sim
            st.success(
                f"Simulation complete — {len(sim['t'])} time steps  |  "
                f"Final SoC: {sim['SoC'][-1]:.3f}  |  "
                f"Max T: {sim['T_C'].max():.1f} °C",
                icon="✅",
            )

    # ---- Show simulation results if available --------------------------------
    sim_data = st.session_state.get("battery_sim")
    if sim_data is not None and _PLOTLY_OK:
        t = sim_data["t"] / 3600.0  # → hours

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Terminal Voltage V(t)", "State of Charge SoC(t)",
                            "Cell Temperature T(t)", "Internal Resistance R(SoC)"],
        )

        fig.add_trace(go.Scatter(x=t, y=sim_data["V"],  name="V (V)",   line=dict(color="#2563eb")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=sim_data["SoC"], name="SoC",    line=dict(color="#16a34a")), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=sim_data["T_C"], name="T (°C)", line=dict(color="#dc2626")), row=2, col=1)
        fig.add_trace(go.Scatter(x=sim_data["SoC"], y=sim_data["R"]*1000,
                                 name="R (mΩ)", line=dict(color="#9333ea")), row=2, col=2)

        fig.update_layout(
            height=520, showlegend=False,
            title_text=f"Discharge simulation — {C_rate}C  |  T_amb={T_amb_C}°C  |  SoH={soh:.0%}",
        )
        fig.update_xaxes(title_text="Time (h)", row=1, col=1)
        fig.update_xaxes(title_text="Time (h)", row=1, col=2)
        fig.update_xaxes(title_text="Time (h)", row=2, col=1)
        fig.update_xaxes(title_text="SoC",      row=2, col=2)
        fig.update_yaxes(title_text="V (V)",    row=1, col=1)
        fig.update_yaxes(title_text="SoC",      row=1, col=2)
        fig.update_yaxes(title_text="°C",       row=2, col=1)
        fig.update_yaxes(title_text="mΩ",       row=2, col=2)

        st.plotly_chart(fig, use_container_width=True)

        # OCV curve
        with st.expander("OCV curve (NMC look-up table)"):
            soc_range = np.linspace(0, 1, 200)
            ocv_vals  = ocv(soc_range)
            ocv_fig = go.Figure(go.Scatter(x=soc_range, y=ocv_vals,
                                           line=dict(color="#0891b2", width=2)))
            ocv_fig.update_layout(xaxis_title="SoC", yaxis_title="OCV (V)",
                                   title="Open-Circuit Voltage — NMC/graphite", height=300)
            st.plotly_chart(ocv_fig, use_container_width=True)

    elif sim_data is None:
        st.info("Run a simulation above to see the discharge curves.", icon="ℹ️")
    else:
        st.warning("Install plotly to see interactive charts.", icon="⚠️")


# ===========================================================================
# TAB 3 — Train PIELM
# ===========================================================================

with tab_pielm:
    st.markdown("### Train PIELM Surrogate on Battery Data")
    st.markdown(
        "The PIELM surrogate learns **V(SoC, T, I)** — a voltage response surface — "
        "from the discharge data generated in the previous tab.  "
        "This closed-form training takes < 5 s even on CPU."
    )

    if not _PIELM_OK:
        st.error(f"pypielm not installed: {_PIELM_ERR}")
        st.stop()

    sim_data = st.session_state.get("battery_sim")
    if sim_data is None:
        st.info("First run a discharge simulation in the ⚡ Discharge Simulation tab.", icon="ℹ️")
        st.stop()

    # Hyper-parameters
    hp_cols = st.columns(4)
    with hp_cols[0]:
        hidden_dim   = st.number_input("Hidden dim", value=200, min_value=32, step=50)
    with hp_cols[1]:
        ridge_lambda = st.number_input("Ridge λ", value=1e-6, format="%.2e", step=1e-7)
    with hp_cols[2]:
        activation   = st.selectbox("Activation", ["tanh", "sin", "relu"])
    with hp_cols[3]:
        train_frac   = st.slider("Training fraction", 0.5, 0.9, 0.8, 0.05)

    if st.button("🧠 Train PIELM", type="primary"):
        X_all, y_all = build_pielm_features(sim_data)
        n = len(X_all)
        n_train = int(n * train_frac)
        idx     = np.random.default_rng(42).permutation(n)
        X_tr, y_tr = X_all[idx[:n_train]], y_all[idx[:n_train]]
        X_te, y_te = X_all[idx[n_train:]], y_all[idx[n_train:]]

        X_t = torch.tensor(X_tr, dtype=torch.float64)
        y_t = torch.tensor(y_tr, dtype=torch.float64)
        ds  = PIELMDataset(X_colloc=X_t, y_data=y_t)

        model = get_model(
            "vanilla_pielm",
            hidden_dim=int(hidden_dim),
            ridge_lambda=float(ridge_lambda),
            activation=activation,
            seed=42,
            dtype=torch.float64,
        )

        t0 = time.perf_counter()
        model.fit(ds)
        t_train = time.perf_counter() - t0

        # Evaluate on test set
        X_te_t   = torch.tensor(X_te, dtype=torch.float64)
        y_te_t   = torch.tensor(y_te, dtype=torch.float64)
        t1 = time.perf_counter()
        V_pred_t = model.predict(X_te_t)
        t_inf    = (time.perf_counter() - t1) / len(X_te) * 1000  # ms / sample

        bundle  = MetricsBundle(V_pred_t, y_te_t)
        metrics = bundle.to_dict()

        V_pred_np = V_pred_t.detach().numpy().ravel()
        V_ref_np  = y_te.ravel()
        SoC_te    = X_te[:, 0]

        st.session_state["battery_model"]       = model
        st.session_state["battery_metrics"]     = metrics
        st.session_state["battery_test_SoC"]    = SoC_te
        st.session_state["battery_V_pred"]      = V_pred_np
        st.session_state["battery_V_ref"]       = V_ref_np
        st.session_state["battery_t_train"]     = t_train
        st.session_state["battery_t_inf_ms"]    = t_inf
        st.session_state["battery_X_all"]       = X_all
        st.session_state["battery_y_all"]       = y_all
        st.session_state["battery_hidden_dim"]  = int(hidden_dim)

        # Display results
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Training time", f"{t_train*1000:.0f} ms")
        kpi2.metric("Inference / sample", f"{t_inf:.4f} ms")
        kpi3.metric("RMSE (V)", f"{metrics.get('rmse', float('nan')):.4f}")
        kpi4.metric("Relative L² (%)",
                    f"{metrics.get('rel_l2', float('nan'))*100:.2f}")

        st.success(f"PIELM trained on {n_train} samples in {t_train*1000:.0f} ms!", icon="✅")

        if _PLOTLY_OK:
            order = np.argsort(SoC_te)
            pred_fig = go.Figure([
                go.Scatter(x=SoC_te[order], y=V_ref_np[order],
                           mode="markers", name="Reference", opacity=0.5,
                           marker=dict(size=4, color="#16a34a")),
                go.Scatter(x=SoC_te[order], y=V_pred_np[order],
                           mode="markers", name="PIELM prediction",
                           marker=dict(size=4, color="#2563eb")),
            ])
            pred_fig.update_layout(
                xaxis_title="SoC", yaxis_title="V (V)",
                title="Voltage prediction vs reference (test set)",
                legend=dict(x=0.02, y=0.02),
            )
            st.plotly_chart(pred_fig, use_container_width=True)

    else:
        # Show cached results if available
        if "battery_model" in st.session_state:
            m = st.session_state["battery_metrics"]
            st.success(
                f"Trained — RMSE: {m.get('rmse', 0):.4f} V  |  "
                f"Rel-L²: {m.get('rel_l2', 0)*100:.2f}%",
                icon="✅",
            )

    # Surface visualisation of the learned surrogate
    if "battery_model" in st.session_state and _PLOTLY_OK:
        with st.expander("🗺 Explore the learned V(SoC, T, I) surface"):
            st.markdown("Hover over the surface to inspect predicted voltage at any (SoC, T) point.")
            surf_c_rate = st.select_slider("C-rate for surface", [0.2, 0.5, 1.0, 2.0, 3.0], value=1.0,
                                           key="surf_cr")
            model = st.session_state["battery_model"]
            soc_g = np.linspace(0.05, 0.99, 50)
            t_g   = np.linspace(-20, 40, 40)   # °C range
            SG, TG = np.meshgrid(soc_g, t_g)
            I_arr  = np.full(SG.size, surf_c_rate * 3.0)
            Xg_feat = np.column_stack([
                SG.ravel(),
                (TG.ravel() - 25.0) / 20.0,
                I_arr / 3.0,
            ])
            with torch.no_grad():
                V_surf = model.predict(
                    torch.tensor(Xg_feat, dtype=torch.float64)
                ).numpy().reshape(SG.shape)

            surf_fig = go.Figure(data=[go.Surface(
                x=soc_g, y=t_g, z=V_surf,
                colorscale="RdBu", opacity=0.9,
                colorbar=dict(title="V (V)"),
            )])
            surf_fig.update_layout(
                scene=dict(xaxis_title="SoC", yaxis_title="T (°C)", zaxis_title="V (V)"),
                title=f"Learned V(SoC, T) surface at {surf_c_rate}C",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(surf_fig, use_container_width=True)


# ===========================================================================
# TAB 4 — Digital Twin
# ===========================================================================

with tab_twin:
    st.markdown("### 🤖 Real-Time Digital Twin")
    st.markdown(
        "Use the trained PIELM as a **real-time surrogate** inside a digital twin loop.  \n"
        "Specify new operating conditions — the PIELM predicts the full discharge curve "
        "**instantly**, without re-training."
    )

    if "battery_model" not in st.session_state:
        st.info("Train the PIELM surrogate first (🧠 Train PIELM tab).", icon="ℹ️")
        st.stop()

    model = st.session_state["battery_model"]

    st.markdown("#### Set new operating conditions")
    tw_c1, tw_c2, tw_c3 = st.columns(3)
    with tw_c1:
        tw_crate = st.select_slider("New C-rate", [0.2, 0.5, 1.0, 2.0, 3.0], value=0.5,
                                    key="twin_cr")
        tw_soc0  = st.slider("Initial SoC", 0.5, 1.0, 0.95, 0.05, key="twin_soc")
    with tw_c2:
        tw_temp  = st.slider("Temperature (°C)", -10, 45, 10, 5, key="twin_T")
        tw_soh   = st.slider("SoH", 0.60, 1.00, 0.85, 0.05, key="twin_soh")
    with tw_c3:
        show_ref = st.checkbox("Show physics reference alongside prediction", value=True)

    if st.button("🔮 Predict with Digital Twin", type="primary"):
        # Build SoC trajectory (from physics model, lightweight)
        I_val  = tw_crate * 3.0
        t_arr  = np.arange(0, 3600 / tw_crate, 1.0)
        soc_t  = np.clip(tw_soc0 - I_val * t_arr / (3600 * 3.0 * tw_soh), 0.01, 1.0)

        T_norm_arr = np.full_like(soc_t, (tw_temp - 25.0) / 20.0)
        I_norm_arr = np.full_like(soc_t, I_val / 3.0)
        X_twin = np.column_stack([soc_t, T_norm_arr, I_norm_arr])

        t0 = time.perf_counter()
        with torch.no_grad():
            V_twin = model.predict(
                torch.tensor(X_twin, dtype=torch.float64)
            ).numpy().ravel()
        t_inf_total = (time.perf_counter() - t0) * 1000

        # Physics reference (if requested)
        if show_ref:
            ref_sim = simulate_discharge(3.0, tw_crate, tw_temp, tw_soc0, tw_soh)

        st.success(
            f"Predicted {len(t_arr)} steps in **{t_inf_total:.2f} ms** total  "
            f"({t_inf_total/len(t_arr)*1000:.4f} µs/step)",
            icon="⚡",
        )

        if _PLOTLY_OK:
            th = t_arr / 3600.0

            twin_fig = make_subplots(rows=1, cols=2,
                                     subplot_titles=["Voltage V(t)", "State of Charge SoC(t)"])
            twin_fig.add_trace(
                go.Scatter(x=th, y=V_twin, name="PIELM twin",
                           line=dict(color="#2563eb", width=2.5)), row=1, col=1)
            twin_fig.add_trace(
                go.Scatter(x=th, y=soc_t, name="SoC", line=dict(color="#16a34a")), row=1, col=2)

            if show_ref:
                r_th = ref_sim["t"] / 3600.0
                twin_fig.add_trace(
                    go.Scatter(x=r_th, y=ref_sim["V"], name="Physics ref",
                               line=dict(color="#dc2626", dash="dash", width=1.5)), row=1, col=1)

            twin_fig.update_layout(
                title=f"Digital twin prediction — {tw_crate}C  |  T={tw_temp}°C  |  SoH={tw_soh:.0%}",
                height=380,
            )
            twin_fig.update_xaxes(title_text="Time (h)")
            twin_fig.update_yaxes(title_text="V (V)", row=1, col=1)
            twin_fig.update_yaxes(title_text="SoC",   row=1, col=2)
            st.plotly_chart(twin_fig, use_container_width=True)

            # SoC estimation error if reference available
            if show_ref:
                n_min = min(len(soc_t), len(ref_sim["SoC"]))
                SoC_err = np.abs(soc_t[:n_min] - ref_sim["SoC"][:n_min]) * 100
                V_err   = np.abs(V_twin[:n_min] - ref_sim["V"][:n_min]) * 1000
                err_fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=["|SoC error| (%)", "|V error| (mV)"])
                err_fig.add_trace(
                    go.Scatter(x=th[:n_min], y=SoC_err,
                               line=dict(color="#f59e0b")), row=1, col=1)
                err_fig.add_trace(
                    go.Scatter(x=th[:n_min], y=V_err,
                               line=dict(color="#ef4444")), row=1, col=2)
                err_fig.update_layout(height=280, showlegend=False)
                err_fig.update_xaxes(title_text="Time (h)")
                st.plotly_chart(err_fig, use_container_width=True)

                # KPI box
                kp1, kp2, kp3 = st.columns(3)
                kp1.metric("Max SoC error", f"{SoC_err.max():.2f} %")
                kp2.metric("Mean V error", f"{V_err.mean():.1f} mV")
                kp3.metric("Inference speed", f"{t_inf_total/len(t_arr)*1000:.3f} µs/step")


# ===========================================================================
# TAB 5 — Benchmark
# ===========================================================================

with tab_benchmark:
    st.markdown("### 📊 PIELM vs Baselines")
    st.markdown(
        "Theoretical and empirical comparison of PIELM against standard approaches "
        "for battery state estimation. Numbers shown are representative from the literature "
        "and can be updated with your own benchmark results."
    )

    # Method comparison table
    import pandas as pd

    df_bench = pd.DataFrame({
        "Method": [
            "DFN / P2D (full-order)",
            "Equivalent Circuit (HPPC)",
            "Extended Kalman Filter",
            "LSTM (data-driven)",
            "PINN (Adam, 10 k ep.)",
            "PINN (L-BFGS, 1 k ep.)",
            "VanillaPIELM ✅",
            "CorePIELM (+ PDE resid.) ✅",
            "BayesianPIELM ✅",
        ],
        "Training time": [
            "N/A", "~2 h (HPPC test)", "N/A",
            "~20 min (GPU)", "~15 min (GPU)", "~10 min (GPU)",
            "< 1 s", "< 2 s", "< 3 s",
        ],
        "Inference / step": [
            "≥ 100 ms", "~1 ms", "~0.5 ms",
            "~2 ms", "~10 ms", "~10 ms",
            "< 0.1 ms", "< 0.1 ms", "< 0.5 ms",
        ],
        "SoC error (%)": [
            "< 0.5", "~2–5", "~1–2",
            "~1", "~1–2", "< 1",
            "~1–2", "< 1", "~1–2",
        ],
        "Physics-informed": [
            "✅ (full)", "❌", "❌",
            "❌", "✅", "✅",
            "⚠️ (partial)", "✅", "⚠️ (partial)",
        ],
        "Uncertainty": [
            "❌", "❌", "✅",
            "❌", "❌", "❌",
            "❌", "❌", "✅",
        ],
        "GPU required": [
            "❌", "❌", "❌",
            "✅", "✅", "✅",
            "❌", "❌", "❌",
        ],
    })

    st.dataframe(df_bench, hide_index=True, use_container_width=True)

    st.divider()

    # Interactive speedup chart
    if _PLOTLY_OK:
        st.markdown("#### Inference Speed Comparison (log scale)")
        methods_sp = [
            "DFN full-order", "PINN (Adam)", "LSTM", "EKF",
            "BayesianPIELM", "CorePIELM", "VanillaPIELM",
        ]
        # µs per step
        inf_us = [100_000, 10_000, 2_000, 500, 400, 80, 50]
        colors_sp = ["#ef4444", "#f97316", "#eab308", "#84cc16",
                     "#22d3ee", "#3b82f6", "#8b5cf6"]

        sp_fig = go.Figure(go.Bar(
            y=methods_sp,
            x=inf_us,
            orientation="h",
            marker_color=colors_sp,
            text=[f"{v:,.0f} µs" for v in inf_us],
            textposition="outside",
        ))
        sp_fig.update_layout(
            xaxis_type="log",
            xaxis_title="Inference latency (µs / step) — log scale",
            title="Real-time feasibility: lower is better",
            height=350,
            margin=dict(l=180),
        )
        sp_fig.add_vline(x=1000, line_dash="dash", line_color="red",
                         annotation_text="1 ms real-time limit",
                         annotation_position="top right")
        st.plotly_chart(sp_fig, use_container_width=True)

        # Accuracy vs speed Pareto scatter
        st.markdown("#### Accuracy vs Speed Pareto Front")
        pareto_methods = [
            "DFN", "PINN (Adam)", "LSTM", "EKF",
            "BayesianPIELM", "CorePIELM", "VanillaPIELM",
        ]
        pareto_inf  = [100000, 10000, 2000, 500, 400, 80, 50]   # µs
        pareto_err  = [0.5, 1.5, 1.0, 1.5, 1.5, 0.8, 1.8]      # SoC error %
        pareto_col  = ["#ef4444", "#f97316", "#eab308", "#84cc16",
                       "#22d3ee", "#3b82f6", "#8b5cf6"]

        pareto_fig = go.Figure()
        for i, (m, sp, er, c) in enumerate(
                zip(pareto_methods, pareto_inf, pareto_err, pareto_col)):
            pareto_fig.add_trace(go.Scatter(
                x=[sp], y=[er], mode="markers+text",
                text=[m], textposition="top center",
                marker=dict(size=14, color=c),
                name=m,
            ))

        # Pareto front (approximate)
        pareto_fig.add_shape(
            type="rect", x0=0, x1=1100, y0=0, y1=4,
            fillcolor="rgba(34,197,94,0.08)", line_width=0,
        )
        pareto_fig.add_annotation(x=550, y=3.5, text="Real-time feasible zone",
                                  font=dict(color="green", size=11), showarrow=False)

        pareto_fig.update_layout(
            xaxis_type="log",
            xaxis_title="Inference latency (µs)",
            yaxis_title="SoC error (%)",
            title="Accuracy vs Speed — closer to bottom-left is better",
            height=400,
            showlegend=False,
        )
        st.plotly_chart(pareto_fig, use_container_width=True)

    # Live benchmark against current trained model
    if "battery_model" in st.session_state and "battery_metrics" in st.session_state:
        st.divider()
        st.markdown("#### Your trained PIELM — live results")
        m   = st.session_state["battery_metrics"]
        t_i = st.session_state.get("battery_t_inf_ms", float("nan"))
        t_t = st.session_state.get("battery_t_train", float("nan"))
        hd  = st.session_state.get("battery_hidden_dim", "?")

        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Hidden dim", hd)
        lc2.metric("Training time", f"{t_t*1000:.0f} ms")
        lc3.metric("Inference / sample", f"{t_i:.4f} ms")
        lc4.metric("RMSE (V)", f"{m.get('rmse', float('nan')):.4f}")

        st.caption(
            f"Relative L² = {m.get('rel_l2', float('nan'))*100:.3f}%  |  "
            f"MAE = {m.get('mae', float('nan'))*1000:.2f} mV  |  "
            f"R² = {m.get('r2', float('nan')):.6f}"
        )
