import streamlit as st
import pandas as pd
import numpy as np

from friction_model import (
    parse_tcx_to_df,
    compute_session_friction,
    apply_reference_and_modulation,
)

st.set_page_config(page_title="Ski Glide Dynamics", layout="wide")

st.title("Ski Glide Dynamics — Friction & Speed Normalization Model")

st.sidebar.header("Филтър за свободно плъзгане")

slope_thr = st.sidebar.number_input("Минимален наклон (%)", value=-5.0, step=0.5)
v_min = st.sidebar.number_input("Минимална скорост (m/s)", value=2.0, step=0.1)
t_cut = st.sidebar.number_input("Изрязване на първите T_cut (s)", value=5, step=1)
t_min_free = st.sidebar.number_input("Минимум свободно плъзгане Tmin_free (s)", value=10, step=1)
trend_window = st.sidebar.number_input("Trend window за изглаждане (s)", value=31, step=2)
a_max = st.sidebar.number_input("Максимално ускорение |a| (m/s²)", value=0.4, step=0.1)

st.sidebar.header("Ограничения на модулацията")
delta_max_up = st.sidebar.number_input("Макс. повишаване на скоростта Δmax↑ (%)", value=20.0)
delta_max_down = st.sidebar.number_input("Макс. намаляване на скоростта Δmax↓ (%)", value=15.0)

uploaded_files = st.file_uploader(
    "Качи един или повече TCX файла", type=["tcx"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл.")
    st.stop()

activities = []

for uf in uploaded_files:
    df = parse_tcx_to_df(uf)
    if df is None or df.empty:
        st.warning(f"⚠ {uf.name}: проблем с данните")
        continue

    res = compute_session_friction(
        df,
        slope_thr=slope_thr,
        v_min=v_min,
        trend_window=trend_window,
        t_cut=t_cut,
        t_min_free=t_min_free,
        a_max=a_max,
    )

    activities.append({
        "name": uf.name,
        "df": res["df"],
        "mu_session": res["mu_session"],
        "valid_seconds": res["valid_seconds"],
    })

if not activities:
    st.error("Няма нито една успешно обработена активност.")
    st.stop()

st.subheader("Избор на референтна активност")

ref_name = st.selectbox(
    "Референтна активност (μ_ref):",
    [a["name"] for a in activities]
)

mu_ref = next(
    a["mu_session"] for a in activities if a["name"] == ref_name
)

for a in activities:
    apply_reference_and_modulation(
        a,
        mu_ref=mu_ref,
        delta_max_up=delta_max_up / 100.0,
        delta_max_down=delta_max_down / 100.0,
    )

summary_rows = []

for a in activities:
    summary_rows.append({
        "Активност": a["name"],
        "μ_session": a["mu_session"],
        "FI": a.get("FI", np.nan),
        "K": a.get("K", np.nan),
        "Валидни секунди": a["valid_seconds"],
    })

summary_df = pd.DataFrame(summary_rows)

st.subheader("Обобщение по активности")
st.dataframe(summary_df.style.format({
    "μ_session": "{:.5f}",
    "FI": "{:.3f}",
    "K": "{:.3f}",
}))

st.subheader("Реална и модулирана скорост за избраната активност")

detail_name = st.selectbox(
    "Целева активност", [a["name"] for a in activities]
)

detail = next(a for a in activities if a["name"] == detail_name)
df_plot = detail["df"].copy()

df_plot["t"] = (df_plot.index - df_plot.index[0]).total_seconds()
plot_df = df_plot[["t", "v", "v_mod"]].set_index("t")

st.line_chart(plot_df)
