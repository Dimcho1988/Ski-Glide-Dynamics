
import streamlit as st
import pandas as pd
import numpy as np

from friction_model import (
    process_activity_file,
    compute_friction_indices_and_modulation,
)

# ---------------- UI CONFIG ---------------- #

st.set_page_config(
    page_title="onFlows — Коефициент на триене",
    layout="wide",
)

st.title("onFlows — Коефициент на триене и модулирана скорост")
st.markdown(
    """
Това е демо приложение на модела за коефициент на триене и модулиране на скоростта
с **фиксирана референтна активност**.

1. Качваш една или повече активности (.tcx или .csv).  
2. Избираш коя да бъде **референтна**.  
3. Приложението изчислява μ_eff за всяка активност и **Friction Index** спрямо референтната.  
4. Виждаш реална и модулирана скорост върху графика.
"""
)

# ---------------- SIDEBAR – ПАРАМЕТРИ ---------------- #

with st.sidebar:
    st.header("Настройки на модела")

    S_thr_percent = st.slider(
        "Праг за наклон S_thr (%) (надолу):",
        min_value=-10.0,
        max_value=-0.5,
        value=-2.0,
        step=0.1,
        help="Всички секунди със slope < S_thr и достатъчна скорост се считат за спускане.",
    )
    v_min = st.slider(
        "Минимална скорост v_min (m/s):",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Под тази скорост секундата не се счита за свободно плъзгане.",
    )
    T_cut = st.slider(
        "Отрязване в началото на сегмента T_cut (s):",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        help="Премахваме първите секунди от всеки сегмент заради инерция от оттласкване.",
    )
    Tmin_free = st.slider(
        "Минимална дължина на свободно плъзгане Tmin_free (s):",
        min_value=1,
        max_value=60,
        value=5,
        step=1,
        help="Сегменти по-къси от това се игнорират.",
    )
    mu_min = st.number_input(
        "Мин. физично допустимо μ_eff:",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    mu_max = st.number_input(
        "Макс. физично допустимо μ_eff:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
    )
    delta_up_percent = st.slider(
        "Макс. повишаване на скоростта Δmax,↑ (%):",
        min_value=0,
        max_value=100,
        value=20,
        step=1,
        help="Горна граница за увеличаване на скоростта при тежък сняг.",
    )
    delta_down_percent = st.slider(
        "Макс. намаляване на скоростта Δmax,↓ (%):",
        min_value=0,
        max_value=100,
        value=15,
        step=1,
        help="Горна граница за намаляване на скоростта при много бърз сняг.",
    )

    delta_up = delta_up_percent / 100.0
    delta_down = delta_down_percent / 100.0

    st.markdown("---")
    st.caption(
        "Очакван формат на CSV: колони time (s), elevation (m), distance (m). "
        "TCX файловете трябва да съдържат Trackpoint с Time, AltitudeMeters, DistanceMeters."
    )

# ---------------- КАЧВАНЕ НА ДАННИ ---------------- #

uploaded_files = st.file_uploader(
    "Качи една или повече активности (.tcx или .csv):",
    type=["tcx", "csv"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Моля, качи поне една активност, за да започнем.")
    st.stop()

# ---------------- ОБРАБОТКА НА ВСИЧКИ АКТИВНОСТИ ---------------- #

activities = {}
errors = []

for file in uploaded_files:
    try:
        act = process_activity_file(
            file,
            filename=file.name,
            S_thr_percent=S_thr_percent,
            v_min=v_min,
            T_cut=T_cut,
            Tmin_free=Tmin_free,
            mu_min=mu_min,
            mu_max=mu_max,
        )
        activities[file.name] = act
    except Exception as e:
        errors.append(f"{file.name}: {e}")

if errors:
    with st.expander("⚠️ Пропуснати/проблемни файлове"):
        for msg in errors:
            st.write(msg)

if not activities:
    st.error("Нито една активност не беше обработена успешно.")
    st.stop()

activity_names = list(activities.keys())

# ---------------- ИЗБОР НА РЕФЕРЕНТНА И ЦЕЛЕВА АКТИВНОСТ ---------------- #

st.subheader("Избор на референтна и целева активност")

ref_name = st.selectbox(
    "Референтна активност (μ_ref):",
    options=activity_names,
    index=0,
)

target_name = st.selectbox(
    "Целева активност за визуализация:",
    options=activity_names,
    index=min(1, len(activity_names) - 1),
)

# ---------------- ИЗЧИСЛЕНИЕ НА FI И МОДУЛИРАНЕ ---------------- #

compute_friction_indices_and_modulation(
    activities=activities,
    ref_name=ref_name,
    delta_up=delta_up,
    delta_down=delta_down,
)

# Таблица с резултати
rows = []
for name, act in activities.items():
    rows.append(
        {
            "Активност": name,
            "μ_session": act["mu_session"],
            "FI (μ_session / μ_ref)": act["FI"],
            "K (модулация)": act["K"],
            "K (%)": (act["K"] - 1.0) * 100.0,
            "Валидни секунди (free glide)": int(act["n_valid"]),
        }
    )

results_df = pd.DataFrame(rows)
st.subheader("Обобщение по активности")
st.dataframe(results_df.style.format({"μ_session": "{:.4f}", "FI (μ_session / μ_ref)": "{:.3f}", "K": "{:.3f}", "K (%)": "{:.1f}"}))

# ---------------- ГРАФИКА ЗА ИЗБРАНАТА АКТИВНОСТ ---------------- #

st.subheader("Реална и модулирана скорост за избраната активност")

if target_name not in activities:
    st.error("Избраната целева активност не е налична.")
    st.stop()

act = activities[target_name]
df = act["df"]

if "v_mod" not in df.columns or df["v_mod"].isna().all():
    st.error("Няма налична модулирана скорост за избраната активност.")
    st.stop()

tab1, tab2 = st.tabs(["Графика", "Данни"])

with tab1:
    import plotly.express as px

    plot_df = df[["time_s", "v", "v_mod"]].copy()
    plot_df = plot_df.rename(
        columns={
            "time_s": "Време (s)",
            "v": "Скорост v (реална, m/s)",
            "v_mod": "Скорост v_mod (модулирана, m/s)",
        }
    )

    long_df = plot_df.melt(
        id_vars=["Време (s)"],
        value_vars=["Скорост v (реална, m/s)", "Скорост v_mod (модулирана, m/s)"],
        var_name="Тип скорост",
        value_name="Стойност (m/s)",
    )

    fig = px.line(
        long_df,
        x="Време (s)",
        y="Стойност (m/s)",
        color="Тип скорост",
        title=f"Реална и модулирана скорост: {target_name}",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(df[["time_s", "v", "v_mod"]].head(500))
    st.caption("Показани са първите 500 реда за удобство.")

st.markdown(
    """
**Забележка:** Това е примерна имплементация за тестване на модела в реални данни.
За продукционна среда (хиляди потребители) ще е нужно оптимизиране, кеширане и
отделно бекенд ниво за обработка.
"""
)
