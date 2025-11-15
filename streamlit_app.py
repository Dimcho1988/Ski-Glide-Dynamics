
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from tcx_utils import process_tcx_file

st.set_page_config(page_title="onFlows – Friction & Speed Modulation", layout="wide")

st.title("onFlows – Модулиране на скоростта по наклон (ролки / ски)")

st.markdown(
    """
Тук можеш да качиш няколко **TCX файла**, да избереш **референтна активност** и да
видиш сравнение между реалната и *модулираната* скорост спрямо референтната.
"""
)

with st.sidebar:
    st.header("Параметри за модела")
    mass = st.number_input("Тегло на спортиста (kg)", min_value=30.0, max_value=120.0, value=72.0, step=1.0)
    cda = st.number_input("CdA (m²)", min_value=0.10, max_value=0.80, value=0.35, step=0.01)
    rho = st.number_input("Плътност на въздуха ρ (kg/m³)", min_value=0.8, max_value=1.5, value=1.2, step=0.05)

    st.markdown("---")
    st.caption("Сегменти за спускане:")
    min_grade = st.number_input("Макс. наклон (≤) за спускане (%)", value=-5.0, step=0.5)
    min_segment_time = st.number_input("Мин. продължителност на целия сегмент (s)", value=15, step=5)
    cut_head_time = st.number_input("Отрязване на началото на сегмента (s)", value=5, step=1)
    min_remaining_time = st.number_input("Мин. оставащо време след отрязването (s)", value=10, step=5)

uploaded_files = st.file_uploader(
    "Качи един или повече TCX файла", type=["tcx"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# Обработка на всички активности
activities = []
for f in uploaded_files:
    try:
        df, summary = process_tcx_file(
            f,
            mass=mass,
            cda=cda,
            rho=rho,
            min_down_grade=min_grade / 100.0,  # превръщаме в относителен наклон
            min_segment_time=min_segment_time,
            cut_head_time=cut_head_time,
            min_remaining_time=min_remaining_time,
        )
        activities.append({"name": f.name, "df": df, "summary": summary})
    except Exception as e:
        st.error(f"Грешка при обработка на файла {f.name}: {e}")

if not activities:
    st.error("Няма успешно обработени активности.")
    st.stop()

# Списък за избор
names = [a["name"] for a in activities]

col1, col2 = st.columns(2)

with col1:
    ref_name = st.selectbox("Избери референтна активност", names, index=0)
with col2:
    view_name = st.selectbox("Коя активност да визуализираме?", names, index=0)

# Търсим референтните стойности
ref_activity = next(a for a in activities if a["name"] == ref_name)
ref_mu = ref_activity["summary"]["median_mu_valid"]
ref_mu_clipped = float(np.clip(ref_mu, 0.005, 1.0))

st.subheader("Обобщение по активности")

summary_rows = []
for a in activities:
    s = a["summary"]
    summary_rows.append(
        {
            "Файл": a["name"],
            "Точки в спускане": s["n_valid_down"],
            "Среден наклон в спускане (%)": s["mean_grade_valid"] * 100 if not np.isnan(s["mean_grade_valid"]) else np.nan,
            "Median μ (валидни)": s["median_mu_valid"],
        }
    )

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

st.markdown(
    f"**Референтна активност:** `{ref_name}`  → median μ = {ref_mu:.4f}"
)

# Визуализиране на избраната активност
view_activity = next(a for a in activities if a["name"] == view_name)
df_view = view_activity["df"].copy()
mu_view = view_activity["summary"]["median_mu_valid"]
mu_view_clipped = float(np.clip(mu_view, 0.005, 1.0))

# Фактор за скалиране на скоростта спрямо референтната
scale_factor = (mu_view_clipped / ref_mu_clipped) ** -1.0
df_view["speed_mod"] = df_view["speed_smooth"] * scale_factor

st.subheader(f"Скорост – реална vs. модулирана ({view_name})")

# Подготвяме данните за графиката
plot_df = df_view[["t_sec", "speed_smooth", "speed_mod"]].copy()
plot_df = plot_df.melt(id_vars="t_sec", var_name="type", value_name="speed")

chart = (
    alt.Chart(plot_df)
    .mark_line()
    .encode(
        x=alt.X("t_sec", title="Време (s)"),
        y=alt.Y("speed", title="Скорост (m/s)"),
        color=alt.Color("type", title="Тип", scale=alt.Scale(domain=["speed_smooth", "speed_mod"],
                                                           range=["#1f77b4", "#ff7f0e"]),
                        legend=alt.Legend(labelExpr="datum.label == 'speed_smooth' ? 'Реална скорост' : 'Модулирана скорост'")),
        tooltip=[
            alt.Tooltip("t_sec", title="Време (s)", format=".1f"),
            alt.Tooltip("speed", title="Скорост (m/s)", format=".2f"),
            alt.Tooltip("type", title="Тип"),
        ],
    )
    .properties(height=400)
)

st.altair_chart(chart, use_container_width=True)

st.markdown(
    """
**Забележка:** Модулираната скорост е получена чрез скалиране на изгладената скорост
спрямо отношението на ефективния коефициент на триене μ между избраната и
референтната активност. Това е приближена линия на тренда, подходяща за
сравнение на трасета и оборудване, а не за прецизни физични симулации.
"""
)
