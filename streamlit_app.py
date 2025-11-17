import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

# ----------------- Помощни функции ----------------- #

def parse_tcx(file_obj) -> pd.DataFrame:
    """Чете TCX файл и връща DataFrame с време, дистанция, височина и сурова скорост."""
    data = file_obj.read()
    tree = ET.parse(io.BytesIO(data))
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    times, dist, alt = [], [], []

    for tp in root.findall(".//tcx:Trackpoint", ns):
        t_el = tp.find("tcx:Time", ns)
        d_el = tp.find("tcx:DistanceMeters", ns)
        a_el = tp.find("tcx:AltitudeMeters", ns)

        if t_el is None or d_el is None or a_el is None:
            continue

        times.append(t_el.text)
        dist.append(float(d_el.text))
        alt.append(float(a_el.text))

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "distance_m": dist,
            "altitude_m": alt,
        }
    )

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)

    return df


def central_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Централна разлика – по-гладка производна."""
    dy = np.zeros_like(y)
    dx = np.zeros_like(y)

    # вътрешни точки
    dy[1:-1] = y[2:] - y[:-2]
    dx[1:-1] = x[2:] - x[:-2]

    # краища
    dy[0] = y[1] - y[0]
    dx[0] = x[1] - x[0]
    dy[-1] = y[-1] - y[-2]
    dx[-1] = x[-1] - x[-2]

    dx[dx == 0] = np.nan
    return dy / dx


def process_df(df: pd.DataFrame, lowess_frac: float = 0.02) -> pd.DataFrame:
    """
    Изглаждане на данните и смятане на наклон и ускорение ИЗЦЯЛО върху тренда.
    """
    t = df["t_sec"].values

    # тренд линии
    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=lowess_frac, return_sorted=False)
    df["distance_smooth"] = lowess(df["distance_m"], t, frac=lowess_frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=lowess_frac, return_sorted=False)

    alt = df["alt_smooth"].values
    dist = df["distance_smooth"].values
    speed = df["speed_smooth"].values

    # производни по време
    dalt_dt = central_diff(alt, t)
    ddist_dt = central_diff(dist, t)
    dspeed_dt = central_diff(speed, t)

    # наклон по тренда: (dh/ds)*100 = (dh/dt)/(ds/dt)*100
    grade_raw = (dalt_dt / ddist_dt) * 100.0
    df["grade_percent"] = lowess(grade_raw, t, frac=0.03, return_sorted=False)

    # ускорение по тренда
    acc_raw = dspeed_dt
    df["acc_m_s2"] = lowess(acc_raw, t, frac=0.05, return_sorted=False)

    return df


def find_downhill_segments(
    df: pd.DataFrame,
    min_total_duration_s: float = 10.0,
    cut_first_s: float = 5.0,
) -> pd.DataFrame:
    """
    Намира сегменти с наклон < 0% (спускания),
    реже първите cut_first_s секунди,
    и дава таблица със САМО реалното ускорение и наклон (средни за сегмента).
    """
    t = df["t_sec"].values
    grade = df["grade_percent"].values
    acc = df["acc_m_s2"].values

    downhill_mask = grade < 0.0
    segments = []

    if not downhill_mask.any():
        return pd.DataFrame(
            columns=[
                "seg_id",
                "t_start",
                "t_end",
                "duration_s",
                "mean_grade_percent",
                "mean_acc_m_s2",
            ]
        )

    idx = np.where(downhill_mask)[0]
    start = idx[0]
    prev = idx[0]
    seg_id = 1

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((start, prev))
        start = i
        prev = i
    segments.append((start, prev))

    rows = []
    for (s_idx, e_idx) in segments:
        t_start = t[s_idx]
        t_end = t[e_idx]
        duration = t_end - t_start
        if duration < min_total_duration_s:
            continue

        # режем първите cut_first_s секунди
        t_cut_start = t_start + cut_first_s
        full_idx = np.arange(len(t))
        seg_mask = (full_idx >= s_idx) & (full_idx <= e_idx) & (t >= t_cut_start)

        if not seg_mask.any():
            continue

        seg_t = t[seg_mask]
        seg_grade = grade[seg_mask]
        seg_acc = acc[seg_mask]

        mean_grade = np.nanmean(seg_grade)
        mean_acc = np.nanmean(seg_acc)

        if np.isnan(mean_grade) or np.isnan(mean_acc):
            continue

        rows.append(
            {
                "seg_id": seg_id,
                "t_start": float(seg_t[0]),
                "t_end": float(seg_t[-1]),
                "duration_s": float(seg_t[-1] - seg_t[0]),
                "mean_grade_percent": mean_grade,
                "mean_acc_m_s2": mean_acc,
            }
        )
        seg_id += 1

    return pd.DataFrame(rows)


# ----------------- STREAMLIT UI ----------------- #

def main():
    st.title("onFlows — Реално ускорение и наклон в спусканията")

    st.markdown(
        """
Тази версия:
- Изглажда скорост, височина и дистанция (LOWESS тренд)
- Изчислява наклон (%) и ускорение **по тренда**
- Намира само участъци със спускане (**grade < 0%**)
- Прилага старите критерии: минимум 10 s, маха първите 5 s от всеки участък
- Показва само **реално ускорение и наклон** за тези сегменти
"""
    )

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл.")
        return

    # Параметри на сегментите – можеш да ги променяш от UI
    st.sidebar.header("Критерии за спусканията")
    min_duration = st.sidebar.number_input(
        "Минимална дължина на сегмента (s)",
        min_value=5.0,
        max_value=120.0,
        value=10.0,
        step=1.0,
    )
    cut_first = st.sidebar.number_input(
        "Изрязване на първите (s) от всеки сегмент",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=1.0,
    )

    results = {}  # filename -> (df_proc, seg_df)

    for file in uploaded_files:
        try:
            df_raw = parse_tcx(file)
            df_proc = process_df(df_raw)
            seg_df = find_downhill_segments(df_proc, min_duration, cut_first)
            results[file.name] = (df_proc, seg_df)
        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    file_names = list(results.keys())
    if not file_names:
        st.error("Няма успешно обработени файлове.")
        return

    selected_name = st.selectbox("Избери файл", file_names)
    df_proc, seg_df = results[selected_name]

    st.subheader(f"Графики – {selected_name}")

    # Скорост
    st.markdown("**Скорост (сурова vs изгладена)**")
    speed_plot_df = df_proc[["t_sec", "speed", "speed_smooth"]].set_index("t_sec")
    st.line_chart(speed_plot_df)

    # Наклон
    st.markdown("**Наклон (%) – по тренда**")
    grade_plot_df = df_proc[["t_sec", "grade_percent"]].set_index("t_sec")
    st.line_chart(grade_plot_df)

    # Ускорение
    st.markdown("**Ускорение (m/s²) – по тренда**")
    acc_plot_df = df_proc[["t_sec", "acc_m_s2"]].set_index("t_sec")
    st.line_chart(acc_plot_df)

    st.subheader("Сегменти със спускане – реално ускорение и наклон")

    if seg_df.empty:
        st.info(
            "Не са намерени сегменти, които да отговарят на критериите "
            f"(наклон < 0%, дължина ≥ {min_duration}s)."
        )
    else:
        st.dataframe(
            seg_df.style.format(
                {
                    "duration_s": "{:.1f}",
                    "mean_grade_percent": "{:.2f}",
                    "mean_acc_m_s2": "{:.3f}",
                }
            )
        )

    # Експорт на пълните изгладени данни
    st.subheader("Експорт на изгладените данни за избрания файл")
    csv_bytes = df_proc.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Свали CSV (trend, grade, acceleration)",
        data=csv_bytes,
        file_name=selected_name.replace(".tcx", "_trend_grade_acc.csv"),
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

