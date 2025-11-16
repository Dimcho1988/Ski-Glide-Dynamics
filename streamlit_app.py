import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess


G = 9.80665  # гравитационно ускорение


def parse_tcx(file_obj) -> pd.DataFrame:
    """Чете TCX и връща DataFrame с време, дистанция, височина."""
    # ако идва от Streamlit, file_obj е BytesIO-like
    data = file_obj.read()
    tree = ET.parse(io.BytesIO(data))
    root = tree.getroot()

    ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    times, dist, alt = [], [], []

    for tp in root.findall('.//tcx:Trackpoint', ns):
        t_el = tp.find('tcx:Time', ns)
        d_el = tp.find('tcx:DistanceMeters', ns)
        a_el = tp.find('tcx:AltitudeMeters', ns)

        if t_el is None or d_el is None or a_el is None:
            continue

        times.append(t_el.text)
        dist.append(float(d_el.text))
        alt.append(float(a_el.text))

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "distance_m": dist,
        "altitude_m": alt
    })

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # сурова скорост (м/с)
    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0)

    return df


def central_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Централна разлика за по-гладка производна."""
    dy = np.zeros_like(y)
    dx = np.zeros_like(y)

    # вътрешни точки
    dy[1:-1] = y[2:] - y[:-2]
    dx[1:-1] = x[2:] - x[:-2]

    # краища – напред/назад
    dy[0] = y[1] - y[0]
    dx[0] = x[1] - x[0]
    dy[-1] = y[-1] - y[-2]
    dx[-1] = x[-1] - x[-2]

    dx[dx == 0] = np.nan
    return dy / dx


def process_df(df: pd.DataFrame, lowess_frac: float = 0.02) -> pd.DataFrame:
    """Изглаждане + ускорение + наклон върху трендовете."""
    t = df["t_sec"].values

    # тренд за височина, дистанция, скорост
    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=lowess_frac,
                              return_sorted=False)
    df["distance_smooth"] = lowess(df["distance_m"], t, frac=lowess_frac,
                                   return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=lowess_frac,
                                return_sorted=False)

    alt = df["alt_smooth"].values
    dist = df["distance_smooth"].values
    speed = df["speed_smooth"].values

    # производни по време
    dalt_dt = central_diff(alt, t)
    ddist_dt = central_diff(dist, t)
    dspeed_dt = central_diff(speed, t)

    # наклон: (dh/ds)*100 = (dh/dt)/(ds/dt)*100
    grade_raw = (dalt_dt / ddist_dt) * 100
    # изглаждаме и наклона още малко
    grade_smooth = lowess(grade_raw, t, frac=0.03, return_sorted=False)

    # ускорение – изглаждаме допълнително
    acc_raw = dspeed_dt
    acc_smooth = lowess(acc_raw, t, frac=0.05, return_sorted=False)

    df["grade_percent"] = grade_smooth
    df["acc_m_s2"] = acc_smooth

    return df


def find_downhill_segments(df: pd.DataFrame,
                           min_total_duration_s: float = 10.0,
                           cut_first_s: float = 5.0) -> pd.DataFrame:
    """
    Намира участъци с наклон < 0% (спускания), реже първите cut_first_s,
    и връща таблица със среден наклон, ускорение,
    теоретично ускорение и индекс на триене.
    """
    t = df["t_sec"].values
    grade = df["grade_percent"].values
    acc = df["acc_m_s2"].values

    downhill_mask = grade < 0  # всичко под 0%

    segments = []
    if not downhill_mask.any():
        return pd.DataFrame(columns=[
            "seg_id", "t_start", "t_end", "duration_s",
            "mean_grade_percent", "mean_acc_m_s2",
            "theoretical_acc_m_s2", "friction_index"
        ])

    # намираме границите на непрекъснатите участъци
    idx = np.where(downhill_mask)[0]
    start = idx[0]
    prev = idx[0]

    seg_id = 1

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        # приключва сегмент
        segments.append((start, prev))
        start = i
        prev = i
    # последен сегмент
    segments.append((start, prev))

    rows = []
    for (s_idx, e_idx) in segments:
        t_start = t[s_idx]
        t_end = t[e_idx]
        duration = t_end - t_start
        if duration < min_total_duration_s:
            continue  # твърде кратък

        # режем първите cut_first_s секунди от сегмента
        t_cut_start = t_start + cut_first_s
        seg_mask = (np.arange(len(t)) >= s_idx) & (np.arange(len(t)) <= e_idx)
        seg_mask = seg_mask & (t >= t_cut_start)

        if not seg_mask.any():
            continue

        seg_t = t[seg_mask]
        seg_grade = grade[seg_mask]
        seg_acc = acc[seg_mask]

        mean_grade = np.nanmean(seg_grade)
        mean_acc = np.nanmean(seg_acc)

        if np.isnan(mean_grade) or np.isnan(mean_acc):
            continue

        # теоретично ускорение (по наклона, без триене)
        # grade <0 => downhill. Вземаме абсолютната стойност.
        grade_frac = -mean_grade / 100.0
        if grade_frac <= 0:
            continue

        a_theor = G * grade_frac  # m/s^2

        # индекс на триене (по твоята идея)
        if abs(mean_acc) < 1e-3:
            friction_index = np.nan
        else:
            friction_index = a_theor / mean_acc

        rows.append({
            "seg_id": seg_id,
            "t_start": float(seg_t[0]),
            "t_end": float(seg_t[-1]),
            "duration_s": float(seg_t[-1] - seg_t[0]),
            "mean_grade_percent": mean_grade,
            "mean_acc_m_s2": mean_acc,
            "theoretical_acc_m_s2": a_theor,
            "friction_index": friction_index
        })
        seg_id += 1

    return pd.DataFrame(rows)


# ============ STREAMLIT UI ============

def main():
    st.title("onFlows – тренд, наклон, ускорение и индекс на триене (TCX)")

    st.markdown(
        "Качи един или няколко **TCX файла** (ролки/ски бягане). "
        "За всеки файл се изчислява изгладена скорост, наклон, ускорение "
        "и участъци със спускане за оценка на триенето."
    )

    weight_kg = st.number_input(
        "Тегло на спортиста (kg) – засега се използва само информативно",
        min_value=30.0, max_value=120.0, value=70.0, step=1.0
    )

    uploaded_files = st.file_uploader(
        "Избери TCX файлове",
        type=["tcx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл, за да продължим.")
        return

    results = {}  # filename -> (df, seg_df)

    for file in uploaded_files:
        try:
            df_raw = parse_tcx(file)
            df_proc = process_df(df_raw)
            seg_df = find_downhill_segments(df_proc)

            results[file.name] = (df_proc, seg_df)
        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    # избор на файл за визуализация
    file_names = list(results.keys())
    selected_name = st.selectbox("Избери файл за визуализация", file_names)

    df_proc, seg_df = results[selected_name]

    st.subheader(f"Графики – {selected_name}")

    # скорост
    st.markdown("**Скорост (сурова vs изгладена)**")
    speed_plot_df = df_proc[["t_sec", "speed", "speed_smooth"]].set_index("t_sec")
    st.line_chart(speed_plot_df)

    # наклон
    st.markdown("**Наклон (%) – изгладен**")
    grade_plot_df = df_proc[["t_sec", "grade_percent"]].set_index("t_sec")
    st.line_chart(grade_plot_df)

    # ускорение
    st.markdown("**Ускорение (m/s²) – изгладено**")
    acc_plot_df = df_proc[["t_sec", "acc_m_s2"]].set_index("t_sec")
    st.line_chart(acc_plot_df)

    st.subheader("Участъци със спускане и индекс на триене")

    if seg_df.empty:
        st.info("Не са намерени достатъчно дълги участъци с наклон < 0%.")
    else:
        st.dataframe(seg_df.style.format({
            "duration_s": "{:.1f}",
            "mean_grade_percent": "{:.2f}",
            "mean_acc_m_s2": "{:.3f}",
            "theoretical_acc_m_s2": "{:.3f}",
            "friction_index": "{:.3f}"
        }))

    # бутон за сваляне на пълните данни
    st.subheader("Експорт на тренд-данните")

    csv_bytes = df_proc.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Свали CSV за избрания файл (тренд, наклон, ускорение)",
        data=csv_bytes,
        file_name=selected_name.replace(".tcx", "_trend_grade_acc.csv"),
        mime="text/csv"
    )

    st.caption(
        "Засега индексът на триене е дефиниран като a_theoretical / a_real. "
        "По-късно можем да го мапнем към модулирана скорост и референтни условия."
    )


if __name__ == "__main__":
    main()
