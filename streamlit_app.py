import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

st.set_page_config(page_title="onFlows — Flat Glide Index (FGNS)", layout="wide")


# ---------- TCX parser ----------

def parse_tcx_bytes(data: bytes) -> pd.DataFrame:
    """
    Парсва TCX от bytes → DataFrame с time, distance_m, altitude_m, speed (m/s).
    """
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
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)  # m/s

    return df


# ---------- helpers ----------

def central_diff(y, x):
    dy = np.zeros_like(y)
    dx = np.zeros_like(y)

    dy[1:-1] = y[2:] - y[:-2]
    dx[1:-1] = x[2:] - x[:-2]

    dy[0] = y[1] - y[0]
    dx[0] = x[1] - x[0]
    dy[-1] = y[-1] - y[-2]
    dx[-1] = x[-1] - x[-2]

    dx[dx == 0] = np.nan
    return dy / dx


def smooth_and_grade(df: pd.DataFrame, frac: float = 0.02) -> pd.DataFrame:
    """
    LOWESS изглаждане на височина, дистанция, скорост + наклон (%).
    """
    t = df["t_sec"].values

    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=frac, return_sorted=False)
    df["dist_smooth"] = lowess(df["distance_m"], t, frac=frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=frac, return_sorted=False)

    dalt_dt = central_diff(df["alt_smooth"].values, t)
    ddist_dt = central_diff(df["dist_smooth"].values, t)

    raw_grade = (dalt_dt / ddist_dt) * 100.0  # %
    df["grade"] = lowess(raw_grade, t, frac=0.05, return_sorted=False)

    return df


def extract_flat_points(
    df: pd.DataFrame,
    grade_abs_thresh: float = 0.3,
    min_segment_duration: float = 6.0,
) -> pd.DataFrame:
    """
    Връща точки от *равни* участъци:
    - |grade| < grade_abs_thresh (напр. 0.3 %)
    - всеки сегмент трябва да е с продължителност ≥ min_segment_duration
    """
    t = df["t_sec"].values
    grade = df["grade"].values

    flat_mask = np.abs(grade) < grade_abs_thresh

    if not flat_mask.any():
        return df.iloc[0:0]

    idx = np.where(flat_mask)[0]
    start = idx[0]
    prev = idx[0]
    segments = []

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((start, prev))
        start = i
        prev = i
    segments.append((start, prev))

    keep_mask = np.zeros_like(t, dtype=bool)

    for s_idx, e_idx in segments:
        t_start = t[s_idx]
        t_end = t[e_idx]
        duration = t_end - t_start
        if duration < min_segment_duration:
            continue
        seg_idx = np.arange(len(t))
        m = (seg_idx >= s_idx) & (seg_idx <= e_idx)
        keep_mask |= m

    return df[keep_mask]


def time_weighted_mean(series: pd.Series, t_sec: pd.Series) -> float:
    """
    Времево претеглена средна стойност.
    """
    if len(series) < 2:
        return np.nan

    t = t_sec.values
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    x = series.values
    w = dt

    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan

    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


# ---------- Streamlit UI ----------

def main():
    st.title("onFlows — Flat-Glide Normalized Speed (FGNS)")

    st.markdown(
        """
Модел за оценка на **плазгаемостта** само от **равните участъци**, където GPS данните
са най-стабилни.

За всяка активност:
- намираме равни сегменти (|grade| под праг)  
- изчисляваме средна скорост на тези сегменти  
- коригираме скоростта спрямо малкото остатъчно наклонче  
- получаваме **Flat-Glide Normalized Speed (FGNS)** – по-висока стойност = по-бързо плъзгане.
"""
    )

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла (желателно от една и съща писта)",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл.")
        return

    st.sidebar.header("Параметри за равните участъци")

    grade_abs_thresh = st.sidebar.number_input(
        "Максимален абсолютен наклон |grade| < (в %)",
        min_value=0.05,
        max_value=2.0,
        value=0.3,
        step=0.05,
    )

    min_seg_dur = st.sidebar.number_input(
        "Минимална дължина на равен сегмент (s)",
        min_value=3.0,
        max_value=60.0,
        value=6.0,
        step=1.0,
    )

    smooth_frac = st.sidebar.slider(
        "Степен на изглаждане (LOWESS frac)",
        min_value=0.01,
        max_value=0.20,
        value=0.02,
        step=0.01,
    )

    # коефициент за корекция на скоростта спрямо наклона (m/s на 1% grade)
    slope_coef = st.sidebar.number_input(
        "Корекция на скоростта спрямо наклон (m/s на 1% grade)",
        min_value=0.0,
        max_value=0.20,
        value=0.04,  # емпирична стойност, можеш да я нагласиш
        step=0.01,
    )

    results = {}

    for file in uploaded_files:
        try:
            data = file.read()  # четем bytes САМО веднъж
            df_raw = parse_tcx_bytes(data)
            df = smooth_and_grade(df_raw, frac=smooth_frac)
            df_flat = extract_flat_points(
                df,
                grade_abs_thresh=grade_abs_thresh,
                min_segment_duration=min_seg_dur,
            )

            if df_flat.empty:
                flat_speed = np.nan
                flat_grade = np.nan
                fgns = np.nan
            else:
                flat_speed = time_weighted_mean(
                    df_flat["speed_smooth"], df_flat["t_sec"]
                )  # m/s
                flat_grade = time_weighted_mean(
                    df_flat["grade"], df_flat["t_sec"]
                )  # %

                # коригирана скорост към идеално равно (grade=0)
                corrected_speed = df_flat["speed_smooth"] - slope_coef * df_flat["grade"]
                fgns = time_weighted_mean(corrected_speed, df_flat["t_sec"])  # m/s

            results[file.name] = {
                "df": df,
                "df_flat": df_flat,
                "flat_speed_m_s": flat_speed,
                "flat_grade_%": flat_grade,
                "FGNS_m_s": fgns,
            }

        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    # --- Резюме ---
    rows = []
    for name, obj in results.items():
        flat_speed_kmh = (
            obj["flat_speed_m_s"] * 3.6 if not np.isnan(obj["flat_speed_m_s"]) else np.nan
        )
        fgns_kmh = obj["FGNS_m_s"] * 3.6 if not np.isnan(obj["FGNS_m_s"]) else np.nan

        rows.append(
            {
                "activity": name,
                "flat_points": len(obj["df_flat"]),
                "mean_flat_speed_kmh": flat_speed_kmh,
                "mean_flat_grade_%": obj["flat_grade_%"],
                "FGNS_kmh": fgns_kmh,
                "GlideIndex_GI": fgns_kmh,
                "ResistanceIndex_RI": 1.0 / fgns_kmh if fgns_kmh and not np.isnan(fgns_kmh) else np.nan,
            }
        )

    res_df = pd.DataFrame(rows)

    st.subheader("Резюме по активности (колкото по-голям FGNS / GI → толкова по-бързо плъзгане)")
    st.dataframe(
        res_df.style.format(
            {
                "mean_flat_speed_kmh": "{:.2f}",
                "mean_flat_grade_%": "{:.2f}",
                "FGNS_kmh": "{:.2f}",
                "GlideIndex_GI": "{:.2f}",
                "ResistanceIndex_RI": "{:.3f}",
            }
        )
    )

    # --- Детайлна визуализация за избрана активност ---
    selected = st.selectbox(
        "Избери активност за детайлна графика",
        res_df["activity"].tolist(),
    )

    obj = results[selected]
    df = obj["df"]
    df_flat = obj["df_flat"]

    st.subheader(f"Графики — {selected}")

    with st.expander("Времеви графики (speed / grade)"):
        st.line_chart(
            df.set_index("t_sec")[["speed_smooth"]].rename(
                columns={"speed_smooth": "speed (m/s)"}
            )
        )
        st.line_chart(
            df.set_index("t_sec")[["grade"]].rename(columns={"grade": "grade (%)"})
        )

    if not df_flat.empty:
        st.markdown("**Точки от равните участъци (speed vs grade):**")
        scatter_df = df_flat[["grade", "speed_smooth"]].copy()
        scatter_df["speed_kmh"] = scatter_df["speed_smooth"] * 3.6
        st.scatter_chart(scatter_df, x="grade", y="speed_kmh")
    else:
        st.info(
            "За тази активност няма равни сегменти, които да отговарят на критериите."
        )


if __name__ == "__main__":
    main()
