import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

st.set_page_config(page_title="onFlows — Glide / Resistance Index", layout="wide")


# ---------- TCX parser ----------

def parse_tcx(file_obj) -> pd.DataFrame:
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

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "distance_m": dist,
        "altitude_m": alt
    })

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)

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


def smooth_and_grade(df, frac=0.02):
    t = df["t_sec"].values

    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=frac, return_sorted=False)
    df["dist_smooth"] = lowess(df["distance_m"], t, frac=frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=frac, return_sorted=False)

    dalt_dt = central_diff(df["alt_smooth"].values, t)
    ddist_dt = central_diff(df["dist_smooth"].values, t)

    raw_grade = (dalt_dt / ddist_dt) * 100.0
    df["grade"] = lowess(raw_grade, t, frac=0.05, return_sorted=False)

    return df


def extract_downhill_points(
    df,
    grade_threshold=-1.0,
    min_segment_duration=10.0,
    cut_first=5.0,
):
    """
    Връща само точки от спускания:
    - grade < grade_threshold (напр. -1%)
    - сегменти с минимална дължина
    - от всеки сегмент се махат първите cut_first секунди
    """
    t = df["t_sec"].values
    grade = df["grade"].values

    downhill_mask = grade < grade_threshold

    if not downhill_mask.any():
        return df.iloc[0:0]  # празен DF

    idx = np.where(downhill_mask)[0]
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

        t_cut = t_start + cut_first
        seg_idx = np.arange(len(t))
        m = (seg_idx >= s_idx) & (seg_idx <= e_idx) & (t >= t_cut)
        keep_mask |= m

    return df[keep_mask]


def time_weighted_mean(series, t_sec):
    """Времево претеглена средна стойност."""
    if len(series) < 2:
        return np.nan

    t = t_sec.values
    dt = np.diff(t)
    # последната точка – ползваме последното dt
    dt = np.append(dt, dt[-1])

    weights = dt
    x = series.values

    mask = np.isfinite(x) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan

    return np.sum(x[mask] * weights[mask]) / np.sum(weights[mask])


# ---------- Streamlit UI ----------

def main():
    st.title("onFlows — Glide / Resistance Index (интуитивен модел)")

    st.markdown("""
Измерваме **плазгаемостта** на снега / ролките по *интуитивен* начин:

- Избираме само **истинските спускания** (grade < праг)  
- Махаме първите X секунди от всяко спускане  
- Вземаме **средната скорост на спусканията** и **средния наклон**  
- От тях изчисляваме:
    - **Glide Index (GI)** – колкото е по-голям, толкова по-бързо „плузга“  
    - **Resistance Index (RI = 1 / GI)** – по-голям → по-голямо съпротивление
""")

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла от една и съща писта",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл.")
        return

    st.sidebar.header("Критерии за спусканията")

    grade_threshold = st.sidebar.number_input(
        "Максимален наклон за спускане (grade < ... %)",
        min_value=-30.0, max_value=0.0, value=-1.0, step=0.5,
    )

    min_seg_dur = st.sidebar.number_input(
        "Минимална дължина на сегмента (s)",
        min_value=5.0, max_value=120.0, value=10.0, step=1.0,
    )

    cut_first = st.sidebar.number_input(
        "Изрязване на първите X секунди от всяко спускане",
        min_value=0.0, max_value=30.0, value=5.0, step=1.0,
    )

    smooth_frac = st.sidebar.slider(
        "Степен на изглаждане (LOWESS frac)",
        min_value=0.01, max_value=0.20, value=0.02, step=0.01,
    )

    results = []

    for file in uploaded_files:
        try:
            df_raw = parse_tcx(file)
            df = smooth_and_grade(df_raw, frac=smooth_frac)
            df_down = extract_downhill_points(
                df,
                grade_threshold=grade_threshold,
                min_segment_duration=min_seg_dur,
                cut_first=cut_first,
            )

            if df_down.empty:
                v_mean = np.nan
                g_mean = np.nan
                gi = np.nan
                ri = np.nan
            else:
                v_mean = time_weighted_mean(df_down["speed_smooth"], df_down["t_sec"])
                g_mean = time_weighted_mean(df_down["grade"], df_down["t_sec"])

                # v_mean в m/s → km/h за по-интуитивно
                v_mean_kmh = v_mean * 3.6

                if g_mean >= 0 or np.isnan(g_mean) or np.isnan(v_mean):
                    gi = np.nan
                    ri = np.nan
                else:
                    gi = v_mean_kmh / np.sqrt(abs(g_mean))   # Glide Index
                    ri = 1.0 / gi                             # Resistance Index

            results.append({
                "activity": file.name,
                "downhill_points": len(df_down),
                "mean_down_speed_kmh": v_mean * 3.6 if not np.isnan(v_mean) else np.nan,
                "mean_down_grade_%": g_mean,
                "GlideIndex_GI": gi,
                "ResistanceIndex_RI": ri,
            })
        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    res_df = pd.DataFrame(results)

    st.subheader("Резюме по активности")
    st.dataframe(
        res_df.style.format({
            "mean_down_speed_kmh": "{:.2f}",
            "mean_down_grade_%": "{:.2f}",
            "GlideIndex_GI": "{:.3f}",
            "ResistanceIndex_RI": "{:.3f}",
        })
    )

    # Визуализация за избрана активност
    selected = st.selectbox(
        "Избери активност за детайлна графика",
        res_df["activity"].tolist()
    )

    file_obj = next(f for f in uploaded_files if f.name == selected)
    df_raw = parse_tcx(file_obj)
    df = smooth_and_grade(df_raw, frac=smooth_frac)
    df_down = extract_downhill_points(
        df,
        grade_threshold=grade_threshold,
        min_segment_duration=min_seg_dur,
        cut_first=cut_first,
    )

    st.subheader(f"Графики — {selected}")

    with st.expander("Времеви графики (speed / grade)"):
        st.line_chart(
            df.set_index("t_sec")[["speed_smooth"]].rename(
                columns={"speed_smooth": "speed (m/s)"}
            )
        )
        st.line_chart(
            df.set_index("t_sec")[["grade"]].rename(
                columns={"grade": "grade (%)"}
            )
        )

    if not df_down.empty:
        st.markdown("**Точки от спусканията (speed vs grade):**")
        scatter_df = df_down[["grade", "speed_smooth"]].copy()
        scatter_df["speed_kmh"] = scatter_df["speed_smooth"] * 3.6
        st.scatter_chart(scatter_df, x="grade", y="speed_kmh")
    else:
        st.info("Няма намерени точки от спускане за тази активност с дадените критерии.")


if __name__ == "__main__":
    main()
