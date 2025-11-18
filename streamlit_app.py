import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ---------------- НАСТРОЙКИ ----------------
SEGMENT_LENGTH_SEC = 15
MIN_SEGMENT_DURATION = 8.0           # реални секунди
MIN_SEGMENT_DISTANCE_M = 30.0        # минимум дистанция в сегмента
MIN_SEGMENT_SPEED_KMH = 15.0         # минимум средна скорост
MAX_ABS_SLOPE_PERCENT = 23.0         # |slope| <= 30%
MIN_ABS_DELTA_ELEV = 0.3             # за да не е само шум във височината

# филтри за суровите данни
MAX_SPEED_M_S = 30.0                 # ~108 km/h
MAX_ALT_RATE_M_S = 5.0               # макс вертикална скорост

# downhill условие за самия сегмент
MIN_DOWNHILL_SLOPE = -7.0            # среден наклон <= -7%

# условие за предходните 10 s
PRE_WINDOW_SEC = 10
MIN_PRE_WINDOW_DIST_M = 10.0         # минимум дистанция в предходния прозорец

# ---------------- TCX ПАРСВАНЕ ----------------
def parse_tcx_bytes(file_bytes: bytes) -> pd.DataFrame:
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    points = []
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        time_el = dist_el = alt_el = None
        for child in tp:
            if child.tag.endswith("Time"):
                time_el = child
            elif child.tag.endswith("DistanceMeters"):
                dist_el = child
            elif child.tag.endswith("AltitudeMeters"):
                alt_el = child

        if time_el is None:
            continue

        points.append({
            "time": pd.to_datetime(time_el.text),
            "distance_m": float(dist_el.text) if dist_el is not None else np.nan,
            "altitude_m": float(alt_el.text) if alt_el is not None else np.nan
        })

    df = pd.DataFrame(points).dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df

# ---------------- СГЛАЖДАНЕ НА ВИСОЧИНАТА ----------------
def smooth_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(3, center=True).median()
    return df

# ---------------- ЧИСТЕНЕ НА АРТЕФАКТИ ----------------
def clean_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = df["time"].diff().dt.total_seconds()
    df["ddist"] = df["distance_m"].diff()
    df["dalt"] = df["altitude_m"].diff()

    speed = df["ddist"] / df["dt"]
    alt_rate = df["dalt"] / df["dt"]

    mask = True
    mask &= (df["dt"].isna() | (df["dt"] > 0))
    mask &= (df["ddist"].isna() | (df["ddist"] >= 0))
    mask &= (speed.isna() | ((speed >= 0) & (speed <= MAX_SPEED_M_S)))
    mask &= (alt_rate.isna() | (alt_rate.abs() <= MAX_ALT_RATE_M_S))

    df = df[mask].copy().reset_index(drop=True)
    df = df[["time", "distance_m", "altitude_m"]]
    return df

# ---------------- СРЕДНА СКОРОСТ НА АКТИВНОСТТА ----------------
def compute_activity_avg_speed(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return np.nan
    t0 = df["time"].iloc[0]
    t1 = df["time"].iloc[-1]
    dt = (t1 - t0).total_seconds()
    if dt <= 0:
        return np.nan
    d0 = df["distance_m"].iloc[0]
    d1 = df["distance_m"].iloc[-1]
    dist = d1 - d0
    if dist <= 0:
        return np.nan
    speed_m_s = dist / dt
    return speed_m_s * 3.6

# ---------------- СЕГМЕНТИРАНЕ + Δv ----------------
def segment_activity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["segment_idx"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    rows = []

    for seg_idx, g in df.groupby("segment_idx"):
        g = g.sort_values("time")
        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        if duration_s < MIN_SEGMENT_DURATION:
            continue

        dist0 = g["distance_m"].iloc[0]
        dist1 = g["distance_m"].iloc[-1]
        alt0 = g["altitude_m"].iloc[0]
        alt1 = g["altitude_m"].iloc[-1]

        dist = dist1 - dist0
        dh = alt1 - alt0

        if dist <= 0:
            continue
        if dist < MIN_SEGMENT_DISTANCE_M:
            continue
        if abs(dh) < MIN_ABS_DELTA_ELEV:
            continue

        avg_speed_m_s = dist / duration_s
        avg_speed_kmh = avg_speed_m_s * 3.6
        if avg_speed_kmh < MIN_SEGMENT_SPEED_KMH:
            continue

        slope_percent = (dh / dist) * 100.0
        if abs(slope_percent) > MAX_ABS_SLOPE_PERCENT:
            continue
        # downhill условие за самия сегмент
        if slope_percent > MIN_DOWNHILL_SLOPE:
            continue  # трябва да е <= -7%

        # --------- ДОПЪЛНИТЕЛЕН ФИЛТЪР: ПРЕДИШНИ 10 s ДА СА СПУСКАНЕ ---------
        window_start = t_start - pd.Timedelta(seconds=PRE_WINDOW_SEC)
        prev = df[(df["time"] >= window_start) & (df["time"] < t_start)]
        if len(prev) < 2:
            continue

        prev_dist = prev["distance_m"].iloc[-1] - prev["distance_m"].iloc[0]
        prev_dh = prev["altitude_m"].iloc[-1] - prev["altitude_m"].iloc[0]

        if prev_dist <= MIN_PRE_WINDOW_DIST_M:
            continue  # твърде малко движение, шум
        if prev_dh >= 0:
            continue  # не е реално спускане в предишните 10 s

        # --------- НАЧАЛНА И КРАЙНА СКОРОСТ (Δv) ---------
        dt_start = (g["time"].iloc[1] - g["time"].iloc[0]).total_seconds()
        if dt_start > 0:
            v_start_m_s = (g["distance_m"].iloc[1] - g["distance_m"].iloc[0]) / dt_start
        else:
            v_start_m_s = avg_speed_m_s

        dt_end = (g["time"].iloc[-1] - g["time"].iloc[-2]).total_seconds()
        if dt_end > 0:
            v_end_m_s = (g["distance_m"].iloc[-1] - g["distance_m"].iloc[-2]) / dt_end
        else:
            v_end_m_s = avg_speed_m_s

        dv_m_s = v_end_m_s - v_start_m_s
        dv_kmh = dv_m_s * 3.6

        rows.append({
            "segment_idx": seg_idx,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": duration_s,
            "segment_distance_m": dist,
            "delta_elev_m": dh,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope_percent,
            "v_start_kmh": v_start_m_s * 3.6,
            "v_end_kmh": v_end_m_s * 3.6,
            "dv_kmh": dv_kmh,
            "dv_m_s": dv_m_s,
        })

    seg_df = pd.DataFrame(rows)
    return seg_df.sort_values("segment_idx").reset_index(drop=True)

# ---------------- СРЕДНИ ПОКАЗАТЕЛИ ЗА DOWNHILL ----------------
def downhill_summary(seg_df: pd.DataFrame) -> dict:
    # тук сег_df вече е само с сегменти, които са downhill и имат предишни 10 s спускане
    if seg_df.empty:
        return {
            "n_downhill_segments": 0,
            "avg_slope_percent": np.nan,
            "avg_downhill_speed_kmh": np.nan,
            "avg_dv_kmh": np.nan,
        }

    return {
        "n_downhill_segments": len(seg_df),
        "avg_slope_percent": seg_df["slope_percent"].mean(),
        "avg_downhill_speed_kmh": seg_df["avg_speed_kmh"].mean(),
        "avg_dv_kmh": seg_df["dv_kmh"].mean(),
    }

# ---------------- ОБРАБОТКА НА ЕДНА АКТИВНОСТ ----------------
def process_activity(file) -> dict:
    file_bytes = file.read()
    df_raw = parse_tcx_bytes(file_bytes)
    df_smooth = smooth_altitude(df_raw)
    df_clean = clean_artifacts(df_smooth)

    avg_activity_speed_kmh = compute_activity_avg_speed(df_clean)
    seg_df = segment_activity(df_clean)
    downhill = downhill_summary(seg_df)

    result = {
        "activity": file.name,
        "n_downhill_segments": downhill["n_downhill_segments"],
        "avg_slope_percent": downhill["avg_slope_percent"],
        "avg_downhill_speed_kmh": downhill["avg_downhill_speed_kmh"],
        "avg_dv_kmh": downhill["avg_dv_kmh"],
        "avg_activity_speed_kmh": avg_activity_speed_kmh,
    }
    return result, seg_df

# ---------------- STREAMLIT UI ----------------
def main():
    st.title("Ski-Glide-Dynamics — 10 s сегменти + предишни 10 s спускане (Δv модел)")

    files = st.file_uploader(
        "Качи един или повече TCX файла",
        type=["tcx"],
        accept_multiple_files=True
    )

    if not files:
        st.info("Качи поне един TCX файл, за да започнем.")
        return

    results = []
    debug_segments = {}

    for f in files:
        try:
            res, seg_df = process_activity(f)
            results.append(res)
            debug_segments[f.name] = seg_df
        except Exception as e:
            st.error(f"Грешка при обработка на {f.name}: {e}")

    if not results:
        st.error("Няма успешно обработени активности.")
        return

    df_results = pd.DataFrame(results)

    st.subheader("Сравнение на активности (4 ключови показателя)")
    st.dataframe(df_results)

    st.write("""
    **Колони:**
    - `avg_slope_percent` – среден наклон (%) на сегментите, които:
      - са по 10 s,
      - имат среден наклон ≤ -7%,
      - имат ≥ 20 m дистанция и ≥ 10 km/h,
      - и предходните 10 s преди тях също са със слизане (отрицателна денивелация).
    - `avg_downhill_speed_kmh` – средна скорост (km/h) в тези сегменти.
    - `avg_dv_kmh` – средна разлика в скоростта между начало и край на сегмента (Δv).
    - `avg_activity_speed_kmh` – средна скорост на цялата активност.
    - `n_downhill_segments` – брой валидни downhill сегменти.
    """)

    # по избор: да разгледаш сегментите за конкретна активност
    selected_name = st.selectbox(
        "Избери активност за да видиш сегментите (по желание):",
        options=list(debug_segments.keys())
    )
    if selected_name:
        st.write(f"Сегменти за: **{selected_name}**")
        st.dataframe(debug_segments[selected_name].head(50))

if __name__ == "__main__":
    main()
