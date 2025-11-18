import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ---- НАСТРОЙКИ ----
SEGMENT_LENGTH_SEC = 15
MIN_SEGMENT_DURATION = 10.0
MIN_SEGMENT_DISTANCE_M = 20.0
MIN_SEGMENT_SPEED_KMH = 10.0
MAX_ABS_SLOPE_PERCENT = 30.0
MIN_ABS_DELTA_ELEV = 0.3

MAX_SPEED_M_S = 30.0
MAX_ALT_RATE_M_S = 5.0

MIN_DOWNHILL_SLOPE = -7.0   # новата граница

# ---- ПАРСВАНЕ ----
def parse_tcx(file):
    file_bytes = file.read()
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

# ---- СГЛАЖДАНЕ ----
def smooth_altitude(df):
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(3, center=True).median()
    return df

# ---- ЧИСТЕНЕ ----
def clean_artifacts(df):
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
    mask &= (alt_rate.isna() | (abs(alt_rate) <= MAX_ALT_RATE_M_S))

    df = df[mask].copy().reset_index(drop=True)
    df = df[["time", "distance_m", "altitude_m"]]
    return df

# ---- СЕГМЕНТИРАНЕ + Δv ----
def segment_activity(df):
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

        slope_percent = (dh / dist) * 100
        if abs(slope_percent) > MAX_ABS_SLOPE_PERCENT:
            continue

        # --- начало/крайна скорост (за Δv) ---
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
            "duration_s": duration_s,
            "segment_distance_m": dist,
            "delta_elev_m": dh,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope_percent,
            "v_start_kmh": v_start_m_s * 3.6,
            "v_end_kmh": v_end_m_s * 3.6,
            "dv_kmh": dv_kmh,         # <-- разлика на скоростите
            "dv_m_s": dv_m_s,
        })

    seg_df = pd.DataFrame(rows)
    return seg_df.sort_values("segment_idx")

# ---- DOWNHILL СТАТИСТИКА ----
def downhill_stats(seg_df):
    down = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE]

    if down.empty:
        return {
            "count": 0,
            "sum_speed": 0.0,
            "sum_slope": 0.0,
            "sum_dv_kmh": 0.0,
            "avg_dv_kmh": 0.0,
            "df": down
        }

    return {
        "count": len(down),
        "sum_speed": down["avg_speed_kmh"].sum(),
        "sum_slope": down["slope_percent"].sum(),
        "sum_dv_kmh": down["dv_kmh"].sum(),
        "avg_dv_kmh": down["dv_kmh"].mean(),
        "df": down
    }

# ---- STREAMLIT ----
def main():
    st.title("Ski-Glide-Dynamics — Δv модел (15 сек сегменти)")

    file = st.file_uploader("Качи TCX файл", type=["tcx"])
    if file is None:
        return

    df_raw = parse_tcx(file)
    df_smooth = smooth_altitude(df_raw)
    df_clean = clean_artifacts(df_smooth)

    seg_df = segment_activity(df_clean)
    if seg_df.empty:
        st.error("Няма валидни сегменти.")
        return

    if st.checkbox("Покажи първите 20 сегмента"):
        st.dataframe(seg_df.head(20))

    results = downhill_stats(seg_df)

    st.subheader("Downhill резултати (slope ≤ -7%)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Брой сегменти", results["count"])
        st.metric("Общ Δv (km/h)", f"{results['sum_dv_kmh']:.2f}")
    with col2:
        st.metric("Среден Δv (km/h)", f"{results['avg_dv_kmh']:.2f}")
        st.metric("Сума скорости (km/h)", f"{results['sum_speed']:.2f}")
    with col3:
        st.metric("Сума наклон (%)", f"{results['sum_slope']:.2f}")

    if st.checkbox("Покажи downhill сегментите"):
        st.dataframe(results["df"].head(50))

if __name__ == "__main__":
    main()
