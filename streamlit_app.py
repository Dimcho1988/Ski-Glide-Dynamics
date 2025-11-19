import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ======================================================
# НАСТРОЙКИ
# ======================================================
SEGMENT_LENGTH_SEC = 10
MIN_SEGMENT_DURATION = 8.0
MIN_SEGMENT_DISTANCE_M = 20.0
MIN_SEGMENT_SPEED_KMH = 10.0
MAX_ABS_SLOPE_PERCENT = 30.0
MIN_ABS_DELTA_ELEV = 0.3

MAX_SPEED_M_S = 30.0
MAX_ALT_RATE_M_S = 5.0

# Условия за downhill сегмент
MIN_DOWNHILL_SLOPE = -7.0

# Условия за 10-те секунди преди сегмента
PRE_WINDOW_SEC = 10
MIN_PRE_WINDOW_DIST_M = 10.0
MIN_PRE_WINDOW_SLOPE = -5.0

# Само отрицателна денивелация в сегмента
MAX_LOCAL_UPHILL_M = 0.1


# ======================================================
# TCX ПАРСВАНЕ
# ======================================================
def parse_tcx_bytes(file_bytes):
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    points = []
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        time_el = dist_el = alt_el = None
        for ch in tp:
            if ch.tag.endswith("Time"):
                time_el = ch
            elif ch.tag.endswith("DistanceMeters"):
                dist_el = ch
            elif ch.tag.endswith("AltitudeMeters"):
                alt_el = ch

        if time_el is None:
            continue

        points.append({
            "time": pd.to_datetime(time_el.text),
            "distance_m": float(dist_el.text) if dist_el is not None else np.nan,
            "altitude_m": float(alt_el.text) if alt_el is not None else np.nan
        })

    df = pd.DataFrame(points).dropna(subset=["time"])
    return df.sort_values("time").reset_index(drop=True)


# ======================================================
# СГЛАЖДАНЕ НА ВИСОЧИНАТА
# ======================================================
def smooth_altitude(df):
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(3, center=True).median()
    return df


# ======================================================
# ЧИСТЕНЕ НА АРТЕФАКТИ
# ======================================================
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
    return df[["time", "distance_m", "altitude_m"]]


# ======================================================
# СРЕДНА СКОРОСТ НА ЦЯЛАТА АКТИВНОСТ
# ======================================================
def compute_activity_avg_speed(df):
    if len(df) < 2:
        return np.nan
    dt = (df["time"].iloc[-1] - df["time"].iloc[0]).total_seconds()
    if dt <= 0:
        return np.nan
    dist = df["distance_m"].iloc[-1] - df["distance_m"].iloc[0]
    if dist <= 0:
        return np.nan
    return (dist / dt) * 3.6


# ======================================================
# СЕГМЕНТИРАНЕ + Δv + филтри
# ======================================================
def segment_activity(df):
    df = df.copy()
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["segment_idx"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    results = []

    for seg_idx, g in df.groupby("segment_idx"):
        g = g.sort_values("time")

        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        dur = (t_end - t_start).total_seconds()
        if dur < MIN_SEGMENT_DURATION:
            continue

        dist = g["distance_m"].iloc[-1] - g["distance_m"].iloc[0]
        dh = g["altitude_m"].iloc[-1] - g["altitude_m"].iloc[0]

        if dist < MIN_SEGMENT_DISTANCE_M:
            continue
        if abs(dh) < MIN_ABS_DELTA_ELEV:
            continue

        avg_speed_m_s = dist / dur
        avg_speed_kmh = avg_speed_m_s * 3.6
        if avg_speed_kmh < MIN_SEGMENT_SPEED_KMH:
            continue

        slope = (dh / dist) * 100
        if abs(slope) > MAX_ABS_SLOPE_PERCENT:
            continue
        if slope > MIN_DOWNHILL_SLOPE:
            continue  # трябва да е ≤ -7%

        # ------------------------------------------------
        # 1) Проверка за предишните 10 секунди (slope ≤ -5%)
        # ------------------------------------------------
        window_start = t_start - pd.Timedelta(seconds=PRE_WINDOW_SEC)
        prev = df[(df["time"] >= window_start) & (df["time"] < t_start)]
        if len(prev) < 2:
            continue

        prev_dist = prev["distance_m"].iloc[-1] - prev["distance_m"].iloc[0]
        prev_dh = prev["altitude_m"].iloc[-1] - prev["altitude_m"].iloc[0]

        if prev_dist < MIN_PRE_WINDOW_DIST_M:
            continue

        prev_slope = (prev_dh / prev_dist) * 100
        if prev_slope > MIN_PRE_WINDOW_SLOPE:
            continue  # slope трябва да е ≤ -5%

        # ------------------------------------------------
        # 2) Сегментът да е САМО спускане (без локални качвания)
        # ------------------------------------------------
        alt_diff = g["altitude_m"].diff().dropna()
        if (alt_diff > MAX_LOCAL_UPHILL_M).any():
            continue

        # ------------------------------------------------
        # 3) Δv (крайна - начална скорост)
        # ------------------------------------------------
        dt_start = (g["time"].iloc[1] - g["time"].iloc[0]).total_seconds()
        dt_end = (g["time"].iloc[-1] - g["time"].iloc[-2]).total_seconds()

        if dt_start > 0:
            v_start = (g["distance_m"].iloc[1] - g["distance_m"].iloc[0]) / dt_start
        else:
            v_start = avg_speed_m_s

        if dt_end > 0:
            v_end = (g["distance_m"].iloc[-1] - g["distance_m"].iloc[-2]) / dt_end
        else:
            v_end = avg_speed_m_s

        dv_m_s = v_end - v_start
        dv_kmh = dv_m_s * 3.6

        results.append({
            "segment_idx": seg_idx,
            "t_start": t_start,
            "duration_s": dur,
            "segment_distance_m": dist,
            "delta_elev_m": dh,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope,
            "v_start_kmh": v_start * 3.6,
            "v_end_kmh": v_end * 3.6,
            "dv_kmh": dv_kmh
        })

    return pd.DataFrame(results)


# ======================================================
# DOWNHILL SUMMARY
# ======================================================
def downhill_summary(seg_df):
    if seg_df.empty:
        return {
            "n_segments": 0,
            "avg_slope": np.nan,
            "avg_speed": np.nan,
            "avg_dv": np.nan
        }

    return {
        "n_segments": len(seg_df),
        "avg_slope": seg_df["slope_percent"].mean(),
        "avg_speed": seg_df["avg_speed_kmh"].mean(),
        "avg_dv": seg_df["dv_kmh"].mean()
    }


# ======================================================
# STREAMLIT UI
# ======================================================
def main():
    st.title("Ski-Glide-Dynamics — чисти 10 сек спускания (Δv модел)")

    files = st.file_uploader("Качи един или повече TCX файла",
                             type=["tcx"],
                             accept_multiple_files=True)

    if not files:
        return

    final_rows = []

    for f in files:
        try:
            raw = parse_tcx_bytes(f.read())
            raw = smooth_altitude(raw)
            clean = clean_artifacts(raw)

            avg_activity_speed = compute_activity_avg_speed(clean)
            seg_df = segment_activity(clean)
            summary = downhill_summary(seg_df)

            final_rows.append({
                "activity": f.name,
                "n_segments": summary["n_segments"],
                "avg_slope_percent": summary["avg_slope"],
                "avg_downhill_speed_kmh": summary["avg_speed"],
                "avg_dv_kmh": summary["avg_dv"],
                "avg_activity_speed_kmh": avg_activity_speed
            })

        except Exception as e:
            st.error(f"Грешка при обработка на {f.name}: {e}")

    df = pd.DataFrame(final_rows)
    st.subheader("Сравнение на активности")
    st.dataframe(df)

if __name__ == "__main__":
    main()
