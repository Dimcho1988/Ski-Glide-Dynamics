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

MIN_DOWNHILL_SLOPE = -7.0   # граница за downhill

# ---- TCX ПАРСВАНЕ ----
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

# ---- СГЛАЖДАНЕ НА ВИСОЧИНАТА ----
def smooth_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(3, center=True).median()
    return df

# ---- ЧИСТЕНЕ НА АРТЕФАКТИ ----
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
    mask &= (alt_rate.isna() | (abs(alt_rate) <= MAX_ALT_RATE_M_S))

    df = df[mask].copy().reset_index(drop=True)
    df = df[["time", "distance_m", "altitude_m"]]
    return df

# ---- СРЕДНА СКОРОСТ НА ЦЯЛАТА АКТИВНОСТ ----
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

# ---- СЕГМЕНТИРАНЕ + Δv ----
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

        # начална скорост (v_start) – от първите две точки
        dt_start = (g["time"].iloc[1] - g["time"].iloc[0]).total_seconds()
        if dt_start > 0:
            v_start_m_s = (g["distance_m"].iloc[1] - g["distance_m"].iloc[0]) / dt_start
        else:
            v_start_m_s = avg_speed_m_s

        # крайна скорост (v_end) – от последните две точки
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
            "dv_kmh": dv_kmh,
            "dv_m_s": dv_m_s,
        })

    seg_df = pd.DataFrame(rows)
    return seg_df.sort_values("segment_idx").reset_index(drop=True)

# ---- DOWNHILL СРЕДНИ ПОКАЗАТЕЛИ ----
def downhill_summary(seg_df: pd.DataFrame) -> dict:
    down = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE]

    if down.empty:
        return {
            "n_downhill_segments": 0,
            "avg_slope_percent": np.nan,
            "avg_downhill_speed_kmh": np.nan,
            "avg_dv_kmh": np.nan,
        }

    return {
        "n_downhill_segments": len(down),
        "avg_slope_percent": down["slope_percent"].mean(),
        "avg_downhill_speed_kmh": down["avg_speed_kmh"].mean(),
        "avg_dv_kmh": down["dv_kmh"].mean(),
    }

# ---- ОБРАБОТКА НА ЕДНА АКТИВНОСТ ----
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
    return result

# ---- STREAMLIT UI ----
def main():
    st.title("Ski-Glide-Dynamics — Многократни активности (Δv модел)")

    files = st.file_uploader("Качи един или повече TCX файла", type=["tcx"], accept_multiple_files=True)
    if not files:
        st.info("Качи поне един TCX файл, за да започнем.")
        return

    results = []
    for f in files:
        try:
            res = process_activity(f)
            results.append(res)
        except Exception as e:
            st.error(f"Грешка при обработка на {f.name}: {e}")

    if not results:
        st.error("Няма успешно обработени активности.")
        return

    df_results = pd.DataFrame(results)

    st.subheader("Сравнение на активности (4-те ключови показателя)")
    st.dataframe(df_results)

    st.write("""
    **Колони:**
    - `avg_slope_percent` – среден наклон (%) на downhill сегментите (slope ≤ -7%).
    - `avg_downhill_speed_kmh` – средна скорост (km/h) в тези сегменти.
    - `avg_dv_kmh` – средна разлика в скоростта между начало и край на сегмента (Δv).
    - `avg_activity_speed_kmh` – средна скорост на цялата активност.
    - `n_downhill_segments` – брой downhill сегменти, върху които са изчислени показателите.
    """)

if __name__ == "__main__":
    main()
