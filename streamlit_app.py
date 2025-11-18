import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ---- НАСТРОЙКИ ----
SEGMENT_LENGTH_SEC = 15           # дължина на сегмента
MIN_SEGMENT_DURATION = 10.0       # реални сек
MIN_SEGMENT_DISTANCE_M = 20.0     # минимум изминати метри
MIN_SEGMENT_SPEED_KMH = 10.0      # минимум средна скорост
MAX_ABS_SLOPE_PERCENT = 30.0      # максимум абсолютен наклон
MIN_ABS_DELTA_ELEV = 0.3          # под това е просто шум

# филтри за суровите данни
MAX_SPEED_M_S = 30.0
MAX_ALT_RATE_M_S = 5.0            # по-реалистичен лимит
MIN_DOWNHILL_SLOPE = -5.0         # филтър за спускане (< -5%)

# ---- ПАРСВАНЕ ----
def parse_tcx(file) -> pd.DataFrame:
    file_bytes = file.read()
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    trackpoints = []
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

        trackpoints.append({
            "time": pd.to_datetime(time_el.text),
            "distance_m": float(dist_el.text) if dist_el is not None else np.nan,
            "altitude_m": float(alt_el.text) if alt_el is not None else np.nan
        })

    df = pd.DataFrame(trackpoints).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


# ---- СГЛАЖДАНЕ НА ВИСОЧИНАТА ----
def smooth_altitude(df):
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(window=3, center=True).median()
    return df


# ---- ЧИСТЕНЕ ----
def clean_artifacts(df):
    df = df.copy()
    df["dt"] = df["time"].diff().dt.total_seconds()
    df["ddist"] = df["distance_m"].diff()
    df["dalt"] = df["altitude_m"].diff()

    speed_m_s = df["ddist"] / df["dt"]
    alt_rate = df["dalt"] / df["dt"]

    mask_valid = True
    mask_valid &= (df["dt"].isna() | (df["dt"] > 0))
    mask_valid &= (df["ddist"].isna() | (df["ddist"] >= 0))
    mask_valid &= (speed_m_s.isna() | ((speed_m_s >= 0) & (speed_m_s <= MAX_SPEED_M_S)))
    mask_valid &= (alt_rate.isna() | (abs(alt_rate) <= MAX_ALT_RATE_M_S))

    df = df[mask_valid].copy().reset_index(drop=True)
    df = df[["time", "distance_m", "altitude_m"]]
    return df


# ---- СЕГМЕНТИРАНЕ ----
def segment_activity(df):
    df = df.copy()
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["segment_idx"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    rows = []
    for seg_idx, g in df.groupby("segment_idx"):
        if len(g) < 2:
            continue

        t_start, t_end = g["time"].iloc[0], g["time"].iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        if duration_s < MIN_SEGMENT_DURATION:
            continue

        dist_start, dist_end = g["distance_m"].iloc[0], g["distance_m"].iloc[-1]
        alt_start, alt_end = g["altitude_m"].iloc[0], g["altitude_m"].iloc[-1]

        segment_distance_m = dist_end - dist_start
        delta_elev_m = alt_end - alt_start

        if segment_distance_m <= 0:
            continue
        if segment_distance_m < MIN_SEGMENT_DISTANCE_M:
            continue
        if abs(delta_elev_m) < MIN_ABS_DELTA_ELEV:
            continue  # твърде малък Δh → шум

        avg_speed_m_s = segment_distance_m / duration_s
        avg_speed_kmh = avg_speed_m_s * 3.6

        if avg_speed_kmh < MIN_SEGMENT_SPEED_KMH:
            continue

        slope_percent = (delta_elev_m / segment_distance_m) * 100
        if abs(slope_percent) > MAX_ABS_SLOPE_PERCENT:
            continue

        rows.append({
            "segment_idx": seg_idx,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": duration_s,
            "segment_distance_m": segment_distance_m,
            "delta_elev_m": delta_elev_m,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope_percent
        })

    seg_df = pd.DataFrame(rows)
    return seg_df.sort_values("segment_idx").reset_index(drop=True)


# ---- АГРЕГАЦИЯ ----
def compute_downhill_sums(seg_df):
    down = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE]

    if down.empty:
        return dict(
            count_segments=0, sum_speed_kmh=0.0, sum_slope_percent=0.0,
            avg_speed_kmh=0.0, avg_slope_percent=0.0, downhill_df=down
        )

    return {
        "count_segments": len(down),
        "sum_speed_kmh": down["avg_speed_kmh"].sum(),
        "sum_slope_percent": down["slope_percent"].sum(),
        "avg_speed_kmh": down["avg_speed_kmh"].mean(),
        "avg_slope_percent": down["slope_percent"].mean(),
        "downhill_df": down
    }


# ---- STREAMLIT ----
def main():
    st.title("Ski-Glide-Dynamics — 15 сек сегменти (прецизна версия)")

    uploaded = st.file_uploader("Качи TCX файл", type=["tcx"])
    if not uploaded:
        st.info("Моля, качи TCX файл")
        return

    df_raw = parse_tcx(uploaded)
    df_smooth = smooth_altitude(df_raw)
    df_clean = clean_artifacts(df_smooth)

    seg_df = segment_activity(df_clean)

    if seg_df.empty:
        st.error("Няма валидни сегменти при тези филтри.")
        return

    st.write(f"Общ брой валидни сегменти: **{len(seg_df)}**")

    results = compute_downhill_sums(seg_df)

    st.subheader("Резултати (наклон < -5%)")
    st.metric("Брой сегменти", results["count_segments"])
    st.metric("Сума от скоростите (km/h)", f"{results['sum_speed_kmh']:.2f}")
    st.metric("Сума от наклоните (%)", f"{results['sum_slope_percent']:.2f}")
    st.metric("Средна скорост (km/h)", f"{results['avg_speed_kmh']:.2f}")
    st.metric("Среден наклон (%)", f"{results['avg_slope_percent']:.2f}")

    if st.checkbox("Покажи сегментите (първите 20)"):
        st.dataframe(seg_df.head(20))

    if st.checkbox("Покажи downhill сегментите"):
        st.dataframe(results["downhill_df"].head(30))


if __name__ == "__main__":
    main()
