import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ------------------------------------------------------------
# НАСТРОЙКИ (можеш да ги променяш лесно)
# ------------------------------------------------------------
SEGMENT_LENGTH_SEC = 15
MIN_SEGMENT_DURATION = 10.0
MIN_SEGMENT_DISTANCE_M = 20.0
MIN_SEGMENT_SPEED_KMH = 10.0
MAX_ABS_SLOPE_PERCENT = 30.0
MIN_ABS_DELTA_ELEV = 0.3

MAX_SPEED_M_S = 30.0
MAX_ALT_RATE_M_S = 5.0

# steady-state условие:
ACCEL_MAX = 0.05       # |a| < 0.05 m/s^2

# downhill условие:
MIN_DOWNHILL_SLOPE = -5.0

# ------------------------------------------------------------
# TCX парсване
# ------------------------------------------------------------
def parse_tcx(file) -> pd.DataFrame:
    file_bytes = file.read()
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
            "altitude_m": float(alt_el.text) if alt_el is not None else np.nan,
        })
    df = pd.DataFrame(points).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df

# ------------------------------------------------------------
# Изглаждане (медианен филтър 3 точки)
# ------------------------------------------------------------
def smooth_altitude(df):
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(3, center=True).median()
    return df

# ------------------------------------------------------------
# Премахване на артефакти
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Сегментиране
# ------------------------------------------------------------
def segment_activity(df):
    df = df.copy()
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["segment_idx"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    rows = []
    for seg_idx, g in df.groupby("segment_idx"):
        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        if duration_s < MIN_SEGMENT_DURATION:
            continue

        dist_start, dist_end = g["distance_m"].iloc[0], g["distance_m"].iloc[-1]
        alt_start, alt_end = g["altitude_m"].iloc[0], g["altitude_m"].iloc[-1]

        dist = dist_end - dist_start
        dh = alt_end - alt_start

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

        # ------------- УСКОРЕНИЕ -------------
        # първи 3 секунди
        g_first = g[g["time"] <= (t_start + pd.Timedelta(seconds=3))]
        g_last = g[g["time"] >= (t_end - pd.Timedelta(seconds=3))]

        if len(g_first) < 2 or len(g_last) < 2:
            continue

        v_start = (g_first["distance_m"].iloc[-1] - g_first["distance_m"].iloc[0]) / \
                  (g_first["time"].iloc[-1] - g_first["time"].iloc[0]).total_seconds()

        v_end = (g_last["distance_m"].iloc[-1] - g_last["distance_m"].iloc[0]) / \
                (g_last["time"].iloc[-1] - g_last["time"].iloc[0]).total_seconds()

        accel = (v_end - v_start) / duration_s

        # steady-state условие
        if abs(accel) > ACCEL_MAX:
            continue

        rows.append({
            "segment_idx": seg_idx,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": duration_s,
            "segment_distance_m": dist,
            "delta_elev_m": dh,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope_percent,
            "v_start_kmh": v_start * 3.6,
            "v_end_kmh": v_end * 3.6,
            "accel_m_s2": accel,
        })

    seg_df = pd.DataFrame(rows)
    return seg_df.sort_values("segment_idx").reset_index(drop=True)

# ------------------------------------------------------------
# Downhill анализ
# ------------------------------------------------------------
def compute_downhill(seg_df):
    down = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE]
    if down.empty:
        return dict(
            count=0, sum_speed=0.0, sum_slope=0.0,
            avg_speed=0.0, avg_slope=0.0, df=down
        )

    return {
        "count": len(down),
        "sum_speed": down["avg_speed_kmh"].sum(),
        "sum_slope": down["slope_percent"].sum(),
        "avg_speed": down["avg_speed_kmh"].mean(),
        "avg_slope": down["slope_percent"].mean(),
        "df": down
    }

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------
def main():
    st.title("Ski-Glide Dynamics — Steady-State Only Model (15 sec)")

    uploaded = st.file_uploader("Качи TCX файл", type=["tcx"])
    if not uploaded:
        st.info("Изчакам те да качиш TCX.")
        return

    df_raw = parse_tcx(uploaded)
    df_smooth = smooth_altitude(df_raw)
    df_clean = clean_artifacts(df_smooth)

    seg_df = segment_activity(df_clean)

    if seg_df.empty:
        st.error("Няма steady-state сегменти при тези филтри.")
        return

    st.success(f"Намерени steady-state сегменти: {len(seg_df)}")
    if st.checkbox("Покажи първите 20 сегмента"):
        st.dataframe(seg_df.head(20))

    # ---- Downhill ----
    res = compute_downhill(seg_df)

    st.subheader("Downhill анализ (slope < -5%)")
    st.metric("Брой сегменти", res["count"])
    st.metric("Средна скорост (km/h)", f"{res['avg_speed']:.2f}")
    st.metric("Среден наклон (%)", f"{res['avg_slope']:.2f}")
    st.metric("Сума скорост (km/h)", f"{res['sum_speed']:.2f}")
    st.metric("Сума наклон (%)", f"{res['sum_slope']:.2f}")

    if st.checkbox("Покажи downhill сегментите"):
        st.dataframe(res["df"].head(30))


if __name__ == "__main__":
    main()
