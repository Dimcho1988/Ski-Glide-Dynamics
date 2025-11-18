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
MIN_ABS_DELTA_ELEV = 0.3          # под това е просто шум във височината

# филтри за суровите данни
MAX_SPEED_M_S = 30.0
MAX_ALT_RATE_M_S = 5.0

# долна граница за downhill сегменти
MIN_DOWNHILL_SLOPE = -7.0         # вместо -5%

# ---- ПАРСВАНЕ НА TCX ----
def parse_tcx(file) -> pd.DataFrame:
    """
    Чете TCX файл и връща DataFrame с:
    time (datetime64), distance_m (float), altitude_m (float)
    """
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

        trackpoints.append(
            {
                "time": pd.to_datetime(time_el.text),
                "distance_m": float(dist_el.text) if dist_el is not None else np.nan,
                "altitude_m": float(alt_el.text) if alt_el is not None else np.nan,
            }
        )

    df = (
        pd.DataFrame(trackpoints)
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df


# ---- СГЛАЖДАНЕ НА ВИСОЧИНАТА ----
def smooth_altitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Медианен филтър 3 точки за изглаждане на височината.
    """
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(window=3, center=True).median()
    return df


# ---- ЧИСТЕНЕ НА АРТЕФАКТИ ----
def clean_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Премахва очевидно невалидни точки:
    - отрицателен или нулев времеви интервал
    - отрицателна промяна в дистанцията
    - нереалистично висока скорост
    - нереалистичен вертикален темп
    """
    df = df.copy()
    df["dt"] = df["time"].diff().dt.total_seconds()
    df["ddist"] = df["distance_m"].diff()
    df["dalt"] = df["altitude_m"].diff()

    speed_m_s = df["ddist"] / df["dt"]
    alt_rate_m_s = df["dalt"] / df["dt"]

    mask_valid = True
    mask_valid &= (df["dt"].isna() | (df["dt"] > 0))
    mask_valid &= (df["ddist"].isna() | (df["ddist"] >= 0))
    mask_valid &= (speed_m_s.isna() | ((speed_m_s >= 0) & (speed_m_s <= MAX_SPEED_M_S)))
    mask_valid &= (alt_rate_m_s.isna() | (alt_rate_m_s.abs() <= MAX_ALT_RATE_M_S))

    df_clean = df[mask_valid].copy().reset_index(drop=True)
    df_clean = df_clean[["time", "distance_m", "altitude_m"]]
    return df_clean


# ---- СЕГМЕНТИРАНЕ В 15-СЕК ИНТЕРВАЛИ + УСКОРЕНИЕ ----
def segment_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сегментира активността на 15-сек интервали.
    За всеки сегмент:
      - duration_s
      - segment_distance_m
      - delta_elev_m
      - avg_speed_kmh
      - slope_percent
      - v_start_kmh, v_end_kmh (от първи и последен интервал)
      - accel_m_s2 (средно ускорение за сегмента)
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    start_time = df["time"].iloc[0]
    df["elapsed_s"] = (df["time"] - start_time).dt.total_seconds()
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

        dist_start = g["distance_m"].iloc[0]
        dist_end = g["distance_m"].iloc[-1]
        alt_start = g["altitude_m"].iloc[0]
        alt_end = g["altitude_m"].iloc[-1]

        segment_distance_m = dist_end - dist_start
        delta_elev_m = alt_end - alt_start

        if segment_distance_m <= 0:
            continue
        if segment_distance_m < MIN_SEGMENT_DISTANCE_M:
            continue
        if abs(delta_elev_m) < MIN_ABS_DELTA_ELEV:
            continue
        if pd.isna(alt_start) or pd.isna(alt_end):
            continue

        avg_speed_m_s = segment_distance_m / duration_s
        avg_speed_kmh = avg_speed_m_s * 3.6
        if avg_speed_kmh < MIN_SEGMENT_SPEED_KMH:
            continue

        slope_percent = (delta_elev_m / segment_distance_m) * 100.0
        if abs(slope_percent) > MAX_ABS_SLOPE_PERCENT:
            continue

        # ---- Скорост в началото (v_start) и в края (v_end) ----
        # v_start: от първите две точки
        if len(g) >= 2:
            dt_start = (g["time"].iloc[1] - g["time"].iloc[0]).total_seconds()
            if dt_start > 0:
                v_start = (g["distance_m"].iloc[1] - g["distance_m"].iloc[0]) / dt_start
            else:
                v_start = avg_speed_m_s
        else:
            v_start = avg_speed_m_s

        # v_end: от последните две точки
        if len(g) >= 2:
            dt_end = (g["time"].iloc[-1] - g["time"].iloc[-2]).total_seconds()
            if dt_end > 0:
                v_end = (g["distance_m"].iloc[-1] - g["distance_m"].iloc[-2]) / dt_end
            else:
                v_end = avg_speed_m_s
        else:
            v_end = avg_speed_m_s

        accel_m_s2 = (v_end - v_start) / duration_s

        rows.append(
            {
                "segment_idx": seg_idx,
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration_s,
                "segment_distance_m": segment_distance_m,
                "delta_elev_m": delta_elev_m,
                "avg_speed_kmh": avg_speed_kmh,
                "slope_percent": slope_percent,
                "v_start_kmh": v_start * 3.6,
                "v_end_kmh": v_end * 3.6,
                "accel_m_s2": accel_m_s2,
            }
        )

    seg_df = pd.DataFrame(rows)
    if seg_df.empty:
        return seg_df

    seg_df = seg_df.sort_values("segment_idx").reset_index(drop=True)
    return seg_df


# ---- ФИЛТЪР ЗА СПУСКАНИЯ И АГРЕГАЦИЯ ----
def compute_downhill_stats(seg_df: pd.DataFrame):
    """
    Взима сегментите с наклон <= MIN_DOWNHILL_SLOPE и
    връща суми, средни стойности и средно ускорение.
    """
    downhill = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE].copy()

    if downhill.empty:
        return {
            "count_segments": 0,
            "sum_speed_kmh": 0.0,
            "sum_slope_percent": 0.0,
            "avg_speed_kmh": 0.0,
            "avg_slope_percent": 0.0,
            "avg_accel_m_s2": 0.0,
            "downhill_df": downhill,
        }

    sum_speed_kmh = downhill["avg_speed_kmh"].sum()
    sum_slope_percent = downhill["slope_percent"].sum()
    avg_speed_kmh = downhill["avg_speed_kmh"].mean()
    avg_slope_percent = downhill["slope_percent"].mean()
    avg_accel_m_s2 = downhill["accel_m_s2"].mean()

    return {
        "count_segments": len(downhill),
        "sum_speed_kmh": sum_speed_kmh,
        "sum_slope_percent": sum_slope_percent,
        "avg_speed_kmh": avg_speed_kmh,
        "avg_slope_percent": avg_slope_percent,
        "avg_accel_m_s2": avg_accel_m_s2,
        "downhill_df": downhill,
    }


# ---- STREAMLIT UI ----
def main():
    st.title("Ski-Glide-Dynamics — 15 сек сегменти + ускорение")

    uploaded_file = st.file_uploader("Качи TCX файл", type=["tcx"])

    if uploaded_file is None:
        st.info("Моля, качи TCX файл, за да започнем анализа.")
        return

    # 1) Парсване
    try:
        df_raw = parse_tcx(uploaded_file)
    except Exception as e:
        st.error(f"Грешка при парсване на TCX файла: {e}")
        return

    st.success(f"Успешно зареден TCX с {len(df_raw)} точки.")
    if st.checkbox("Покажи суровите данни (първите 10 реда)"):
        st.dataframe(df_raw.head(10))

    # 2) Сглаждане и чистене
    df_smooth = smooth_altitude(df_raw)
    df_clean = clean_artifacts(df_smooth)
    st.write(f"След почистване останаха **{len(df_clean)}** точки.")

    # 3) Сегментиране
    seg_df = segment_activity(df_clean)
    if seg_df.empty:
        st.error("Не бяха създадени валидни сегменти при тези филтри.")
        return

    st.write(f"Общ брой валидни сегменти: **{len(seg_df)}**")

    if st.checkbox("Покажи всички сегменти (първите 20):"):
        st.dataframe(seg_df.head(20))

    # 4) Downhill статистика
    results = compute_downhill_stats(seg_df)

    st.subheader("Резултати за сегментите с наклон ≤ -7%")

    st.write(f"Брой downhill сегменти: **{results['count_segments']}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Сума от скоростите (km/h)", f"{results['sum_speed_kmh']:.2f}")
        st.metric("Средна скорост (km/h)", f"{results['avg_speed_kmh']:.2f}")
    with col2:
        st.metric("Сума от наклоните (%)", f"{results['sum_slope_percent']:.2f}")
        st.metric("Среден наклон (%)", f"{results['avg_slope_percent']:.2f}")
    with col3:
        st.metric("Средно ускорение (m/s²)", f"{results['avg_accel_m_s2']:.4f}")

    if results["count_segments"] == 0:
        st.warning("Няма сегменти с наклон ≤ -7% при тези условия.")
    else:
        if st.checkbox("Покажи downhill сегментите (първите 30):"):
            st.dataframe(results["downhill_df"].head(30))


if __name__ == "__main__":
    main()
