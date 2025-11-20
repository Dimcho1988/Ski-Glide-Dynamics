import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ==============================
# НАСТРОЙКИ НА МОДЕЛА
# ==============================

SEG_LEN_SEC = 10                 # дължина на сегмента (секунди)
MIN_SEG_DIST_M = 20.0            # минимална дължина на сегмент (метри)
MIN_SEG_SPEED_KMH = 15.0         # минимална средна скорост в сегмент (km/h)
MAX_ABS_SLOPE_PCT = 30.0         # макс. абсолютен наклон в % (артефакт филтър)
MIN_SLOPE_DOWN_PCT = -5.0        # прагов наклон за "достатъчно спускане"
MIN_POINTS_PER_SEG = 3           # минимален брой точки в сегмент
ROLLING_ALT_WINDOW = 3           # прозорец за медианно изглаждане на височината


# ==============================
# ПОМОЩНИ ФУНКЦИИ
# ==============================

def parse_tcx(file) -> pd.DataFrame:
    """
    Чете TCX файл и връща DataFrame с колони:
    time, t_rel, dist, alt, alt_smooth, speed_mps, speed_kmh
    """
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    # TCX namespace може да е различен, затова ползваме wildcard {*} в path-овете
    trackpoints = root.findall(".//{*}Trackpoint")

    records = []
    for tp in trackpoints:
        t_el = tp.find(".//{*}Time")
        d_el = tp.find(".//{*}DistanceMeters")
        a_el = tp.find(".//{*}AltitudeMeters")

        if t_el is None or d_el is None or a_el is None:
            continue

        try:
            time = pd.to_datetime(t_el.text)
        except Exception:
            continue

        try:
            dist = float(d_el.text)
        except Exception:
            dist = np.nan

        try:
            alt = float(a_el.text)
        except Exception:
            alt = np.nan

        records.append(
            {
                "time": time,
                "dist": dist,
                "alt": alt,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).dropna(subset=["time", "dist", "alt"]).reset_index(drop=True)

    # Подреждаме по време и изчисляваме относителното време
    df = df.sort_values("time").reset_index(drop=True)
    df["t_rel"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # Скорост от дистанция/време
    df["delta_t"] = df["t_rel"].diff()
    df["delta_d"] = df["dist"].diff()

    # Премахваме първата точка (няма delta_t/d)
    df = df.iloc[1:].reset_index(drop=True)

    # Филтър за некоректни времеви стъпки
    df = df[df["delta_t"] > 0].copy()

    # Скорост (m/s и km/h)
    df["speed_mps"] = df["delta_d"] / df["delta_t"]
    df["speed_kmh"] = df["speed_mps"] * 3.6

    # Сглаждане на височината
    df["alt_smooth"] = df["alt"].rolling(window=ROLLING_ALT_WINDOW, center=True, min_periods=1).median()

    df = df.reset_index(drop=True)

    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разделя активността на 10-секундни сегменти и връща DataFrame с метрики
    за всички сегменти (преди финалния филтър за предходен сегмент).
    """
    if df.empty:
        return pd.DataFrame()

    # Определяме сегментен индекс
    df["seg_id"] = (df["t_rel"] // SEG_LEN_SEC).astype(int)

    segments = []

    for seg_id, seg in df.groupby("seg_id"):
        seg = seg.sort_values("t_rel")
        if len(seg) < MIN_POINTS_PER_SEG:
            continue

        t_start = seg["t_rel"].iloc[0]
        t_end = seg["t_rel"].iloc[-1]
        duration_s = t_end - t_start
        if duration_s <= 0:
            continue

        dist_start = seg["dist"].iloc[0]
        dist_end = seg["dist"].iloc[-1]
        segment_dist_m = dist_end - dist_start

        # Средна скорост от дистанция и време
        mean_speed_kmh = (segment_dist_m / duration_s) * 3.6 if segment_dist_m > 0 else 0.0

        # Наклон по старт/край върху изгладената височина
        alt_start = seg["alt_smooth"].iloc[0]
        alt_end = seg["alt_smooth"].iloc[-1]
        delta_h = alt_end - alt_start

        if segment_dist_m > 0:
            slope_pct = 100.0 * (delta_h / segment_dist_m)
        else:
            slope_pct = np.nan

        # Монотонично спускане: всички Δalt <= 0
        alt_diff = seg["alt_smooth"].diff().dropna()
        monotonic_downhill = bool((alt_diff <= 0).all()) if not alt_diff.empty else False

        # Реални скорости в началото и края
        start_speed_kmh = seg["speed_kmh"].iloc[0]
        end_speed_kmh = seg["speed_kmh"].iloc[-1]
        delta_speed_kmh = end_speed_kmh - start_speed_kmh

        # Базов филтър за артефакти
        base_valid = True
        if segment_dist_m < MIN_SEG_DIST_M:
            base_valid = False
        if mean_speed_kmh < MIN_SEG_SPEED_KMH:
            base_valid = False
        if np.isnan(slope_pct) or abs(slope_pct) > MAX_ABS_SLOPE_PCT:
            base_valid = False

        downhill_enough = False
        if not np.isnan(slope_pct) and slope_pct <= MIN_SLOPE_DOWN_PCT and monotonic_downhill:
            downhill_enough = True

        segments.append(
            {
                "seg_id": seg_id,
                "t_start_s": t_start,
                "t_end_s": t_end,
                "duration_s": duration_s,
                "segment_dist_m": segment_dist_m,
                "mean_speed_kmh": mean_speed_kmh,
                "slope_pct": slope_pct,
                "monotonic_downhill": monotonic_downhill,
                "base_valid": base_valid,
                "downhill_enough": downhill_enough,
                "start_speed_kmh": start_speed_kmh,
                "end_speed_kmh": end_speed_kmh,
                "delta_speed_kmh": delta_speed_kmh,
            }
        )

    if not segments:
        return pd.DataFrame()

    seg_df = pd.DataFrame(segments).sort_values("seg_id").reset_index(drop=True)
    return seg_df


def select_final_segments(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Избира сегментите, които:
    - са валидни
    - имат наклон <= -5% и са монотонично спускащи
    - са предхождани от сегмент със същите свойства
    """
    if seg_df.empty:
        return seg_df.copy()

    seg_df = seg_df.sort_values("seg_id").reset_index(drop=True)
    selected_flags = []

    for i in range(len(seg_df)):
        if i == 0:
            selected_flags.append(False)
            continue

        cur = seg_df.iloc[i]
        prev = seg_df.iloc[i - 1]

        cond_cur = (
            cur["base_valid"]
            and cur["downhill_enough"]
        )
        cond_prev = (
            prev["base_valid"]
            and prev["downhill_enough"]
        )

        selected_flags.append(bool(cond_cur and cond_prev))

    seg_df["selected"] = selected_flags
    return seg_df


# ==============================
# STREAMLIT UI
# ==============================

st.title("Ski Glide Dynamics – сегментен анализ при спускане")
st.write(
    """
Анализ на TCX ски/ролери активности:

1. Почистване на данните и изглаждане на височината  
2. Разделяне на 10-секундни сегменти  
3. Избор на сегменти с:
   - наклон ≤ -5%  
   - предходен сегмент също с наклон ≤ -5%  
   - само намаляваща денивелация в рамките на сегмента  
   - средна скорост ≥ 15 km/h и дължина ≥ 20 m
"""
)

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла",
    type=["tcx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("⬆ Качи поне един TCX файл, за да започне анализът.")
    st.stop()

all_segments = []
all_selected = []

for file in uploaded_files:
    st.subheader(f"Файл: {file.name}")

    try:
        df = parse_tcx(file)
    except Exception as e:
        st.error(f"Грешка при парсване на {file.name}: {e}")
        continue

    if df.empty:
        st.warning(f"{file.name}: няма валидни Trackpoint данни.")
        continue

    seg_df = build_segments(df)
    if seg_df.empty:
        st.warning(f"{file.name}: не бяха създадени валидни сегменти.")
        continue

    seg_df = select_final_segments(seg_df)

    # Добавяме името на файла
    seg_df["file_name"] = file.name

    # Всички сегменти
    all_segments.append(seg_df)

    # Само избраните сегменти според финалния филтър
    selected = seg_df[seg_df["selected"]].copy()
    all_selected.append(selected)

    st.write(f"Общ брой сегменти: {len(seg_df)}")
    st.write(f"Сегменти, преминали финалния филтър: {len(selected)}")

    if not selected.empty:
        st.write("Преглед на избраните сегменти за този файл:")
        st.dataframe(
            selected[
                [
                    "seg_id",
                    "t_start_s",
                    "t_end_s",
                    "duration_s",
                    "segment_dist_m",
                    "mean_speed_kmh",
                    "slope_pct",
                    "start_speed_kmh",
                    "end_speed_kmh",
                    "delta_speed_kmh",
                ]
            ]
        )

if not all_segments:
    st.error("Нито един файл не беше обработен успешно.")
    st.stop()

# Обединени данни
all_segments_df = pd.concat(all_segments, ignore_index=True)
all_selected_df = pd.concat(all_selected, ignore_index=True) if any(len(x) > 0 for x in all_selected) else pd.DataFrame()

st.header("Глобален резултат – всички файлове")

st.subheader("Всички сегменти (преди финалния филтър)")
st.dataframe(
    all_segments_df[
        [
            "file_name",
            "seg_id",
            "t_start_s",
            "t_end_s",
            "duration_s",
            "segment_dist_m",
            "mean_speed_kmh",
            "slope_pct",
            "monotonic_downhill",
            "base_valid",
            "downhill_enough",
            "selected",
            "start_speed_kmh",
            "end_speed_kmh",
            "delta_speed_kmh",
        ]
    ]
)

if all_selected_df.empty:
    st.warning("Няма сегменти, които да отговарят на всички критерии (наклон, предходен сегмент, монотонично спускане и т.н.).")
else:
    st.subheader("Сегменти, преминали финалния филтър (по всички критерии)")
    st.dataframe(
        all_selected_df[
            [
                "file_name",
                "seg_id",
                "t_start_s",
                "t_end_s",
                "duration_s",
                "segment_dist_m",
                "mean_speed_kmh",
                "slope_pct",
                "start_speed_kmh",
                "end_speed_kmh",
                "delta_speed_kmh",
            ]
        ]
    )

    # Сводна таблица по файл
    summary = (
        all_selected_df.groupby("file_name")
        .agg(
            n_segments=("seg_id", "count"),
            mean_speed_kmh=("mean_speed_kmh", "mean"),
            mean_slope_pct=("slope_pct", "mean"),
            mean_delta_speed_kmh=("delta_speed_kmh", "mean"),
            mean_dist_m=("segment_dist_m", "mean"),
        )
        .reset_index()
    )

    st.subheader("Своден анализ по файл (само избраните сегменти)")
    st.dataframe(summary)
