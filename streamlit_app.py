import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from math import radians, sin, cos, sqrt, atan2

# ==============================
# НАСТРОЙКИ НА МОДЕЛА
# ==============================
SEGMENT_LENGTH_SEC = 5.0        # дължина на сегмента (номинално)
MIN_SEG_DURATION_SEC = 4.0      # минимална реална продължителност на сегмент
MIN_SEG_DISTANCE_M = 10.0       # минимум изминати метри в сегмента
MIN_SEG_SPEED_MPS = 2.5         # минимум средна скорост (m/s) ~ 9 km/h

MAX_SPEED_MPS = 16.0            # горна граница за скоростта (~ 58 km/h)
MAX_DT_SEC = 30.0               # максимум стъпка по време, иначе шум

# Условия за "глайд" сегмент
# ТУК СА НОВИТЕ ОГРАНИЧЕНИЯ
MIN_DOWNHILL_SLOPE = -12.0      # минимален наклон (по-стръмно надолу)
MAX_DOWNHILL_SLOPE = -5.0       # максимален наклон (по-полегато надолу)
PREV_MIN_SLOPE = -5.0           # предходният 5 s сегмент да е поне -5%

MIN_GLADE_SEGMENTS_PER_ACTIVITY = 5  # минимум глайд сегменти за надеждна оценка


# ==============================
# ПОМОЩНИ ФУНКЦИИ
# ==============================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Връща разстояние в метри между две GPS точки."""
    R = 6371000.0  # радиус на Земята (m)
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi / 2)**2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def parse_tcx(uploaded_file):
    """
    Парсва TCX файл в DataFrame с колони:
    time, sec, dist_m, elev_m, speed_mps
    """
    try:
        uploaded_file.seek(0)
        tree = ET.parse(uploaded_file)
        root = tree.getroot()
    except Exception as e:
        st.error(f"Грешка при парсване на {uploaded_file.name}: {e}")
        return None

    ns_trackpoint = ".//{*}Trackpoint"

    times = []
    dist_m = []
    elev_m = []
    lat_list = []
    lon_list = []
    speed_raw = []

    for tp in root.findall(ns_trackpoint):
        t_str = tp.findtext(".//{*}Time")
        if t_str is None:
            continue

        times.append(t_str)

        d_str = tp.findtext(".//{*}DistanceMeters")
        dist_m.append(float(d_str) if d_str is not None else np.nan)

        a_str = tp.findtext(".//{*}AltitudeMeters")
        elev_m.append(float(a_str) if a_str is not None else np.nan)

        lat_str = tp.findtext(".//{*}LatitudeDegrees")
        lon_str = tp.findtext(".//{*}LongitudeDegrees")
        lat_list.append(float(lat_str) if lat_str is not None else np.nan)
        lon_list.append(float(lon_str) if lon_str is not None else np.nan)

        # Скорост от Extensions, ако я има
        s_val = None
        for ext in tp.findall(".//{*}Extensions"):
            s_candidate = ext.findtext(".//{*}Speed")
            if s_candidate is not None:
                s_val = s_candidate
                break
        speed_raw.append(float(s_val) if s_val is not None else np.nan)

    if len(times) == 0:
        st.warning(f"{uploaded_file.name}: няма Trackpoint данни.")
        return None

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "dist_m": dist_m,
        "elev_m": elev_m,
        "lat": lat_list,
        "lon": lon_list,
        "speed_raw": speed_raw,
    })

    # Сортираме по време
    df = df.sort_values("time").reset_index(drop=True)

    # Време в секунди от началото
    t0 = df["time"].iloc[0]
    df["sec"] = (df["time"] - t0).dt.total_seconds()

    # Ако нямаме DistanceMeters, смятаме дистанцията от GPS
    if df["dist_m"].isna().all():
        dist_vals = []
        total_dist = 0.0
        prev_lat, prev_lon = None, None
        for lat, lon in zip(df["lat"], df["lon"]):
            if prev_lat is not None and not np.isnan(lat) and not np.isnan(lon):
                d = haversine_distance(prev_lat, prev_lon, lat, lon)
                total_dist += d
            dist_vals.append(total_dist)
            prev_lat, prev_lon = lat, lon
        df["dist_m"] = dist_vals
    else:
        # Запълваме евентуални дупки леко напред
        df["dist_m"] = df["dist_m"].fillna(method="ffill").fillna(method="bfill")

    # Запълваме височината (ако има кратки дупки)
    if df["elev_m"].notna().sum() > 0:
        df["elev_m"] = df["elev_m"].fillna(method="ffill").fillna(method="bfill")

    # Първоначална стъпка по време и дистанция
    df["dt"] = df["sec"].diff()
    df["ddist"] = df["dist_m"].diff()

    # Филтър за време
    df = df[(df["dt"] > 0) & (df["dt"] < MAX_DT_SEC)].copy()
    df.reset_index(drop=True, inplace=True)

    # Пресмятаме скоростта
    df["speed_mps"] = df["speed_raw"]
    # Ако няма скорост, ползваме ddist/dt
    mask_no_speed = df["speed_mps"].isna()
    df.loc[mask_no_speed, "speed_mps"] = df.loc[mask_no_speed, "ddist"] / df.loc[mask_no_speed, "dt"]

    # Отново филтрираме време и дистанция след евентуални NaN
    df = df.dropna(subset=["sec", "dist_m", "elev_m", "speed_mps"])
    df = df[df["speed_mps"] >= 0]
    df = df[df["speed_mps"] <= MAX_SPEED_MPS]
    df.reset_index(drop=True, inplace=True)

    # Пресмятаме dt и ddist отново, за да са консистентни
    df["dt"] = df["sec"].diff()
    df["ddist"] = df["dist_m"].diff()

    return df


def preprocess_slopes(df):
    """
    Изглажда височината и изчислява наклона (%) за всеки интервал.
    """
    if "elev_m" not in df.columns:
        return None

    # медианно изглаждане (3-5 точки)
    df["elev_smooth"] = df["elev_m"].rolling(window=5, center=True, min_periods=1).median()

    df["dh"] = df["elev_smooth"].diff()
    df["slope"] = 0.0
    valid = df["ddist"] > 0
    df.loc[valid, "slope"] = 100.0 * df.loc[valid, "dh"] / df.loc[valid, "ddist"]
    # ограничаваме екстремните стойности
    df["slope"] = df["slope"].clip(-30.0, 30.0)

    return df


def build_segments(df):
    """
    Разделя активността на сегменти от SEGMENT_LENGTH_SEC.
    Връща DataFrame със сегменти (един ред = един сегмент).
    """
    if df is None or len(df) < 3:
        return pd.DataFrame()

    segments = []
    t_start = df["sec"].iloc[0]
    t_end = df["sec"].iloc[-1]

    seg_idx = 0
    t = t_start

    while t + SEGMENT_LENGTH_SEC <= t_end:
        t1 = t
        t2 = t + SEGMENT_LENGTH_SEC

        sub = df[(df["sec"] >= t1) & (df["sec"] < t2)]
        if len(sub) < 3:
            t += SEGMENT_LENGTH_SEC
            continue

        duration = sub["sec"].iloc[-1] - sub["sec"].iloc[0]
        if duration < MIN_SEG_DURATION_SEC:
            t += SEGMENT_LENGTH_SEC
            continue

        dist = sub["dist_m"].iloc[-1] - sub["dist_m"].iloc[0]
        if dist < MIN_SEG_DISTANCE_M:
            t += SEGMENT_LENGTH_SEC
            continue

        mean_speed = dist / duration
        if mean_speed < MIN_SEG_SPEED_MPS:
            t += SEGMENT_LENGTH_SEC
            continue

        # среден наклон - претеглен по дистанция
        w = sub["ddist"].clip(lower=0.0)
        if w.sum() > 0:
            mean_slope = np.average(sub["slope"], weights=w)
        else:
            mean_slope = sub["slope"].mean()

        segments.append({
            "seg_idx": seg_idx,
            "start_sec": t1,
            "duration": duration,
            "dist_m": dist,
            "mean_speed_mps": mean_speed,
            "mean_slope_pct": mean_slope,
        })

        seg_idx += 1
        t += SEGMENT_LENGTH_SEC

    return pd.DataFrame(segments)


def select_glide_segments(segments_df):
    """
    Избира сегменти, които отговарят на глайд условията:
    - текущият сегмент: наклон в [MIN_DOWNHILL_SLOPE, MAX_DOWNHILL_SLOPE]
    - предходният сегмент: наклон <= PREV_MIN_SLOPE
    Добавя norm_speed (нормализирана скорост спрямо наклона).
    """
    if segments_df.empty:
        return segments_df

    segs = segments_df.copy().reset_index(drop=True)
    segs["prev_slope_pct"] = segs["mean_slope_pct"].shift(1)

    cond_current = (segs["mean_slope_pct"] >= MIN_DOWNHILL_SLOPE) & (segs["mean_slope_pct"] <= MAX_DOWNHILL_SLOPE)
    cond_prev = segs["prev_slope_pct"] <= PREV_MIN_SLOPE

    glide = segs[cond_current & cond_prev].copy()

    if glide.empty:
        return glide

    # Нормализиране спрямо наклона – по-слабо влияние (2x по-малко от преди)
    slope_abs = np.abs(glide["mean_slope_pct"]).clip(lower=0.5)
    glide["norm_speed"] = glide["mean_speed_mps"] / ((slope_abs / 100.0) ** 0.25)

    return glide


def compute_activity_summary(name, df, glide_segments):
    """
    Връща речник с основните метрики за една активност.
    """
    if df is None or len(df) < 2:
        return {
            "Activity": name,
            "Distance_km": np.nan,
            "MovingTime_min": np.nan,
            "AvgSpeed_kmh": np.nan,
            "GlideSegments": 0,
            "GlideMetric": np.nan,
        }

    total_dist_m = df["dist_m"].iloc[-1] - df["dist_m"].iloc[0]
    total_time_s = df["sec"].iloc[-1] - df["sec"].iloc[0]

    distance_km = total_dist_m / 1000.0 if total_dist_m > 0 else np.nan
    moving_time_min = total_time_s / 60.0 if total_time_s > 0 else np.nan
    avg_speed_kmh = (distance_km / (moving_time_min / 60.0)) if (moving_time_min and moving_time_min > 0) else np.nan

    n_glide = len(glide_segments)
    if n_glide >= MIN_GLADE_SEGMENTS_PER_ACTIVITY:
        glide_metric = glide_segments["norm_speed"].median()
    else:
        glide_metric = np.nan

    return {
        "Activity": name,
        "Distance_km": distance_km,
        "MovingTime_min": moving_time_min,
        "AvgSpeed_kmh": avg_speed_kmh,
        "GlideSegments": n_glide,
        "GlideMetric": glide_metric,
    }


# ==============================
# STREAMLIT UI
# ==============================

st.set_page_config(page_title="Ski Glide – Коeфициент на плъзгане", layout="wide")

st.title("🎿 Ski Glide – модел за коефициент на плъзгане между активности")

st.markdown(
    """
Качи няколко `.tcx` файла от ски бягане.  
Приложението ще:
- филтрира и изглади данните,
- открие глайд сегменти (спускане + следващ участък),
- изчисли **индекс на плъзгане (GlideIndex)** за всяка активност,
- оцени каква би била скоростта при **стандартно ниво на плъзгане**.
"""
)

uploaded_files = st.file_uploader(
    "Качи един или повече TCX файла от ски бягане",
    type=["tcx"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("👉 Качи поне един `.tcx` файл, за да започнем.")
    st.stop()

activity_summaries = []
glide_details = {}  # име → glide_segments DataFrame

with st.spinner("Обработка на файловете..."):
    for file in uploaded_files:
        name = file.name

        df = parse_tcx(file)
        if df is None or len(df) < 5:
            st.warning(f"{name}: недостатъчно данни за анализ.")
            continue

        df = preprocess_slopes(df)
        if df is None:
            st.warning(f"{name}: липсва височина за изчисляване на наклон.")
            continue

        segments = build_segments(df)
        glide_segments = select_glide_segments(segments)

        summary = compute_activity_summary(name, df, glide_segments)
        activity_summaries.append(summary)
        glide_details[name] = glide_segments

if len(activity_summaries) == 0:
    st.error("Няма нито една активност с достатъчно валидни данни.")
    st.stop()

summary_df = pd.DataFrame(activity_summaries)

st.subheader("📊 Обобщение на активностите (преди стандартизация)")
st.dataframe(
    summary_df.round(3),
    use_container_width=True,
)

# Филтрираме тези с валиден GlideMetric
valid_glide_df = summary_df.dropna(subset=["GlideMetric"]).copy()
if valid_glide_df.empty:
    st.error("Няма активност с достатъчно глайд сегменти за надеждна оценка (GlideMetric).")
    st.stop()

# ==============================
# ИЗБОР НА РЕФЕРЕНТНА АКТИВНОСТ / СТАНДАРТ
# ==============================

st.sidebar.header("Настройки на стандарта")

options = ["Медиана от всички активности"] + list(valid_glide_df["Activity"])
ref_choice = st.sidebar.selectbox(
    "Стандартни условия (референтна активност):",
    options,
)

if ref_choice == "Медиана от всички активности":
    baseline_glide = valid_glide_df["GlideMetric"].median()
    ref_label = "Медиана от всички"
else:
    baseline_glide = valid_glide_df.loc[valid_glide_df["Activity"] == ref_choice, "GlideMetric"].values[0]
    ref_label = ref_choice

st.sidebar.markdown(f"**GlideMetric (стандарт):** `{baseline_glide:.4f}`")

# ==============================
# ИНДЕКС НА ПЛЪЗГАНЕ И СКОРОСТ ПРИ СТАНДАРТНИ УСЛОВИЯ
# ==============================

summary_df["GlideIndex"] = summary_df["GlideMetric"] / baseline_glide

# Скорост при стандартни условия
summary_df["StdSpeed_kmh"] = summary_df["AvgSpeed_kmh"] / summary_df["GlideIndex"]
summary_df["DeltaSpeed_kmh"] = summary_df["AvgSpeed_kmh"] - summary_df["StdSpeed_kmh"]
summary_df["DeltaSpeed_%"] = 100.0 * (summary_df["AvgSpeed_kmh"] / summary_df["StdSpeed_kmh"] - 1.0)

st.subheader("🏁 Индекс на плъзгане и скорост при стандартни условия")

display_cols = [
    "Activity",
    "Distance_km",
    "MovingTime_min",
    "AvgSpeed_kmh",
    "GlideSegments",
    "GlideMetric",
    "GlideIndex",
    "StdSpeed_kmh",
    "DeltaSpeed_kmh",
    "DeltaSpeed_%",
]
st.dataframe(summary_df[display_cols].round(3), use_container_width=True)

# ==============================
# ВИЗУАЛИЗАЦИЯ – BAR CHART
# ==============================

st.subheader("📈 Сравнение на плъзгането между активностите")

chart_df = summary_df.dropna(subset=["GlideIndex"]).copy()
chart_df = chart_df.set_index("Activity")[["GlideIndex"]]

st.bar_chart(chart_df)

# ==============================
# ДЕТАЙЛИ ЗА ОТДЕЛНА АКТИВНОСТ
# ==============================

st.subheader("🔍 Детайлен преглед на глайд сегментите")

act_for_details = st.selectbox(
    "Избери активност за детайлен преглед на глайд сегментите:",
    list(glide_details.keys()),
)

details_df = glide_details.get(act_for_details, pd.DataFrame())
if details_df.empty:
    st.info("За тази активност няма глайд сегменти (по зададените критерии).")
else:
    st.markdown(
        "Показани са само сегментите, които отговарят на глайд условията "
        f"({MIN_DOWNHILL_SLOPE}% до {MAX_DOWNHILL_SLOPE}%, предходен сегмент ≤ {PREV_MIN_SLOPE}%)."
    )
    st.dataframe(details_df.round(4), use_container_width=True)
