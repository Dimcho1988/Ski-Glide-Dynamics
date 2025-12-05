import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

from cs_modulator import (
    apply_cs_modulation,
    calibrate_k_for_target_t90,
    predict_t90_for_reference,
)

# ---------------------------------------------------------
# НАСТРОЙКИ (фиксирани прагове, но V_crit и DAMP_GLIDE са в UI)
# ---------------------------------------------------------
T_SEG = 7.0            # дължина на сегмента [s]
MIN_D_SEG = 5.0        # минимум хоризонтална дистанция [m]
MIN_T_SEG = 4.0        # минимум продължителност [s]
MAX_ABS_SLOPE = 15.0   # макс. наклон [%]
V_JUMP_KMH = 15.0      # праг за "скачане" на скоростта между сегменти
V_JUMP_MIN = 20.0      # гледаме спайкове само над тази скорост [km/h]

GLIDE_POLY_DEG = 2     # степен на полинома за плъзгаемост
SLOPE_POLY_DEG = 2     # степен на полинома за наклон

# Зонна система като % от критичната скорост
ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


# ---------------------------------------------------------
# ВСПОМОГАТЕЛНИ ФУНКЦИИ
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def poly_to_str(poly, var="s"):
    """Форматира np.poly1d като четим стринг (до квадратична степен)."""
    if poly is None:
        return "няма модел (недостатъчно данни)"

    coeffs = poly.coefficients
    deg = poly.order

    def fmt_coef(c):
        return f"{c:.4f}"

    if deg == 2:
        a, b, c = coeffs
        return (f"{fmt_coef(a)}·{var}² "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}·{var} "
                f"{'+ ' if c >= 0 else '- '}{fmt_coef(abs(c))}")
    elif deg == 1:
        a, b = coeffs
        return (f"{fmt_coef(a)}·{var} "
                f"{'+ ' if b >= 0 else '- '}{fmt_coef(abs(b))}")
    else:
        return " + ".join(
            f"{fmt_coef(c)}·{var}^{p}"
            for p, c in zip(range(deg, -1, -1), coeffs)
        )


def seconds_to_hhmmss(seconds: float) -> str:
    """Превръща секунди във формат ч:мм:сс."""
    if pd.isna(seconds):
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h}:{m:02d}:{sec:02d}"
def clean_speed_for_cs(g, v_max_cs=50.0):
    """
    Чисти скоростта преди CS-модулацията:
      - клипва v_flat_eq в [0, v_max_cs]
      - за сегментите със speed_spike=True прави линейна интерполация
        между най-близките 'чисти' сегменти.
    g е под-DataFrame за дадена активност.
    """
    v = g["v_flat_eq"].to_numpy(dtype=float)

    # 1) глобален клип – отрязваме абсурдни стойности
    v = np.clip(v, 0.0, v_max_cs)

    # 2) интерполация върху спайковете
    if "speed_spike" in g.columns:
        is_spike = g["speed_spike"].to_numpy(dtype=bool)
    else:
        is_spike = np.zeros_like(v, dtype=bool)

    if not is_spike.any():
        return v  # няма спайкове

    v_clean = v.copy()
    idx = np.arange(len(v_clean))

    good = ~is_spike
    if good.sum() == 0:
        # в крайен случай – ако всичко е спайк, връщаме оригинала
        return v_clean
    if good.sum() == 1:
        # само една добра точка – всички спайкове стават равни на нея
        v_clean[is_spike] = v_clean[good][0]
        return v_clean

    # линейна интерполация по индекса
    v_clean[is_spike] = np.interp(idx[is_spike], idx[good], v_clean[good])
    return v_clean


# ---------------------------------------------------------
# TCX PARSER – с пулс
# ---------------------------------------------------------
def parse_tcx(file, activity_label):
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    rows = []
    for lap in root.findall(".//tcx:Lap", ns):
        for tp in lap.findall(".//tcx:Trackpoint", ns):
            t_el = tp.find("tcx:Time", ns)
            if t_el is None:
                continue
            time = datetime.fromisoformat(t_el.text.replace("Z", "+00:00"))

            pos_el = tp.find("tcx:Position", ns)
            lat = lon = None
            if pos_el is not None:
                lat_el = pos_el.find("tcx:LatitudeDegrees", ns)
                lon_el = pos_el.find("tcx:LongitudeDegrees", ns)
                if lat_el is not None and lon_el is not None:
                    lat = float(lat_el.text)
                    lon = float(lon_el.text)

            alt_el = tp.find("tcx:AltitudeMeters", ns)
            elev = float(alt_el.text) if alt_el is not None else None

            dist_el = tp.find("tcx:DistanceMeters", ns)
            dist = float(dist_el.text) if dist_el is not None else None

            hr_el = tp.find(".//tcx:HeartRateBpm/tcx:Value", ns)
            hr = float(hr_el.text) if hr_el is not None else np.nan

            rows.append({
                "activity": activity_label,
                "time": time,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "dist": dist,
                "hr": hr,
            })

    if not rows:
        return pd.DataFrame(columns=["activity", "time", "lat", "lon", "elev", "dist", "hr"])

    df = pd.DataFrame(rows)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ако няма дистанция – смятаме грубо от lat/lon
    if df["dist"].isna().all():
        df["dist"] = 0.0
        for i in range(1, len(df)):
            if None in (df.at[i-1, "lat"], df.at[i-1, "lon"],
                        df.at[i, "lat"], df.at[i, "lon"]):
                df.at[i, "dist"] = df.at[i-1, "dist"]
                continue
            d = haversine(
                df.at[i-1, "lat"], df.at[i-1, "lon"],
                df.at[i, "lat"], df.at[i, "lon"]
            )
            df.at[i, "dist"] = df.at[i-1, "dist"] + d

    df["dist"] = df["dist"].ffill()
    return df


# ---------------------------------------------------------
# СЕГМЕНТИРАНЕ НА 7 s (с hr_mean)
# ---------------------------------------------------------
def build_segments(df_activity, activity_label):
    if df_activity.empty:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    df_activity = df_activity.sort_values("time").reset_index(drop=True)

    times = df_activity["time"].to_numpy()
    elevs = df_activity["elev"].to_numpy()
    dists = df_activity["dist"].to_numpy()
    hrs = df_activity["hr"].to_numpy()

    n = len(df_activity)
    start_idx = 0
    seg_idx = 0
    seg_rows = []

    while start_idx < n - 1:
        t0 = times[start_idx]

        end_idx = start_idx + 1
        while end_idx < n:
            dt_tmp = (times[end_idx] - t0) / np.timedelta64(1, "s")
            if dt_tmp >= T_SEG:
                break
            end_idx += 1

        if end_idx >= n:
            break

        t1 = times[end_idx]
        dt = (t1 - t0) / np.timedelta64(1, "s")

        d0 = dists[start_idx]
        d1 = dists[end_idx]
        elev0 = elevs[start_idx]
        elev1 = elevs[end_idx]
        d_m = max(0.0, d1 - d0)

        if dt < MIN_T_SEG or d_m < MIN_D_SEG:
            start_idx = end_idx
            continue

        if elev0 is None or elev1 is None or np.isnan(elev0) or np.isnan(elev1):
            slope = np.nan
        else:
            slope = (elev1 - elev0) / d_m * 100.0 if d_m > 0 else np.nan

        v_kmh = (d_m / dt) * 3.6
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx + 1]))

        seg_rows.append({
            "activity": activity_label,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0),
            "t_end": pd.to_datetime(t1),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if not np.isnan(slope) else np.nan,
            "v_kmh": float(v_kmh),
            "hr_mean": hr_mean
        })

        seg_idx += 1
        start_idx = end_idx

    if not seg_rows:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh", "hr_mean"
        ])

    return pd.DataFrame(seg_rows)


# ---------------------------------------------------------
# ФИЛТРИ ЗА НЕРЕАЛИСТИЧНИ СЕГМЕНТИ
# ---------------------------------------------------------
def apply_basic_filters(segments):
    seg = segments.copy()

    valid_slope = seg["slope_pct"].between(-MAX_ABS_SLOPE, MAX_ABS_SLOPE)
    valid_slope &= seg["slope_pct"].notna()
    seg["valid_basic"] = valid_slope

    def mark_speed_spikes(group):
        group = group.sort_values("seg_idx").copy()
        spike = np.zeros(len(group), dtype=bool)
        v = group["v_kmh"].values
        for i in range(1, len(group)):
            dv = abs(v[i] - v[i-1])
            vmax = max(v[i], v[i-1])
            if dv > V_JUMP_KMH and vmax > V_JUMP_MIN:
                spike[i] = True
        group["speed_spike"] = spike
        return group

    seg = seg.groupby("activity", group_keys=False).apply(mark_speed_spikes)
    seg["speed_spike"] = seg["speed_spike"].fillna(False)
    seg.loc[seg["speed_spike"], "valid_basic"] = False

    return seg


# ---------------------------------------------------------
# МОДЕЛ ЗА ПЛЪЗГАЕМОСТ (GLIDE) – множител K_glide
# ---------------------------------------------------------
def get_glide_training_segments(seg):
    """
    Сегменти за модел на плъзгаемостта:
    - сегментът и предишният са валидни
    - и двата имат наклон <= -5%  (спускания)
    """
    df = seg.copy()
    df["prev_slope"] = df.groupby("activity")["slope_pct"].shift(1)
    df["prev_valid"] = df.groupby("activity")["valid_basic"].shift(1)

    cond = (
        df["valid_basic"] &
        df["prev_valid"].fillna(False) &
        (df["slope_pct"] <= -5.0) &
        (df["prev_slope"] <= -5.0)
    )

    train = df[cond].copy()
    return train


def fit_glide_poly(train_df):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["v_kmh"].values.astype(float)
    if len(x) <= GLIDE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, GLIDE_POLY_DEG)
    return np.poly1d(coeffs)


def compute_glide_coefficients(seg, glide_poly, DAMP_GLIDE):
    """
    Връща dict activity -> K_glide.
    1) k_raw = V_model / V_real
    2) клипваме k_raw в [0.90, 1.25]
    3) омекотяваме към 1 с коефициент DAMP_GLIDE:
       K = 1 + α * (k_clipped - 1)
    """
    train = get_glide_training_segments(seg)
    if glide_poly is None or train.empty:
        return {}

    coeffs = {}
    for act, g in train.groupby("activity"):
        s_mean = g["slope_pct"].mean()
        v_real = g["v_kmh"].mean()
        if v_real <= 0:
            continue
        v_model = float(glide_poly(s_mean))
        if v_model <= 0:
            continue

        k_raw = v_model / v_real
        k_clipped = max(0.9, min(1.25, k_raw))
        k_final = 1.0 + DAMP_GLIDE * (k_clipped - 1.0)

        coeffs[act] = k_final

    return coeffs


def apply_glide_modulation(seg, glide_coeffs):
    seg = seg.copy()
    seg["K_glide"] = seg["activity"].map(glide_coeffs).fillna(1.0)
    seg["v_glide"] = seg["v_kmh"] * seg["K_glide"]
    return seg


# ---------------------------------------------------------
# МОДЕЛ ЗА НАКЛОН
# ---------------------------------------------------------
def compute_flat_ref_speeds(seg_glide):
    flat_refs = {}
    for act, g in seg_glide.groupby("activity"):
        mask_flat = g["slope_pct"].between(-1.0, 1.0) & g["valid_basic"]
        g_flat = g[mask_flat]
        if g_flat.empty:
            continue
        v_flat = g_flat["v_glide"].mean()
        if v_flat > 0:
            flat_refs[act] = v_flat
    return flat_refs


def get_slope_training_data(seg_glide, flat_refs):
    """
    Обучаващите данни за F(slope):
    - използваме сегменти с наклон в [-15, +15]%
    """
    df = seg_glide.copy()
    df["V_flat_ref"] = df["activity"].map(flat_refs)
    mask = (
        df["valid_basic"] &
        df["slope_pct"].between(-15.0, 15.0) &
        df["V_flat_ref"].notna() &
        (df["v_glide"] > 0)
    )
    train = df[mask].copy()
    if train.empty:
        return pd.DataFrame(columns=["slope_pct", "F"])
    train["F"] = train["V_flat_ref"] / train["v_glide"]
    return train[["slope_pct", "F"]]


def fit_slope_poly(train_df):
    if train_df.empty:
        return None
    x = train_df["slope_pct"].values.astype(float)
    y = train_df["F"].values.astype(float)
    if len(x) <= SLOPE_POLY_DEG:
        return None
    coeffs = np.polyfit(x, y, SLOPE_POLY_DEG)
    return np.poly1d(coeffs)


def apply_slope_modulation(seg_glide, slope_poly, V_crit):
    df = seg_glide.copy()
    if slope_poly is None:
        df["v_flat_eq"] = df["v_glide"]
        return df

    slopes = df["slope_pct"].values.astype(float)
    F_vals = slope_poly(slopes)
    F_vals = np.clip(F_vals, 0.7, 1.7)

    # |slope| <= 1% -> F = 1
    mask_mid = np.abs(slopes) <= 1.0
    F_vals[mask_mid] = 1.0
    # спускане <-1% -> F<=1
    mask_down = slopes < -1.0
    F_vals[mask_down] = np.minimum(F_vals[mask_down], 1.0)
    # изкачване >1% -> F>=1
    mask_up = slopes > 1.0
    F_vals[mask_up] = np.maximum(F_vals[mask_up], 1.0)

    v_flat_eq = df["v_glide"].values * F_vals

    # силни спускания под -3% -> 70% от V_crit
    if V_crit is not None and V_crit > 0:
        idx_below = df["slope_pct"] < -3.0
        v_flat_eq[idx_below] = 0.7 * V_crit

    df["v_flat_eq"] = v_flat_eq
    return df


# ---------------------------------------------------------
# ОБОБЩЕНИЕ ЗА UI ТАБЛИЦАТА ПО АКТИВНОСТИ
# ---------------------------------------------------------
def build_activity_summary(
    segments_f,
    train_glide,
    seg_glide,
    seg_slope,
    seg_slope_cs,
    glide_coeffs
):
    activities = sorted(segments_f["activity"].unique())
    summary = pd.DataFrame({"activity": activities})

    # 1) Среден наклон и скорост на обучаващите сегменти за плъзгаемост
    if not train_glide.empty:
        glide_train_agg = train_glide.groupby("activity").agg(
            slope_glide_mean=("slope_pct", "mean"),
            v_glide_train_mean=("v_kmh", "mean"),
        ).reset_index()
        summary = summary.merge(glide_train_agg, on="activity", how="left")
    else:
        summary["slope_glide_mean"] = np.nan
        summary["v_glide_train_mean"] = np.nan

    # 2) Коефициент K_glide по активност
    if glide_coeffs:
        K_glide_df = pd.DataFrame(
            {"activity": list(glide_coeffs.keys()),
             "K_glide": list(glide_coeffs.values())}
        )
        summary = summary.merge(K_glide_df, on="activity", how="left")
    else:
        summary["K_glide"] = np.nan

    # 3) Реална средна скорост (преди модулации)
    real_agg = segments_f[segments_f["valid_basic"]].groupby("activity").agg(
        v_real_mean=("v_kmh", "mean")
    ).reset_index()
    summary = summary.merge(real_agg, on="activity", how="left")

    # 4) Средна скорост след плъзгаемост
    glide_agg = seg_glide[seg_glide["valid_basic"]].groupby("activity").agg(
        v_glide_mean=("v_glide", "mean")
    ).reset_index()
    summary = summary.merge(glide_agg, on="activity", how="left")

    # 5) Средна скорост след наклон (еквивалентна на равно)
    slope_agg = seg_slope[seg_slope["valid_basic"]].groupby("activity").agg(
        v_flat_mean=("v_flat_eq", "mean")
    ).reset_index()
    summary = summary.merge(slope_agg, on="activity", how="left")

    # 6) Средна скорост след CS модулация
    cs_agg = seg_slope_cs[seg_slope_cs["valid_basic"]].groupby("activity").agg(
        v_flat_cs_mean=("v_flat_eq_cs", "mean")
    ).reset_index()
    summary = summary.merge(cs_agg, on="activity", how="left")

    # 7) Ефективен коефициент наклон
    summary["K_slope_eff"] = summary["v_flat_mean"] / summary["v_glide_mean"]

    # Подреждаме колоните
    summary = summary[
        [
            "activity",
            "slope_glide_mean",
            "v_glide_train_mean",
            "K_glide",
            "v_real_mean",
            "v_glide_mean",
            "v_flat_mean",
            "v_flat_cs_mean",
            "K_slope_eff",
        ]
    ]

    # Преименуване за UI
    summary = summary.rename(columns={
        "activity": "Активност",
        "slope_glide_mean": "Среден наклон на спусканията за модел [%]",
        "v_glide_train_mean": "Средна скорост на спусканията за модел [km/h]",
        "K_glide": "Коефициент плъзгаемост K_glide",
        "v_real_mean": "Средна реална скорост [km/h]",
        "v_glide_mean": "Средна скорост след плъзгаемост [km/h]",
        "v_flat_mean": "Средна скорост еквив. на равно [km/h]",
        "v_flat_cs_mean": "Средна скорост след CS модулация [km/h]",
        "K_slope_eff": "Ефективен коефициент наклон K_slope",
    })

    return summary

# ---------------------------------------------------------
# ЗОНИ ПО СКОРОСТ И ПУЛС (по твоята методика)
# ---------------------------------------------------------
def assign_speed_zones(seg_slope, V_crit):
    df = seg_slope.copy()
    if V_crit is None or V_crit <= 0:
        df["rel_crit"] = np.nan
        df["zone"] = None
        return df

    df["rel_crit"] = df["v_flat_eq"] / V_crit

    zones = []
    for r in df["rel_crit"]:
        if pd.isna(r):
            zones.append(None)
            continue
        z_name = None
        for i in range(len(ZONE_NAMES)):
            if ZONE_BOUNDS[i] <= r < ZONE_BOUNDS[i + 1]:
                z_name = ZONE_NAMES[i]
                break
        zones.append(z_name)
    df["zone"] = zones
    return df


def summarize_speed_zones(seg_zones):
    df = seg_zones.dropna(subset=["zone"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["zone", "n_segments", "total_time_s", "mean_v_flat_eq"])
    agg = df.groupby("zone").agg(
        n_segments=("seg_idx", "count"),
        total_time_s=("dt_s", "sum"),
        mean_v_flat_eq=("v_flat_eq", "mean"),
    ).reset_index()
    agg = agg.sort_values("zone")
    return agg


def compute_zone_hr_from_counts(seg_df, zone_counts):
    """
    seg_df: DataFrame със сегментите (вкл. и тези с |slope| > 15%)
    zone_counts: dict {zone: n_segments}, на база скоростните зони.

    Алгоритъм:
      - филтрираме само:
          * сегменти без speed_spike
          * сегменти с наличен hr_mean
      - сортираме тези сегменти по hr_mean ↑
      - за Z1 взимаме първите N1, за Z2 – следващите N2, ...
    """
    df_hr = seg_df.copy()
    if "speed_spike" in df_hr.columns:
        df_hr = df_hr[~df_hr["speed_spike"].fillna(False)]

    df_hr = df_hr.dropna(subset=["hr_mean"]).copy()

    if df_hr.empty:
        rows = [{"zone": z, "mean_hr_zone": np.nan} for z in ZONE_NAMES]
        return pd.DataFrame(rows)

    df_hr = df_hr.sort_values("hr_mean").reset_index(drop=True)

    results = []
    start_idx = 0
    for z in ZONE_NAMES:
        n = int(zone_counts.get(z, 0))
        if n <= 0 or start_idx >= len(df_hr):
            results.append({"zone": z, "mean_hr_zone": np.nan})
            continue
        end_idx = min(start_idx + n, len(df_hr))
        subset = df_hr.iloc[start_idx:end_idx]
        mean_hr = subset["hr_mean"].mean() if not subset.empty else np.nan
        results.append({"zone": z, "mean_hr_zone": mean_hr})
        start_idx = end_idx

    return pd.DataFrame(results)


def build_zone_speed_hr_table(seg_zones, V_crit, activity=None):
    """
    Връща таблица по зони:
      Зона | Брой сегменти | Време [ч:мм:сс] | Средна скорост | Среден пулс
    Ако activity е None -> всички активности.
    """
    if activity is not None:
        df = seg_zones[seg_zones["activity"] == activity].copy()
    else:
        df = seg_zones.copy()

    if df.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    speed_summary = summarize_speed_zones(df)
    if speed_summary.empty:
        return pd.DataFrame(columns=[
            "Зона", "Брой сегменти", "Време [ч:мм:сс]",
            "Средна скорост [km/h]", "Среден пулс [bpm]"
        ])

    zone_counts = dict(zip(speed_summary["zone"], speed_summary["n_segments"]))
    hr_summary = compute_zone_hr_from_counts(df, zone_counts)

    merged = pd.merge(speed_summary, hr_summary, on="zone", how="left")

    merged["time_hhmmss"] = merged["total_time_s"].apply(seconds_to_hhmmss)

    merged = merged.rename(columns={
        "zone": "Зона",
        "n_segments": "Брой сегменти",
        "time_hhmmss": "Време [ч:мм:сс]",
        "mean_v_flat_eq": "Средна скорост [km/h]",
        "mean_hr_zone": "Среден пулс [bpm]",
    })

    merged = merged[[
        "Зона", "Брой сегменти", "Време [ч:мм:сс]",
        "Средна скорост [km/h]", "Среден пулс [bpm]"
    ]]

    return merged


# ---------------------------------------------------------
# STREAMLIT APP – ИЗЧИСТЕН UI + КОНТРОЛЕН ПАНЕЛ + CS МОДЕЛ
# ---------------------------------------------------------
st.set_page_config(page_title="Ski Glide & Slope Model", layout="wide")
st.title("Модел за плъзгаемост, наклон и кислороден дълг при ски бягане")

# ---------- Sidebar: основни параметри ----------
st.sidebar.header("Параметри на наклона и плъзгаемостта")

V_crit = st.sidebar.number_input(
    "Критична скорост V_crit [km/h]",
    min_value=5.0,
    max_value=40.0,
    value=20.0,
    step=0.5
)

DAMP_GLIDE = st.sidebar.slider(
    "Омекотяване на плъзгаемостта α (0 = без ефект, 1 = пълен ефект)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.sidebar.markdown("---")

# ---------- Sidebar: CS / кислороден дълг ----------
st.sidebar.header("CS модел (кислороден „дълг“)")
use_vcrit_as_cs = st.sidebar.checkbox("Използвай V_crit като CS", value=True)

if use_vcrit_as_cs:
    CS = V_crit
else:
    CS = st.sidebar.number_input(
        "Критична скорост CS [km/h]",
        min_value=5.0,
        max_value=40.0,
        value=18.0,
        step=0.5
    )

tau_min = st.sidebar.number_input(
    "τ_min (s) – минимална константа",
    min_value=5.0,
    max_value=120.0,
    value=25.0,
    step=1.0
)

k_par = st.sidebar.number_input(
    "k – растеж на τ с отклонението",
    min_value=0.0,
    max_value=500.0,
    value=35.0,
    step=1.0
)

q_par = st.sidebar.number_input(
    "q – нелинейност на τ(Δv)",
    min_value=0.1,
    max_value=3.0,
    value=1.3,
    step=0.1
)

gamma_cs = st.sidebar.slider(
    "γ – каква част от ликвидацията „повдига“ скоростта",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05
)

st.sidebar.subheader("Калибрация по референтен сценарий")
ref_percent = st.sidebar.number_input(
    "Референтна интензивност (% от CS)",
    min_value=101.0,
    max_value=200.0,
    value=105.0,
    step=0.5
)
target_t90 = st.sidebar.number_input(
    "Желано t₉₀ (s)",
    min_value=10.0,
    max_value=1200.0,
    value=60.0,
    step=5.0
)
do_calibrate = st.sidebar.button("Приложи калибрация (пресметни k)")

st.caption(
    f"Текущи параметри: V_crit = {V_crit:.1f} km/h, CS = {CS:.1f} km/h, "
    f"α (DAMP_GLIDE) = {DAMP_GLIDE:.2f}"
)

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла:",
    type=["tcx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# 1) Парсване на файловете
all_points = []
for f in uploaded_files:
    label = f.name
    df_act = parse_tcx(f, label)
    if df_act.empty:
        continue
    all_points.append(df_act)

if not all_points:
    st.error("Не успях да извлека данни от файловете.")
    st.stop()

points = pd.concat(all_points, ignore_index=True)

# 2) Сегментиране
seg_list = []
for act, g in points.groupby("activity"):
    seg_df = build_segments(g, act)
    if not seg_df.empty:
        seg_list.append(seg_df)

segments = pd.concat(seg_list, ignore_index=True) if seg_list else pd.DataFrame()
if segments.empty:
    st.error("Не успях да създам сегменти. Провери TCX файловете.")
    st.stop()

# 3) Базови филтри
segments_f = apply_basic_filters(segments)

# 4) Модел за плъзгаемост
train_glide = get_glide_training_segments(segments_f)
glide_poly = fit_glide_poly(train_glide)

if glide_poly is None:
    glide_coeffs = {}
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)
else:
    glide_coeffs = compute_glide_coefficients(segments_f, glide_poly, DAMP_GLIDE)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

# 5) Модел за наклон (F(0)=1)
flat_refs = compute_flat_ref_speeds(seg_glide)
slope_train = get_slope_training_data(seg_glide, flat_refs)
raw_slope_poly = fit_slope_poly(slope_train)

if raw_slope_poly is None:
    slope_poly = None
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit)
else:
    F0 = float(raw_slope_poly(0.0))
    offset = F0 - 1.0
    coeffs = raw_slope_poly.coefficients.copy()
    coeffs[-1] -= offset  # корекция на свободния член => F(0)=1
    slope_poly = np.poly1d(coeffs)
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit)

# 5a) Почистване на v_flat_eq от нереалистични спайкове
seg_slope = seg_slope.sort_values(["activity", "t_start"]).reset_index(drop=True)
for act, g_act in seg_slope.groupby("activity"):
    v_clean = clean_speed_for_cs(g_act, v_max_cs=50.0)
    seg_slope.loc[g_act.index, "v_flat_eq"] = v_clean


# 5b) CS модулация върху v_flat_eq
if do_calibrate:
    k_par = calibrate_k_for_target_t90(CS, ref_percent, tau_min, q_par, target_t90)
    st.sidebar.success(f"Нов k = {k_par:.2f} (приложен)")

# добавяме time_s (кумулативно) по активност
seg_slope = seg_slope.sort_values(["activity", "t_start"]).reset_index(drop=True)
seg_slope["time_s"] = seg_slope.groupby("activity")["dt_s"].cumsum() - seg_slope["dt_s"]

cs_rows = []
for act, g in seg_slope.groupby("activity"):
    # 1) чистим скоростта: режем над v_max_cs и изглаждаме спайковете
    v_clean = clean_speed_for_cs(g, v_max_cs=50.0)
    dt_arr = g["dt_s"].to_numpy(dtype=float)

    # 2) прилагаме CS модела върху изчистената скорост
    out_cs = apply_cs_modulation(
        v=v_clean,
        dt=dt_arr,
        CS=CS,
        tau_min=tau_min,
        k_par=k_par,
        q_par=q_par,
        gamma=gamma_cs,
    )

    g_cs = g.copy()
    g_cs["v_flat_eq_cs"] = out_cs["v_mod"]
    g_cs["delta_v_plus_kmh"] = out_cs["delta_v_plus"]
    g_cs["r_kmh"] = out_cs["r"]
    g_cs["tau_s"] = out_cs["tau_s"]
    cs_rows.append(g_cs)

seg_slope_cs = pd.concat(cs_rows, ignore_index=True)

# CS диагностична метрика t90 за референтния сценарий
dv_ref, tau_ref_now, t90_now = predict_t90_for_reference(CS, ref_percent, tau_min, k_par, q_par)
st.caption(
    f"CS модел: Δv_ref = {dv_ref:.2f} km/h, τ_ref ≈ {tau_ref_now:.1f} s, "
    f"t₉₀ ≈ {t90_now:.0f} s при {ref_percent:.1f}% от CS."
)

# 6) Обобщена таблица по активности
summary_df = build_activity_summary(
    segments_f,
    train_glide,
    seg_glide,
    seg_slope,
    seg_slope_cs,
    glide_coeffs
)


st.subheader("Обобщение по активности (след нормализация по наклон и плъзгаемост)")
st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------------
# ГРАФИКА 1 – ПЛЪЗГАЕМОСТ: скорост vs наклон
# ---------------------------------------------------------
st.subheader("Зависимост по плъзгаемост: скорост vs наклон (спускания под -5%)")

if not train_glide.empty and glide_poly is not None:
    s_min = train_glide["slope_pct"].min()
    s_max = train_glide["slope_pct"].max()
    s_grid = np.linspace(s_min, s_max, 200)
    df_glide_curve = pd.DataFrame({
        "slope_pct": s_grid,
        "v_model": glide_poly(s_grid)
    })

    chart_points = alt.Chart(train_glide).mark_circle(size=30).encode(
        x=alt.X("slope_pct", title="Наклон [%]"),
        y=alt.Y("v_kmh", title="Скорост [km/h]"),
        color="activity:N"
    )
    chart_curve = alt.Chart(df_glide_curve).mark_line().encode(
        x="slope_pct",
        y="v_model"
    )
    st.altair_chart(chart_points + chart_curve, use_container_width=True)

    st.markdown(f"**Модел за плъзгаемост:**  v(s) = {poly_to_str(glide_poly, var='s')}")
else:
    st.info("Няма достатъчно данни за изграждане на модел за плъзгаемост.")

# ---------------------------------------------------------
# ГРАФИКА 2 – НАКЛОН: F(наклон)
# ---------------------------------------------------------
st.subheader("Зависимост по наклон: F(slope)")

if not slope_train.empty and slope_poly is not None:
    s_min2 = slope_train["slope_pct"].min()
    s_max2 = slope_train["slope_pct"].max()
    s_grid2 = np.linspace(s_min2, s_max2, 200)
    df_slope_curve = pd.DataFrame({
        "slope_pct": s_grid2,
        "F_model": slope_poly(s_grid2)
    })

    chart_points2 = alt.Chart(slope_train).mark_circle(size=30).encode(
        x=alt.X("slope_pct", title="Наклон [%]"),
        y=alt.Y("F", title="F = V_flat_ref / v_glide"),
    )
    chart_curve2 = alt.Chart(df_slope_curve).mark_line().encode(
        x="slope_pct",
        y="F_model"
    )
    st.altair_chart(chart_points2 + chart_curve2, use_container_width=True)

    st.markdown(f"**Модел за наклон:**  F(s) = {poly_to_str(slope_poly, var='s')}")
else:
    st.info("Няма достатъчно данни за изграждане на модел за наклона.")

# ---------------------------------------------------------
# ГРАФИКА 3 – CS модулация върху v_flat_eq
# ---------------------------------------------------------
st.subheader("CS модулация на еквивалентната скорост (по активности)")

act_list = sorted(seg_slope_cs["activity"].unique())
act_cs_selected = st.selectbox(
    "Избери активност за CS-графика:",
    act_list,
    key="cs_act_select"
)

g_plot = seg_slope_cs[seg_slope_cs["activity"] == act_cs_selected].copy()

if not g_plot.empty:
    base = alt.Chart(g_plot).encode(
        x=alt.X("time_s:Q", title="Време [s]")
    )

    line_orig = base.mark_line().encode(
        y=alt.Y("v_flat_eq:Q", title="Скорост [km/h]"),
        color=alt.value("#1f77b4")
    )
    line_cs = base.mark_line(strokeDash=[4, 4]).encode(
        y="v_flat_eq_cs:Q",
        color=alt.value("#ff7f0e")
    )

    st.altair_chart(line_orig + line_cs, use_container_width=True)

    st.caption("Плътна линия – v_flat_eq (само наклон+плъзгаемост); "
               "пунктирана – v_flat_eq_cs (допълнително CS-модулирана).")

# ---------------------------------------------------------
# ЗОНИ – СКОРОСТ + ПУЛС (ВСИЧКИ АКТИВНОСТИ)
# ---------------------------------------------------------
seg_zones = assign_speed_zones(seg_slope, V_crit)

st.subheader("Разпределение по зони – скорост и пулс (всички активности, без CS)")
zone_table_all = build_zone_speed_hr_table(seg_zones, V_crit, activity=None)
st.dataframe(zone_table_all, use_container_width=True)

# CS-зони: използваме v_flat_eq_cs като v_flat_eq
seg_slope_cs_for_zones = seg_slope_cs.copy()
seg_slope_cs_for_zones["v_flat_eq"] = seg_slope_cs_for_zones["v_flat_eq_cs"]
seg_zones_cs = assign_speed_zones(seg_slope_cs_for_zones, V_crit)

st.subheader("Разпределение по зони – скорост и пулс (всички активности, с CS)")
zone_table_all_cs = build_zone_speed_hr_table(seg_zones_cs, V_crit, activity=None)
st.dataframe(zone_table_all_cs, use_container_width=True)

# ---------------------------------------------------------
# ЗОНИ – СКОРОСТ + ПУЛС (ИЗБРАНА АКТИВНОСТ)
# ---------------------------------------------------------
st.subheader("Разпределение по зони – скорост и пулс (избрана активност)")

act_selected = st.selectbox(
    "Избери активност за зонен анализ:",
    act_list,
    key="zone_act_select"
)

zone_table_act = build_zone_speed_hr_table(seg_zones, V_crit, activity=act_selected)
st.markdown("**Без CS модулация:**")
st.dataframe(zone_table_act, use_container_width=True)

zone_table_act_cs = build_zone_speed_hr_table(seg_zones_cs, V_crit, activity=act_selected)
st.markdown("**С CS модулация:**")
st.dataframe(zone_table_act_cs, use_container_width=True)

# ---------------------------------------------------------
# ЕКСПОРТ НА СЕГМЕНТИТЕ
# ---------------------------------------------------------
st.subheader("Експорт на сегментите (след двете модулации + CS)")

export_cols = [
    "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
    "slope_pct", "v_kmh", "valid_basic", "speed_spike",
    "K_glide", "v_glide",
    "v_flat_eq", "v_flat_eq_cs",
    "time_s",
    "delta_v_plus_kmh", "r_kmh", "tau_s",
    "hr_mean"
]

available_export_cols = [c for c in export_cols if c in seg_slope_cs.columns]
export_df = seg_slope_cs[available_export_cols].copy()

csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Свали сегментите като CSV",
    data=csv_data,
    file_name="segments_glide_slope_cs_mod.csv",
    mime="text/csv"
)
