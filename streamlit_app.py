import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

# ---------------------------------------------------------
# НАСТРОЙКИ ПО ПОДРАЗБИРАНЕ (фиксирани – не се показват в UI)
# ---------------------------------------------------------
T_SEG = 7.0            # дължина на сегмента [s]
MIN_D_SEG = 5.0        # минимум хоризонтална дистанция [m]
MIN_T_SEG = 4.0        # минимум продължителност [s]
MAX_ABS_SLOPE = 15.0   # макс. наклон [%]
V_JUMP_KMH = 15.0      # праг за "скачане" на скоростта между сегменти
V_JUMP_MIN = 20.0      # гледаме спайкове само над тази скорост [km/h]

GLIDE_POLY_DEG = 2     # степен на полинома за плъзгаемост (1 или 2)
SLOPE_POLY_DEG = 2     # степен на полинома за наклон (1 или 2)
DAMP_GLIDE = 1.0       # омекотяване на коефициента на плъзгаемост (0–1)

V_CRIT = 20.0          # фиксирана критична скорост [km/h] – нужна за силни спускания


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
        # общ случай – просто показваме коефициентите
        return " + ".join(
            f"{fmt_coef(c)}·{var}^{p}"
            for p, c in zip(range(deg, -1, -1), coeffs)
        )


# ---------------------------------------------------------
# TCX PARSER
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

            rows.append({
                "activity": activity_label,
                "time": time,
                "lat": lat,
                "lon": lon,
                "elev": elev,
                "dist": dist,
            })

    if not rows:
        return pd.DataFrame(columns=["activity", "time", "lat", "lon", "elev", "dist"])

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
# СЕГМЕНТИРАНЕ НА 7 s
# ---------------------------------------------------------
def build_segments(df_activity, activity_label):
    if df_activity.empty:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh"
        ])

    df_activity = df_activity.sort_values("time").reset_index(drop=True)

    times = df_activity["time"].to_numpy()
    elevs = df_activity["elev"].to_numpy()
    dists = df_activity["dist"].to_numpy()

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

        seg_rows.append({
            "activity": activity_label,
            "seg_idx": seg_idx,
            "t_start": pd.to_datetime(t0),
            "t_end": pd.to_datetime(t1),
            "dt_s": float(dt),
            "d_m": float(d_m),
            "slope_pct": float(slope) if not np.isnan(slope) else np.nan,
            "v_kmh": float(v_kmh),
        })

        seg_idx += 1
        start_idx = end_idx

    if not seg_rows:
        return pd.DataFrame(columns=[
            "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
            "slope_pct", "v_kmh"
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


def compute_glide_coefficients(seg, glide_poly):
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
# ОБОБЩЕНИЕ ЗА UI ТАБЛИЦАТА
# ---------------------------------------------------------
def build_activity_summary(segments_f, train_glide, seg_glide, seg_slope, glide_coeffs):
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
    K_glide_df = pd.DataFrame(
        {"activity": list(glide_coeffs.keys()),
         "K_glide": list(glide_coeffs.values())}
    )
    summary = summary.merge(K_glide_df, on="activity", how="left")

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

    # 6) Ефективен коефициент от наклона
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
            "K_slope_eff",
        ]
    ]

    # Преименуваме колоните за UI (български етикети)
    summary = summary.rename(columns={
        "activity": "Активност",
        "slope_glide_mean": "Среден наклон на спусканията за модел [%]",
        "v_glide_train_mean": "Средна скорост на спусканията за модел [km/h]",
        "K_glide": "Коефициент плъзгаемост K_glide",
        "v_real_mean": "Средна реална скорост [km/h]",
        "v_glide_mean": "Средна скорост след плъзгаемост [km/h]",
        "v_flat_mean": "Средна скорост еквивалентна на равно [km/h]",
        "K_slope_eff": "Ефективен коефициент наклон K_slope",
    })

    return summary


# ---------------------------------------------------------
# STREAMLIT APP – ИЗЧИСТЕН UI
# ---------------------------------------------------------
st.set_page_config(page_title="Ski Glide & Slope Model", layout="wide")
st.title("Модел за плъзгаемост и наклон при ски бягане")

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
    glide_coeffs = compute_glide_coefficients(segments_f, glide_poly)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

# 5) Модел за наклон
flat_refs = compute_flat_ref_speeds(seg_glide)
slope_train = get_slope_training_data(seg_glide, flat_refs)
raw_slope_poly = fit_slope_poly(slope_train)

if raw_slope_poly is None:
    slope_poly = None
    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_CRIT)
else:
    # изместване така, че F(0)=1
    F0 = float(raw_slope_poly(0.0))
    offset = F0 - 1.0

    # взимаме копие на коефициентите и коригираме свободния член
    coeffs = raw_slope_poly.coefficients.copy()
    coeffs[-1] -= offset           # последният е свободният член (s^0)
    slope_poly = np.poly1d(coeffs) # нов полином с F(0) = 1

    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_CRIT)

# 6) Обобщена таблица по активности
summary_df = build_activity_summary(
    segments_f, train_glide, seg_glide, seg_slope, glide_coeffs
)

st.subheader("Обобщение по активности")
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
# ЕКСПОРТ НА СЕГМЕНТИТЕ
# ---------------------------------------------------------
st.subheader("Експорт на сегментите (след двете модулации)")

export_cols = [
    "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
    "slope_pct", "v_kmh", "valid_basic", "K_glide", "v_glide", "v_flat_eq"
]
available_export_cols = [c for c in export_cols if c in seg_slope.columns]
export_df = seg_slope[available_export_cols].copy()

csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Свали сегментите като CSV",
    data=csv_data,
    file_name="segments_glide_slope.csv",
    mime="text/csv"
)
