import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import math
import altair as alt

# ---------------------------------------------------------
# НАСТРОЙКИ ПО ПОДРАЗБИРАНЕ (ще се презапишат от sidebar)
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

# 6-зонна система като % от критичната скорост
ZONE_BOUNDS = [0.0, 0.75, 0.85, 0.95, 1.05, 1.15, np.inf]
ZONE_NAMES = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]


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


# ---------------------------------------------------------
# СЕГМЕНТИРАНЕ НА 7 s
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
        hr_mean = float(np.nanmean(hrs[start_idx:end_idx+1]))

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
# МОДЕЛ ЗА ПЛЪЗГАЕМОСТ (GLIDE)
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
    Връща dict activity -> K_glide (вече омекотен с DAMP_GLIDE).
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
        k_raw = v_model / v_real          # пълният коефициент
        k_damped = 1.0 + DAMP_GLIDE * (k_raw - 1.0)  # омекотен
        coeffs[act] = k_damped
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
    - използваме само сегменти с |slope| >= 1% (изключваме -1..+1)
    - диапазонът на наклона е [-3, 30]%.
    """
    df = seg_glide.copy()
    df["V_flat_ref"] = df["activity"].map(flat_refs)
    mask = (
        df["valid_basic"] &
        df["slope_pct"].between(-15.0, 15.0) &
        (np.abs(df["slope_pct"]) >= 1.0) &
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

    # 1) Без модулация около нулата: |slope| <= 1% -> F = 1
    mask_mid = np.abs(slopes) <= 1.0
    F_vals[mask_mid] = 1.0

    # 2) За наклон < -1% (леко спускане) искаме F <= 1
    mask_down = slopes < -1.0
    F_vals[mask_down] = np.minimum(F_vals[mask_down], 1.0)

    # 3) За наклон > 1% (изкачване) искаме F >= 1
    mask_up = slopes > 1.0
    F_vals[mask_up] = np.maximum(F_vals[mask_up], 1.0)

    v_flat_eq = df["v_glide"].values * F_vals

    # Сегменти под -3% ги приравняваме директно към 70% от V_crit
    if V_crit is not None and V_crit > 0:
        idx_below = df["slope_pct"] < -3.0
        v_flat_eq[idx_below] = 0.7 * V_crit

    df["v_flat_eq"] = v_flat_eq
    return df


# ---------------------------------------------------------
# ЗОНИ СПРЯМО КРИТИЧНА СКОРОСТ
# ---------------------------------------------------------
def assign_zones(df, V_crit):
    df = df.copy()
    if V_crit is None or V_crit <= 0:
        df["rel_crit"] = np.nan
        df["zone"] = None
        return df

    df["rel_crit"] = df["v_flat_eq"] / V_crit

    zones = []
    for r in df["rel_crit"]:
        z_name = None
        if np.isnan(r):
            zones.append(z_name)
            continue
        for i in range(len(ZONE_NAMES)):
            if ZONE_BOUNDS[i] <= r < ZONE_BOUNDS[i+1]:
                z_name = ZONE_NAMES[i]
                break
        zones.append(z_name)
    df["zone"] = zones
    return df


def summarize_zones(df):
    if df.empty:
        return pd.DataFrame(columns=["activity", "zone", "total_time_s", "mean_v_flat_eq", "mean_hr"])

    agg = df.dropna(subset=["zone"]).groupby(["activity", "zone"]).agg(
        total_time_s=("dt_s", "sum"),
        mean_v_flat_eq=("v_flat_eq", "mean"),
        mean_hr=("hr_mean", "mean"),
    ).reset_index()
    return agg


# ---------------------------------------------------------
# ОБОБЩЕНИ ТАБЛИЦИ ЗА МОДУЛАЦИИТЕ
# ---------------------------------------------------------
def summarize_glide(seg_glide):
    """Обобщение след първата модулация (плъзгаемост)."""
    df = seg_glide[seg_glide["valid_basic"]].copy()
    if df.empty:
        return pd.DataFrame(columns=[
            "activity", "v_real_mean", "v_glide_mean", "K_glide_const", "K_glide_eff"
        ])

    g = df.groupby("activity").agg(
        v_real_mean=("v_kmh", "mean"),
        v_glide_mean=("v_glide", "mean"),
        K_glide_const=("K_glide", "mean")
    ).reset_index()
    g["K_glide_eff"] = g["v_glide_mean"] / g["v_real_mean"]
    return g


def summarize_full(seg_glide, seg_slope):
    """Обобщение след двете модулации (плъзгаемост + наклон)."""
    g1 = summarize_glide(seg_glide)

    df2 = seg_slope[seg_slope["valid_basic"]].copy()
    if df2.empty:
        g1["v_glide_mean2"] = np.nan
        g1["v_flat_mean"] = np.nan
        g1["K_slope_eff"] = np.nan
        g1["K_total"] = np.nan
        return g1

    g2 = df2.groupby("activity").agg(
        v_glide_mean2=("v_glide", "mean"),
        v_flat_mean=("v_flat_eq", "mean"),
    ).reset_index()

    out = pd.merge(g1, g2, on="activity", how="left")

    out["K_slope_eff"] = out["v_flat_mean"] / out["v_glide_mean2"]
    out["K_total"] = out["v_flat_mean"] / out["v_real_mean"]
    return out


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
st.set_page_config(page_title="Ski Glide & Slope Model", layout="wide")
st.title("Модел за плъзгаемост и наклон – от нулата")

st.sidebar.header("Настройки")

# Критична скорост
V_crit_input = st.sidebar.number_input(
    "Критична скорост V_crit [km/h]",
    min_value=0.0, max_value=60.0, value=20.0, step=0.5
)

# Степен на полиномите
glide_deg = st.sidebar.selectbox(
    "Степен на полинома за плъзгаемост",
    options=[1, 2],
    index=1
)
slope_deg = st.sidebar.selectbox(
    "Степен на полинома за наклон",
    options=[1, 2],
    index=1
)

# Омекотяване на коефициента по плъзгаемост
glide_damp = st.sidebar.slider(
    "Омекотяване на коефициента на плъзгаемост (α)",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="α=1 – пълен ефект, α=0 – без корекция по плъзгаемост"
)

# презаписваме глобалните настройки
GLIDE_POLY_DEG = glide_deg
SLOPE_POLY_DEG = slope_deg
DAMP_GLIDE = glide_damp

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла:",
    type=["tcx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Качи поне един TCX файл, за да започнем.")
    st.stop()

# 1) Парсване – име на файла = име на активността
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

st.subheader("Сурови точки (пример)")
st.dataframe(points.head(20))

# 2) Сегментиране
seg_list = []
for act, g in points.groupby("activity"):
    seg_df = build_segments(g, act)
    seg_list.append(seg_df)

segments = pd.concat(seg_list, ignore_index=True) if seg_list else pd.DataFrame()
if segments.empty:
    st.error("Не успях да създам сегменти. Провери TCX файловете.")
    st.stop()

st.subheader("Сегменти преди филтриране (7 s, средна скорост и наклон)")
st.dataframe(
    segments[["activity", "seg_idx", "dt_s", "d_m", "slope_pct", "v_kmh", "hr_mean"]].head(30)
)

# 3) Базови филтри
segments_f = apply_basic_filters(segments)

st.subheader("Сегменти след базови филтри")
st.write(f"Общо сегменти: {len(segments_f)}, валидни: {segments_f['valid_basic'].sum()}")
st.dataframe(
    segments_f[["activity", "seg_idx", "slope_pct", "v_kmh", "valid_basic", "speed_spike"]].head(30)
)

# 4–7) МОДЕЛ ЗА ПЛЪЗГАЕМОСТ
train_glide = get_glide_training_segments(segments_f)
st.subheader("Сегменти за модел на плъзгаемостта (наклон ≤ -5% и предходен сегмент също ≤ -5%)")
st.write(f"Сегменти за обучение: {len(train_glide)}")
st.dataframe(train_glide[["activity", "seg_idx", "slope_pct", "v_kmh"]].head(30))

glide_poly = fit_glide_poly(train_glide)
if glide_poly is None:
    st.warning("Не успях да фитна полином за плъзгаемостта (твърде малко данни). v_glide = v_kmh.")
    glide_coeffs = {}
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)
else:
    st.write(f"Полином за плъзгаемост (степен {GLIDE_POLY_DEG}), коефициенти:", glide_poly.coefficients)
    glide_coeffs = compute_glide_coefficients(segments_f, glide_poly)
    st.write("Коефициенти на плъзгаемост по активност (след омекотяване):", glide_coeffs)
    seg_glide = apply_glide_modulation(segments_f, glide_coeffs)

    # визуализация на модела
    if not train_glide.empty:
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
        st.subheader("Модел за плъзгаемост: скорост vs наклон (спускания под -5%)")
        st.altair_chart(chart_points + chart_curve, use_container_width=True)

st.subheader("Сегменти с модулирана по плъзгаемост скорост")
st.dataframe(
    seg_glide[["activity", "seg_idx", "slope_pct", "v_kmh", "K_glide", "v_glide"]].head(30)
)

# Обобщена таблица след първата модулация
glide_summary = summarize_glide(seg_glide)
st.subheader("Обобщение по активности – първа модулация (плъзгаемост)")
st.dataframe(glide_summary)

# 8) МОДЕЛ ЗА НАКЛОН
flat_refs = compute_flat_ref_speeds(seg_glide)
st.subheader("Референтни скорости на равно (от v_glide) по активност")
st.write(flat_refs)

slope_train = get_slope_training_data(seg_glide, flat_refs)
st.subheader("Сегменти за модел на наклона (F = V_flat_ref / v_glide)")
st.write(f"Сегменти за обучение: {len(slope_train)}")
st.dataframe(slope_train.head(30))

slope_poly = fit_slope_poly(slope_train)
if slope_poly is None:
    st.warning("Не успях да фитна полином за наклона (твърде малко данни). v_flat_eq = v_glide.")
    seg_slope = apply_slope_modulation(seg_glide, None, V_crit_input)
else:
    st.write(f"Полином за наклон (степен {SLOPE_POLY_DEG}), коефициенти:", slope_poly.coefficients)
    # визуализация F(slope)
    if not slope_train.empty:
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
        st.subheader("Модел за наклон: F(slope)")
        st.altair_chart(chart_points2 + chart_curve2, use_container_width=True)

    seg_slope = apply_slope_modulation(seg_glide, slope_poly, V_crit_input)

st.subheader("Сегменти с модулирана по наклон скорост (еквивалентна на равно)")
st.dataframe(
    seg_slope[["activity", "seg_idx", "slope_pct", "v_glide", "v_flat_eq"]].head(30)
)

# Обобщена таблица след двете модулации
full_summary = summarize_full(seg_glide, seg_slope)
st.subheader("Обобщение по активности – след двете модулации (плъзгаемост + наклон)")
st.dataframe(full_summary)

# 9) ЗОНИ
seg_zones = assign_zones(seg_slope, V_crit_input)
zone_summary = summarize_zones(seg_zones)

st.subheader("Обобщение по зони и активности")
st.dataframe(zone_summary)

st.subheader("Общо за всички активности (агрегирано по зони)")
zone_summary_all = zone_summary.groupby("zone").agg(
    total_time_s=("total_time_s", "sum"),
    mean_v_flat_eq=("mean_v_flat_eq", "mean"),
    mean_hr=("mean_hr", "mean"),
).reset_index()
st.dataframe(zone_summary_all)

# Детайлен изглед по активност
st.subheader("Детайлен изглед на сегментите по активност")
act_list = sorted(seg_zones["activity"].unique())
act_selected = st.selectbox("Избери активност (име на файла):", act_list)

act_df = seg_zones[seg_zones["activity"] == act_selected].copy()
st.dataframe(
    act_df[[
        "activity", "seg_idx", "t_start", "dt_s", "d_m",
        "slope_pct", "v_kmh", "K_glide", "v_glide",
        "v_flat_eq", "rel_crit", "zone", "hr_mean"
    ]]
)

# Експорт – CSV
st.subheader("Експорт на всички сегменти (CSV)")
export_cols = [
    "activity", "seg_idx", "t_start", "t_end", "dt_s", "d_m",
    "slope_pct", "v_kmh", "valid_basic", "K_glide", "v_glide",
    "v_flat_eq", "rel_crit", "zone", "hr_mean"
]
export_df = seg_zones[export_cols].copy()
csv_data = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Свали сегментите като CSV",
    data=csv_data,
    file_name="segments_glide_slope_zones.csv",
    mime="text/csv"
)

