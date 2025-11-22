import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
from typing import List, Dict, Tuple, Optional

# =========================
# НАСТРОЙКИ (можеш да пипаш)
# =========================
SEG_LENGTH_SEC = 5.0          # T_seg
MIN_SEG_POINTS = 5
MIN_SEG_DIST_M = 5.0
MIN_SEG_TIME_S = 3.0
MAX_ABS_SLOPE_PERCENT = 30.0

# Диапазони за наклон
GLIDE_SLOPE_MIN = -15.0       # %
GLIDE_SLOPE_MAX = -5.0        # %
FLAT_SLOPE_ABS_MAX = 1.0      # %
DV_SLOPE_MIN = -3.0           # %
DV_SLOPE_MAX = 10.0           # %
DV_EXCLUDE_FLAT_ABS = 1.0     # % около 0, които изключваме

# Филтър за височина
MIN_ABS_DH_M = 0.3            # h_min
MAX_VERT_RATE_MS = 4.0        # g_max ≈ 4–5 m/s

# Максимална разумна скорост
V_MAX_KMH = 80.0

# Минимален брой сегменти за регресиите
MIN_SEG_GLIDE_MODEL = 30
MIN_SEG_DV_MODEL = 30

# Зона 1 горна граница (ratio)
R_Z1_HIGH = 0.80

# Зонални граници (ratio = V_eff / CS)
ZONE_BOUNDS = {
    "Z1": (0.0, 0.80),
    "Z2": (0.80, 1.00),
    "Z3": (1.00, 1.10),
    "Z4": (1.10, 1.20),
    "Z5": (1.20, 1.40),
    "Z6": (1.40, 10.0),
}

# =========================
# ПАРСВАНЕ НА TCX
# =========================

def parse_tcx(file) -> pd.DataFrame:
    """
    Връща DataFrame с колони:
    ['time_s', 'dist_m', 'alt_m']
    time_s – секунди от началото на активността
    """
    content = file.read()
    try:
        tree = ET.fromstring(content)
    except ET.ParseError:
        # опит с decode, ако е bytes
        tree = ET.fromstring(content.decode("utf-8"))

    ns = {
        "tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2",
    }

    # Някои TCX нямат namespace – fallback
    def _findall(elem, path):
        res = elem.findall(path, ns)
        if not res:
            res = elem.findall(path)
        return res

    trackpoints = _findall(tree, ".//tcx:Trackpoint")
    if not trackpoints:
        trackpoints = _findall(tree, ".//Trackpoint")

    times = []
    dists = []
    alts = []

    for tp in trackpoints:
        t_el = tp.find("tcx:Time", ns) or tp.find("Time")
        if t_el is None or t_el.text is None:
            continue
        time_str = t_el.text.strip()

        # Разчитаме ISO време (може и само да го поредим)
        times.append(pd.to_datetime(time_str))

        d_el = tp.find("tcx:DistanceMeters", ns) or tp.find("DistanceMeters")
        a_el = tp.find("tcx:AltitudeMeters", ns) or tp.find("AltitudeMeters")

        dist = float(d_el.text) if (d_el is not None and d_el.text) else np.nan
        alt = float(a_el.text) if (a_el is not None and a_el.text) else np.nan

        dists.append(dist)
        alts.append(alt)

    if len(times) == 0:
        return pd.DataFrame(columns=["time_s", "dist_m", "alt_m"])

    df = pd.DataFrame(
        {
            "time": times,
            "dist_m": dists,
            "alt_m": alts,
        }
    ).sort_values("time").reset_index(drop=True)

    # време в секунди от началото
    t0 = df["time"].iloc[0]
    df["time_s"] = (df["time"] - t0).dt.total_seconds()

    # Попълваме липсващи височини и дистанции (ако има)
    df["dist_m"] = df["dist_m"].interpolate().bfill().ffill()
    df["alt_m"] = df["alt_m"].interpolate().bfill().ffill()

    return df[["time_s", "dist_m", "alt_m"]]


# =========================
# PREPROCESSING
# =========================

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Почистване, сглаждане на височината, филтър за вертикален шум и нереалистични скорости.
    """
    if df.empty:
        return df.copy()

    df = df.sort_values("time_s").reset_index(drop=True)

    # Елементарни разлики
    df["dt"] = df["time_s"].diff()
    df["ddist"] = df["dist_m"].diff()
    df["dalt_raw"] = df["alt_m"].diff()

    # Премахване на невалидни интервали
    mask_valid = (
        (df["dt"] > 0)
        & (df["dt"] < 30.0)         # избягваме гигантски дупки във времето
        & (df["ddist"] >= 0)        # без обратно движение
    )
    df = df[mask_valid].reset_index(drop=True)

    if df.empty:
        return df

    # Сглаждане на височината – медианен филтър 3 точки
    alt_smooth = (
        df["alt_m"]
        .rolling(window=3, center=True, min_periods=1)
        .median()
    )
    df["alt_smooth"] = alt_smooth
    df["dalt"] = df["alt_smooth"].diff()

    # Вертикален шум / нереалистична промяна
    df["vert_rate"] = df["dalt"].abs() / df["dt"].replace(0, np.nan)
    # Филтър
    mask_vert = ~(
        (df["dalt"].abs() < MIN_ABS_DH_M)
        | (df["vert_rate"] > MAX_VERT_RATE_MS)
    )
    df = df[mask_vert].reset_index(drop=True)

    if df.empty:
        return df

    # Пресмятаме отново разликите след филтъра
    df["dt"] = df["time_s"].diff()
    df["ddist"] = df["dist_m"].diff()
    df["dalt"] = df["alt_smooth"].diff()

    # моментна скорост (km/h)
    df["speed_kmh"] = (df["ddist"] / df["dt"]).replace(np.inf, np.nan) * 3.6
    df["speed_kmh"] = df["speed_kmh"].clip(lower=0, upper=V_MAX_KMH)

    # Премахване на първия ред (няма dt/ddist)
    return df.iloc[1:].reset_index(drop=True)


# =========================
# СЕГМЕНТИРАНЕ
# =========================

def segment_activity(df: pd.DataFrame, activity_id: int) -> pd.DataFrame:
    """
    Делим по фиксирани 5 s сегменти, без припокриване.
    Връщаме по-сгъстен DF с по 1 ред на сегмент.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "activity_id", "seg_id", "t_start", "t_end",
            "dur_s", "dist_m", "dh_m", "slope_pct", "v_mean_kmh"
        ])

    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    n_seg = int(np.floor((t_max - t_min) / SEG_LENGTH_SEC))

    seg_rows = []

    for s in range(n_seg):
        seg_start = t_min + s * SEG_LENGTH_SEC
        seg_end = seg_start + SEG_LENGTH_SEC

        seg_df = df[(df["time_s"] >= seg_start) & (df["time_s"] < seg_end)]
        if seg_df.empty:
            continue

        n_points = len(seg_df)
        if n_points < MIN_SEG_POINTS:
            continue

        t0 = seg_df["time_s"].iloc[0]
        t1 = seg_df["time_s"].iloc[-1]
        dur = t1 - t0
        if dur < MIN_SEG_TIME_S:
            continue

        d0 = seg_df["dist_m"].iloc[0]
        d1 = seg_df["dist_m"].iloc[-1]
        dist = d1 - d0
        if dist < MIN_SEG_DIST_M:
            continue

        h0 = seg_df["alt_smooth"].iloc[0]
        h1 = seg_df["alt_smooth"].iloc[-1]
        dh = h1 - h0

        slope_pct = (dh / dist) * 100.0 if dist > 0 else 0.0

        if abs(slope_pct) > MAX_ABS_SLOPE_PERCENT:
            continue

        v_mean_kmh = dist / dur * 3.6

        seg_rows.append(
            dict(
                activity_id=activity_id,
                seg_id=s,
                t_start=t0,
                t_end=t1,
                dur_s=dur,
                dist_m=dist,
                dh_m=dh,
                slope_pct=slope_pct,
                v_mean_kmh=v_mean_kmh,
                n_points=n_points,
            )
        )

    if not seg_rows:
        return pd.DataFrame(columns=[
            "activity_id", "seg_id", "t_start", "t_end",
            "dur_s", "dist_m", "dh_m", "slope_pct", "v_mean_kmh"
        ])

    seg_df = pd.DataFrame(seg_rows)

    # дисперсия на моментната скорост вътре в сегмента (по желание)
    # за лекота засега не я използваме в моделите, но я изчисляваме
    seg_df["v_var"] = np.nan
    # (ако искаш, можем после да добавим и този критерий)

    return seg_df


# =========================
# МОДЕЛ 1 – GLIDE
# =========================

def build_glide_model(segments: pd.DataFrame, alpha_glide: float) -> Tuple[pd.DataFrame, Dict]:
    """
    Строи линейна регресия V = a*slope + b върху downhill сегментите.
    Връща:
      - сегментен DF със v_glide
      - речник с параметри на модела + индекси по активност
    """
    seg = segments.copy()
    if seg.empty:
        return seg, {
            "a": 0.0,
            "b": 0.0,
            "used_segments": 0,
            "glide_indices": {},
        }

    # Downhill сегменти
    downhill_mask = (seg["slope_pct"] >= GLIDE_SLOPE_MIN) & (seg["slope_pct"] <= GLIDE_SLOPE_MAX)
    seg["downhill"] = downhill_mask

    # Условие за инерция – предходният сегмент също downhill
    seg = seg.sort_values(["activity_id", "seg_id"]).reset_index(drop=True)
    prev_downhill = seg.groupby("activity_id")["downhill"].shift(1).fillna(False)
    seg["downhill_inertia"] = seg["downhill"] & prev_downhill

    D = seg[seg["downhill_inertia"]].copy()
    if D.empty:
        # няма Glide модел
        seg["v_glide_kmh"] = seg["v_mean_kmh"]
        return seg, {
            "a": 0.0,
            "b": 0.0,
            "used_segments": 0,
            "glide_indices": {aid: 1.0 for aid in seg["activity_id"].unique()},
        }

    # Премахване на аутлайъри по R = V / slope
    # (slope е отрицателен, но това ни е достатъчно като статистика)
    D["R"] = D["v_mean_kmh"] / D["slope_pct"]
    R_q05 = D["R"].quantile(0.05)
    R_q95 = D["R"].quantile(0.95)
    D_star = D[(D["R"] >= R_q05) & (D["R"] <= R_q95)].copy()

    if len(D_star) < MIN_SEG_GLIDE_MODEL:
        # няма достатъчно сегменти за регресия
        seg["v_glide_kmh"] = seg["v_mean_kmh"]
        return seg, {
            "a": 0.0,
            "b": 0.0,
            "used_segments": len(D_star),
            "glide_indices": {aid: 1.0 for aid in seg["activity_id"].unique()},
        }

    # Линейна регресия V = a*slope + b
    x = D_star["slope_pct"].values
    y = D_star["v_mean_kmh"].values
    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Индекси на плъзгаемост по активност
    glide_indices = {}
    for aid in seg["activity_id"].unique():
        D_A = D_star[D_star["activity_id"] == aid]
        if D_A.empty:
            glide_indices[aid] = 1.0
            continue

        # Времево-претеглена средна
        w = D_A["dur_s"]
        s_bar = np.average(D_A["slope_pct"], weights=w)
        V_real = np.average(D_A["v_mean_kmh"], weights=w)
        V_model = a * s_bar + b

        if V_model <= 0:
            glide_indices[aid] = 1.0
            continue

        K_raw = V_real / V_model

        # стабилност – ако е много крайно, зануляваме
        if (K_raw < 0.5) or (K_raw > 1.5):
            glide_indices[aid] = 1.0
        else:
            # омекотяване
            K_soft = 1.0 + alpha_glide * (K_raw - 1.0)
            glide_indices[aid] = K_soft

    # Прилагаме Glide корекция
    seg["K_glide_soft"] = seg["activity_id"].map(glide_indices)
    seg["v_glide_kmh"] = seg["v_mean_kmh"] / seg["K_glide_soft"].replace(0, 1.0)

    model_info = {
        "a": float(a),
        "b": float(b),
        "used_segments": int(len(D_star)),
        "glide_indices": glide_indices,
    }

    return seg, model_info


# =========================
# МОДЕЛ 2 – ΔV% И НАКЛОН
# =========================

def build_slope_model(seg: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Строи ΔV% = c0 + c1*slope + c2*slope^2 върху v_glide_kmh,
    връща сегментен DF с v_final_kmh и параметри.
    """
    df = seg.copy()
    if df.empty or "v_glide_kmh" not in df.columns:
        df["v_final_kmh"] = df.get("v_glide_kmh", df.get("v_mean_kmh", 0.0))
        return df, {"V_flat": None, "c0": 0.0, "c1": 0.0, "c2": 0.0, "used_segments": 0}

    # Референтна "равна" скорост V_flat (|slope| <= 1%)
    flat = df[df["slope_pct"].abs() <= FLAT_SLOPE_ABS_MAX]
    if flat["dur_s"].sum() >= 180:  # >= 3 минути
        V_flat = np.average(flat["v_glide_kmh"], weights=flat["dur_s"])
    else:
        # fallback – средна за всички
        V_flat = np.average(df["v_glide_kmh"], weights=df["dur_s"])

    # Сегменти за ΔV% модела
    cond_range = (df["slope_pct"] > DV_SLOPE_MIN) & (df["slope_pct"] < DV_SLOPE_MAX)
    cond_not_flat = df["slope_pct"].abs() > DV_EXCLUDE_FLAT_ABS
    S = df[cond_range & cond_not_flat].copy()

    if len(S) < MIN_SEG_DV_MODEL or V_flat <= 0:
        df["v_final_kmh"] = df["v_glide_kmh"]
        return df, {"V_flat": V_flat, "c0": 0.0, "c1": 0.0, "c2": 0.0, "used_segments": len(S)}

    # ΔV_real%
    S["dV_real_pct"] = (S["v_glide_kmh"] - V_flat) / V_flat * 100.0

    # Квадратична регресия
    x = S["slope_pct"].values
    X = np.vstack([np.ones_like(x), x, x ** 2]).T
    y = S["dV_real_pct"].values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1, c2 = coeffs

    # Прогнозна ΔV_model%
    df["dV_model_pct"] = c0 + c1 * df["slope_pct"] + c2 * (df["slope_pct"] ** 2)

    # Финална наклоново-коригирана скорост
    denom = 1.0 + df["dV_model_pct"] / 100.0
    denom = denom.replace(0, np.nan)
    df["v_final_kmh"] = df["v_glide_kmh"] / denom
    df["v_final_kmh"] = df["v_final_kmh"].replace([np.inf, -np.inf], np.nan).fillna(df["v_glide_kmh"])

    model_info = {
        "V_flat": float(V_flat),
        "c0": float(c0),
        "c1": float(c1),
        "c2": float(c2),
        "used_segments": int(len(S)),
    }

    return df, model_info


# =========================
# МОДЕЛ 3 – CS ЗОНИ
# =========================

def compute_cs_zones(seg: pd.DataFrame, CS_kmh: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Изчислява V_eff, относителна интензивност и време в CS зони.
    Връща:
      - сегментен DF с V_eff и ratio
      - таблица с времето по зони (all activities)
    """
    df = seg.copy()
    if df.empty or CS_kmh <= 0:
        return df, pd.DataFrame(columns=["Zone", "Time_s", "Pct_time", "Veff_mean_kmh"])

    # V_eff – основно v_final, но ограничаваме downhill влиянието
    df["V_eff_kmh"] = df["v_final_kmh"]

    # Ако е силно спускане, не позволяваме да вдига зоната над Z1_high
    mask_strong_down = df["slope_pct"] < GLIDE_SLOPE_MAX  # примерно под -5%
    V_cap = R_Z1_HIGH * CS_kmh
    df.loc[mask_strong_down & (df["V_eff_kmh"] > V_cap), "V_eff_kmh"] = V_cap

    # Относителна интензивност
    df["ratio"] = df["V_eff_kmh"] / CS_kmh

    # Зона за всеки сегмент
    def _assign_zone(r):
        for z, (lo, hi) in ZONE_BOUNDS.items():
            if (r >= lo) and (r < hi):
                return z
        return "Z6"

    df["Zone"] = df["ratio"].apply(_assign_zone)

    total_time = df["dur_s"].sum()
    rows = []
    for z in ZONE_BOUNDS.keys():
        df_z = df[df["Zone"] == z]
        if df_z.empty:
            rows.append(dict(Zone=z, Time_s=0.0, Pct_time=0.0, Veff_mean_kmh=np.nan))
            continue
        t_z = df_z["dur_s"].sum()
        pct = (t_z / total_time * 100.0) if total_time > 0 else 0.0
        Vmean = np.average(df_z["V_eff_kmh"], weights=df_z["dur_s"])
        rows.append(dict(Zone=z, Time_s=t_z, Pct_time=pct, Veff_mean_kmh=Vmean))

    zone_table = pd.DataFrame(rows)

    return df, zone_table


# =========================
# UI – STREAMLIT APP
# =========================

st.set_page_config(page_title="Ski Glide + Slope + CS Zones", layout="wide")

st.title("⛷ onFlows – Ski Glide + Slope + CS Zones")

st.markdown(
    """
Малко, леко, но функционално приложение за анализ на ски-бягане активности:

1. **Плъзгаемост (Glide)** – оценка и нормализиране спрямо референтна плъзгаемост.  
2. **Наклон (Slope)** – скоростите се пренасят към еквивалентна скорост на равно.  
3. **CS зони** – разпределение по физиологични зони според критична скорост (CS).
"""
)

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Настройки")

alpha_glide = st.sidebar.slider(
    "Омекотяване на плъзгаемостта (α_glide)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
    help="0 = игнориране на плъзгаемостта, 1 = пълно влияние на измерения индекс.",
)

CS_kmh = st.sidebar.number_input(
    "Критична скорост (CS, km/h)",
    min_value=1.0,
    max_value=30.0,
    value=10.0,
    step=0.5,
    help="Въведи CS за атлета – може от тест или друг onFlows модул.",
)

uploaded_files = st.sidebar.file_uploader(
    "Качи един или повече TCX файла",
    type=["tcx"],
    accept_multiple_files=True,
)

run_btn = st.sidebar.button("🚀 Стартирай анализа")

if not uploaded_files:
    st.info("Качи поне един TCX файл отляво, за да започнем.")
    st.stop()

if not run_btn:
    st.warning("Натисни бутона **„Стартирай анализа“** в ляво, за да изчислим моделите.")
    st.stop()

# =========================
# PIPELINE
# =========================

activities_segments = []
activity_names = []

for i, f in enumerate(uploaded_files):
    name = f.name
    activity_names.append((i, name))

    df_raw = parse_tcx(f)
    if df_raw.empty:
        st.warning(f"⚠️ {name}: няма валидни Trackpoints или TCX е празен.")
        continue

    df_prep = preprocess_df(df_raw)
    if df_prep.empty:
        st.warning(f"⚠️ {name}: след почистване не останаха валидни данни.")
        continue

    seg_df = segment_activity(df_prep, activity_id=i)
    if seg_df.empty:
        st.warning(f"⚠️ {name}: не успяхме да конструираме нито един стабилен сегмент.")
        continue

    activities_segments.append(seg_df)

if not activities_segments:
    st.error("Няма нито една активност с валидни сегменти. Провери TCX файловете.")
    st.stop()

segments_all = pd.concat(activities_segments, ignore_index=True)

# ---- Модел 1: Glide ----
segments_all, glide_info = build_glide_model(segments_all, alpha_glide=alpha_glide)

# ---- Модел 2: Наклон ----
segments_all, slope_info = build_slope_model(segments_all)

# ---- Модел 3: CS зони ----
segments_all, zone_table = compute_cs_zones(segments_all, CS_kmh=CS_kmh)

# =========================
# ИЗХОДИ – ОБЩ ПРЕГЛЕД
# =========================

st.subheader("📊 Обобщение по активности")

summary_rows = []
for aid, name in activity_names:
    segA = segments_all[segments_all["activity_id"] == aid]
    if segA.empty:
        continue
    t_total = segA["dur_s"].sum()
    V_real = np.average(segA["v_mean_kmh"], weights=segA["dur_s"])
    V_glide = np.average(segA["v_glide_kmh"], weights=segA["dur_s"])
    V_final = np.average(segA["v_final_kmh"], weights=segA["dur_s"])
    K_glide = glide_info["glide_indices"].get(aid, 1.0)

    summary_rows.append(
        dict(
            Activity=name,
            Time_min=t_total / 60.0,
            V_real_kmh=V_real,
            V_glide_kmh=V_glide,
            V_final_kmh=V_final,
            K_glide_soft=K_glide,
        )
    )

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df.style.format(
    {"Time_min": "{:.1f}", "V_real_kmh": "{:.2f}", "V_glide_kmh": "{:.2f}",
     "V_final_kmh": "{:.2f}", "K_glide_soft": "{:.3f}"}
))

# =========================
# ДЕТАЙЛИ ЗА МОДЕЛИТЕ
# =========================

with st.expander("🔍 Параметри на Glide модела (V = a·slope + b)", expanded=False):
    st.write(f"Брой използвани downhill сегменти: **{glide_info['used_segments']}**")
    st.write(f"a = **{glide_info['a']:.4f}**, b = **{glide_info['b']:.4f}**")

with st.expander("🔍 Параметри на ΔV% модела (квадратичен)", expanded=False):
    st.write(f"V_flat = **{slope_info['V_flat']:.2f} km/h**")
    st.write(
        f"ΔV% = c0 + c1·slope + c2·slope², където:  \n"
        f"c0 = **{slope_info['c0']:.4f}**, c1 = **{slope_info['c1']:.4f}**, c2 = **{slope_info['c2']:.4f}**"
    )
    st.write(f"Брой сегменти в ΔV% модела: **{slope_info['used_segments']}**")

# =========================
# CS ЗОНИ – ТАБЛИЦА
# =========================

st.subheader("🏁 Разпределение по CS зони (всички активности)")

if not zone_table.empty:
    st.dataframe(
        zone_table.style.format(
            {"Time_s": "{:.1f}", "Pct_time": "{:.1f}", "Veff_mean_kmh": "{:.2f}"}
        )
    )

# =========================
# ВИЗУАЛИЗАЦИЯ НА ЕДНА АКТИВНОСТ
# =========================

st.subheader("📈 Графики за избрана активност")

activity_labels = {aid: name for aid, name in activity_names}
selected_aid = st.selectbox(
    "Избери активност",
    options=[aid for aid, _ in activity_names],
    format_func=lambda x: activity_labels.get(x, f"Activity {x}"),
)

seg_sel = segments_all[segments_all["activity_id"] == selected_aid].copy()
if seg_sel.empty:
    st.info("За тази активност няма сегменти.")
else:
    # „време“ в минути по сегменти
    seg_sel = seg_sel.sort_values("t_start")
    seg_sel["t_min"] = seg_sel["t_start"] / 60.0

    tabs = st.tabs(["Скорост", "Наклон", "CS зони"])

    with tabs[0]:
        st.markdown("**Реална vs Glide vs Финална скорост**")
        chart_df = seg_sel[["t_min", "v_mean_kmh", "v_glide_kmh", "v_final_kmh"]].melt(
            id_vars="t_min",
            var_name="Type",
            value_name="Speed_kmh",
        )
        st.line_chart(
            chart_df,
            x="t_min",
            y="Speed_kmh",
            color="Type",
        )

    with tabs[1]:
        st.markdown("**Наклон по време**")
        st.line_chart(seg_sel.set_index("t_min")["slope_pct"])

    with tabs[2]:
        st.markdown("**CS зони по време**")
        # малка таблица за тази активност
        seg_sel_act, zone_table_act = compute_cs_zones(seg_sel, CS_kmh=CS_kmh)
        if not zone_table_act.empty:
            st.dataframe(
                zone_table_act.style.format(
                    {"Time_s": "{:.1f}", "Pct_time": "{:.1f}", "Veff_mean_kmh": "{:.2f}"}
                )
            )
        # проста хистограма ratio
        st.bar_chart(
            seg_sel_act["Zone"].value_counts().sort_index()
        )

st.success("Готово! Можеш да fine-tune-неш параметрите в sidebar-а според твоите нужди.")
