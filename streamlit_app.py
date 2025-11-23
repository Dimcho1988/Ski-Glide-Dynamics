import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime

# =========================
# НАСТРОЙКИ (можеш да ги пипаш)
# =========================
T_SEG = 5.0              # дължина на сегмента [s]
MIN_POINTS_SEG = 5       # минимум точки в сегмент
MIN_SEG_DURATION = 3.0   # минимум продължителност [s]
MIN_SEG_DISTANCE = 5.0   # минимум хоризонтална дистанция [m]
MAX_ABS_SLOPE = 30.0     # максимум |наклон| [%]

DOWNHILL_MIN = -15.0     # за Glide модел
DOWNHILL_MAX = -5.0

SLOPE_MODEL_MIN = -3.0   # за ΔV% модел
SLOPE_MODEL_MAX = 10.0

FLAT_SLOPE_ABS = 1.0     # |slope| <= 1% = "равно"

V_MAX_MS = 15.0          # макс скорост [m/s] ~ 54 km/h
H_MIN = 0.3              # минимум вертикална промяна [m]
G_MAX = 5.0              # макс вертикален градиент [m/s]

ALPHA_GLIDE_DEFAULT = 0.5  # омекотяване на плъзгаемостта
Z1_HIGH_DEFAULT = 0.80     # горна граница на Z1 (ratio от CS)

# =========================
# TCX ПАРСВАНЕ
# =========================
def parse_tcx_file(uploaded_file, activity_id):
    """Връща DataFrame с колони:
    ['activity_id','time','dist_m','elev_m','hr']
    """
    data = []

    # Прочитаме байтовете, за да може да парснем с ElementTree
    content = uploaded_file.read()
    root = ET.fromstring(content)

    # TCX namespace хак – игнорираме namespace
    ns = {"ns": root.tag.split("}")[0].strip("{")}

    trackpoints = root.findall(".//ns:Trackpoint", ns)
    for tp in trackpoints:
        # Time
        t_elem = tp.find("ns:Time", ns)
        if t_elem is None or t_elem.text is None:
            continue
        try:
            t = datetime.fromisoformat(t_elem.text.replace("Z", "+00:00"))
        except Exception:
            continue

        # Distance
        d_elem = tp.find("ns:DistanceMeters", ns)
        dist = float(d_elem.text) if d_elem is not None and d_elem.text else None

        # Altitude
        h_elem = tp.find("ns:AltitudeMeters", ns)
        elev = float(h_elem.text) if h_elem is not None and h_elem.text else None

        # Heart rate
        hr_elem = tp.find(".//ns:HeartRateBpm/ns:Value", ns)
        hr = float(hr_elem.text) if hr_elem is not None and hr_elem.text else None

        if dist is None or elev is None:
            # за модела ни трябват и двете
            continue

        data.append(
            {
                "activity_id": activity_id,
                "time": t,
                "dist_m": dist,
                "elev_m": elev,
                "hr": hr,
            }
        )

    if not data:
        return pd.DataFrame(columns=["activity_id", "time", "dist_m", "elev_m", "hr"])

    df = pd.DataFrame(data)
    df = df.sort_values(["activity_id", "time"]).reset_index(drop=True)
    return df


# =========================
# ПРЕДВАРИТЕЛНА ОБРАБОТКА
# =========================
def preprocess_points(df_points: pd.DataFrame) -> pd.DataFrame:
    """Почистване на нереални интервали + базови величини."""
    if df_points.empty:
        return df_points

    df = df_points.copy()
    df = df.sort_values(["activity_id", "time"]).reset_index(drop=True)

    # Изчисляваме Δt, Δd, Δh, v
    df["time_shift"] = df.groupby("activity_id")["time"].shift(1)
    df["dist_shift"] = df.groupby("activity_id")["dist_m"].shift(1)
    df["elev_shift"] = df.groupby("activity_id")["elev_m"].shift(1)

    df["dt"] = (df["time"] - df["time_shift"]).dt.total_seconds()
    df["dd"] = df["dist_m"] - df["dist_shift"]
    df["dh"] = df["elev_m"] - df["elev_shift"]

    # Премахваме първата точка от всяка активност (няма предходна)
    df = df[~df["time_shift"].isna()].copy()

    # Премахване на невалидни интервали
    # 1) dt <= 0
    mask_valid = df["dt"] > 0

    # 2) много големи скокове във времето (например > 30 s)
    mask_valid &= df["dt"] <= 30.0

    # 3) скорост > V_MAX_MS
    v_ms = df["dd"] / df["dt"]
    mask_valid &= v_ms.between(0, V_MAX_MS)

    # 4) вертикален шум: или твърде малка |dh| или твърде голям вертикален градиент
    grad_vert = df["dh"].abs() / df["dt"]
    mask_valid &= ~((df["dh"].abs() < H_MIN) | (grad_vert > G_MAX))

    df = df[mask_valid].copy()

    # Окончателни скорости
    df["v_ms"] = df["dd"] / df["dt"]
    df["v_kmh"] = df["v_ms"] * 3.6

    # Нормализирано време от начало на активността
    df["t0"] = df.groupby("activity_id")["time"].transform("min")
    df["elapsed_s"] = (df["time"] - df["t0"]).dt.total_seconds()

    return df[["activity_id", "time", "elapsed_s", "dist_m", "elev_m", "hr", "dt", "dd", "dh", "v_ms", "v_kmh"]]


# =========================
# СЕГМЕНТИРАНЕ
# =========================
def build_segments(df_points: pd.DataFrame) -> pd.DataFrame:
    """Конструира 5s сегменти и връща DataFrame със сегментни величини."""
    if df_points.empty:
        return pd.DataFrame()

    df = df_points.copy()
    df["seg_id"] = np.floor(df["elapsed_s"] / T_SEG).astype(int)

    agg = df.groupby(["activity_id", "seg_id"]).agg(
        t_start=("time", "min"),
        t_end=("time", "max"),
        dist_start=("dist_m", "min"),
        dist_end=("dist_m", "max"),
        elev_start=("elev_m", "min"),
        elev_end=("elev_m", "max"),
        hr_mean=("hr", "mean"),
        n_points=("time", "size"),
    ).reset_index()

    agg["dt_seg"] = (agg["t_end"] - agg["t_start"]).dt.total_seconds()
    agg["D_seg"] = agg["dist_end"] - agg["dist_start"]
    agg["dh_seg"] = agg["elev_end"] - agg["elev_start"]
    agg["V_seg_kmh"] = (agg["D_seg"] / agg["dt_seg"]) * 3.6
    agg["slope_pct"] = (agg["dh_seg"] / agg["D_seg"]) * 100.0

    # Условия за валиден сегмент
    valid = (
        (agg["n_points"] >= MIN_POINTS_SEG)
        & (agg["dt_seg"] >= MIN_SEG_DURATION)
        & (agg["D_seg"] >= MIN_SEG_DISTANCE)
        & (agg["slope_pct"].abs() <= MAX_ABS_SLOPE)
    )

    segs = agg[valid].copy()
    return segs.reset_index(drop=True)


# =========================
# МОДЕЛ 1 – ПЛЪЗГАЕМОСТ
# =========================
def fit_glide_model(segments: pd.DataFrame, alpha_glide: float):
    """Връща:
    - segments с V_glide
    - summary_df по активност
    """
    segs = segments.copy()
    if segs.empty:
        return segs.assign(V_glide=np.nan), pd.DataFrame()

    # downhill маска + условие предходен сегмент
    segs = segs.sort_values(["activity_id", "t_start"]).reset_index(drop=True)
    downhill_mask = (segs["slope_pct"] >= DOWNHILL_MIN) & (segs["slope_pct"] <= DOWNHILL_MAX)

    segs["downhill"] = downhill_mask
    segs["downhill_prev"] = segs.groupby("activity_id")["downhill"].shift(1).fillna(False)
    segs["downhill_use"] = segs["downhill"] & segs["downhill_prev"]

    D = segs[segs["downhill_use"]].copy()
    if D.empty or len(D) < 20:
        # няма достатъчно данни за модел
        segs["K_glide_soft"] = 1.0
        segs["V_glide"] = segs["V_seg_kmh"]
        summary = compute_glide_summary_no_model(segs)
        return segs, summary

    # outlier филтър R = V / |slope|
    D["R"] = D["V_seg_kmh"] / D["slope_pct"].abs()
    r_low, r_high = D["R"].quantile([0.05, 0.95])
    D_star = D[(D["R"] >= r_low) & (D["R"] <= r_high)].copy()

    if D_star.empty or len(D_star) < 20:
        segs["K_glide_soft"] = 1.0
        segs["V_glide"] = segs["V_seg_kmh"]
        summary = compute_glide_summary_no_model(segs)
        return segs, summary

    # КВАДРАТИЧЕН модел V = a2*slope^2 + a1*slope + a0
    x = D_star["slope_pct"].values
    y = D_star["V_seg_kmh"].values
    try:
        coeffs = np.polyfit(x, y, deg=2)
    except Exception:
        segs["K_glide_soft"] = 1.0
        segs["V_glide"] = segs["V_seg_kmh"]
        summary = compute_glide_summary_no_model(segs)
        return segs, summary

    a2, a1, a0 = coeffs
    segs["V_glide"] = segs["V_seg_kmh"]  # ще коригираме по-долу

    # Изчисляваме K_glide за всяка активност
    glide_info = []
    for act_id, dfA in segs.groupby("activity_id"):
        df_downA = dfA[dfA["downhill_use"]].copy()
        if df_downA.empty:
            K_raw = 1.0
            K_soft = 1.0
            mean_slope = np.nan
            mean_V_real = np.nan
            V_model = np.nan
            n_down = 0
        else:
            w = df_downA["dt_seg"]
            mean_slope = np.average(df_downA["slope_pct"], weights=w)
            mean_V_real = np.average(df_downA["V_seg_kmh"], weights=w)
            V_model = a2 * mean_slope ** 2 + a1 * mean_slope + a0
            if V_model <= 0 or np.isnan(V_model):
                K_raw = 1.0
            else:
                K_raw = mean_V_real / V_model
            # safeguard
            if K_raw <= 0 or np.isnan(K_raw) or K_raw > 2.5:
                K_raw = 1.0
            K_soft = 1.0 + alpha_glide * (K_raw - 1.0)
            n_down = len(df_downA)

        glide_info.append(
            dict(
                activity_id=act_id,
                n_downhill=n_down,
                mean_down_slope_pct=mean_slope,
                mean_down_V_real=mean_V_real,
                V_down_model=V_model,
                K_glide_raw=K_raw,
                K_glide_soft=K_soft,
            )
        )

    glide_df = pd.DataFrame(glide_info)

    # прикачаме K_glide_soft към сегментите
    segs = segs.merge(glide_df[["activity_id", "K_glide_soft"]], on="activity_id", how="left")
    segs["K_glide_soft"] = segs["K_glide_soft"].fillna(1.0)
    segs["V_glide"] = segs["V_seg_kmh"] / segs["K_glide_soft"]

    # средни скорости по активност
    summary_rows = []
    for act_id, dfA in segs.groupby("activity_id"):
        w = dfA["dt_seg"]
        V_real_overall = np.average(dfA["V_seg_kmh"], weights=w)
        V_glide_overall = np.average(dfA["V_glide"], weights=w)
        info = glide_df[glide_df["activity_id"] == act_id].iloc[0].to_dict()
        info.update(
            dict(
                V_overall_real=V_real_overall,
                V_overall_glide=V_glide_overall,
            )
        )
        summary_rows.append(info)

    summary_df = pd.DataFrame(summary_rows)

    return segs, summary_df


def compute_glide_summary_no_model(segs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for act_id, dfA in segs.groupby("activity_id"):
        w = dfA["dt_seg"]
        V_real_overall = np.average(dfA["V_seg_kmh"], weights=w)
        V_glide_overall = np.average(dfA["V_seg_kmh"], weights=w)
        rows.append(
            dict(
                activity_id=act_id,
                n_downhill=0,
                mean_down_slope_pct=np.nan,
                mean_down_V_real=np.nan,
                V_down_model=np.nan,
                K_glide_raw=1.0,
                K_glide_soft=1.0,
                V_overall_real=V_real_overall,
                V_overall_glide=V_glide_overall,
            )
        )
    return pd.DataFrame(rows)


# =========================
# МОДЕЛ 2 – НАКЛОН (ΔV%)
# =========================
def fit_slope_model(segs_glide: pd.DataFrame):
    """Работи върху V_glide; връща:
    - segments с V_final
    - summary_df по активност
    """
    segs = segs_glide.copy()
    if segs.empty:
        segs["V_final"] = np.nan
        return segs, pd.DataFrame()

    # 1) V_flat за ВСЯКА активност поотделно (|slope| <= 1%)
    flat_info = []
    V_flat_dict = {}
    for act_id, dfA in segs.groupby("activity_id"):
        flatA = dfA[dfA["slope_pct"].abs() <= FLAT_SLOPE_ABS].copy()
        if flatA.empty:
            V_flat = np.nan
        else:
            w = flatA["dt_seg"]
            V_flat = np.average(flatA["V_glide"], weights=w)
        V_flat_dict[act_id] = V_flat
        flat_info.append(dict(activity_id=act_id, V_flat=V_flat))
    flat_df = pd.DataFrame(flat_info)

    segs = segs.merge(flat_df, on="activity_id", how="left")

    # 2) Сегменти за обучение на ΔV% модел
    # -3% < slope < 10%, не включваме равните (|slope| <= 1%) за да избегнем шум
    mask_model = (
        (segs["slope_pct"] > SLOPE_MODEL_MIN)
        & (segs["slope_pct"] < SLOPE_MODEL_MAX)
        & (segs["slope_pct"].abs() > FLAT_SLOPE_ABS)
        & (~segs["V_flat"].isna())
        & (segs["V_flat"] > 0)
    )

    M = segs[mask_model].copy()
    if M.empty or len(M) < 30:
        # няма стабилен модел – няма да коригираме по наклон
        segs["V_final"] = segs["V_glide"]
        summary = slope_summary_no_model(segs)
        return segs, summary

    # ΔV_real_s = 100 * (V_glide_s - V_flat_A) / V_flat_A
    M["DeltaV_real_pct"] = 100.0 * (M["V_glide"] - M["V_flat"]) / M["V_flat"]

    # КВАДРАТИЧЕН ΔV модел: ΔV% = c2*slope^2 + c1*slope + c0
    x = M["slope_pct"].values
    y = M["DeltaV_real_pct"].values
    try:
        coeffs = np.polyfit(x, y, deg=2)
    except Exception:
        segs["V_final"] = segs["V_glide"]
        summary = slope_summary_no_model(segs)
        return segs, summary

    c2, c1, c0 = coeffs

    # 3) Финален коригиращ фактор f_slope
    def f_slope(slope):
        if np.isnan(slope):
            return 1.0
        if abs(slope) <= FLAT_SLOPE_ABS:
            return 1.0
        if SLOPE_MODEL_MIN < slope < SLOPE_MODEL_MAX:
            DeltaV_model = c2 * slope ** 2 + c1 * slope + c0  # в %
            return 1.0 + (DeltaV_model / 100.0)
        return 1.0

    segs["f_slope"] = segs["slope_pct"].apply(f_slope)
    segs["V_final"] = segs["V_glide"] / segs["f_slope"].replace(0, np.nan)

    # summary по активност
    summary_rows = []
    for act_id, dfA in segs.groupby("activity_id"):
        w = dfA["dt_seg"]
        V_real_overall = np.average(dfA["V_seg_kmh"], weights=w)
        V_glide_overall = np.average(dfA["V_glide"], weights=w)
        V_final_overall = np.average(dfA["V_final"], weights=w)
        # колко сегмента влязоха в ΔV модела за тази активност
        n_model = mask_model[dfA.index].sum()
        mean_slope_model = dfA.loc[mask_model[dfA.index], "slope_pct"].mean() if n_model > 0 else np.nan
        V_flat_A = dfA["V_flat"].iloc[0]
        summary_rows.append(
            dict(
                activity_id=act_id,
                V_flat=V_flat_A,
                n_slope_segments=int(n_model),
                mean_slope_model=mean_slope_model,
                V_overall_real=V_real_overall,
                V_overall_glide=V_glide_overall,
                V_overall_final=V_final_overall,
                c0=c0,
                c1=c1,
                c2=c2,
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    return segs, summary_df


def slope_summary_no_model(segs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for act_id, dfA in segs.groupby("activity_id"):
        w = dfA["dt_seg"]
        V_real_overall = np.average(dfA["V_seg_kmh"], weights=w)
        V_glide_overall = np.average(dfA["V_glide"], weights=w)
        V_final_overall = V_glide_overall
        rows.append(
            dict(
                activity_id=act_id,
                V_flat=np.nan,
                n_slope_segments=0,
                mean_slope_model=np.nan,
                V_overall_real=V_real_overall,
                V_overall_glide=V_glide_overall,
                V_overall_final=V_final_overall,
                c0=0.0,
                c1=0.0,
                c2=0.0,
            )
        )
    return pd.DataFrame(rows)


# =========================
# МОДЕЛ 3 – CS ЗОНИ + ПУЛС
# =========================
def compute_cs_zones(segs_final: pd.DataFrame, CS_kmh: float, z1_high: float):
    """Връща:
    - таблица по зони за всички активности общо
    - таблица по зони за всяка активност
    """
    segs = segs_final.copy()
    if segs.empty or CS_kmh <= 0:
        return pd.DataFrame(), pd.DataFrame()

    # V_eff: при slope <= -3% режем до Z1 high
    def v_eff(row):
        if row["slope_pct"] <= -3.0:
            return CS_kmh * z1_high
        return row["V_final"]

    segs["V_eff"] = segs.apply(v_eff, axis=1)
    segs["ratio"] = segs["V_eff"] / CS_kmh

    # дефиниция на зоните
    def zone_label(r):
        if r < 0.80:
            return "Z1 (<=80%)"
        elif r < 0.90:
            return "Z2 (80–90%)"
        elif r < 1.00:
            return "Z3 (90–100%)"
        elif r < 1.10:
            return "Z4 (100–110%)"
        elif r < 1.20:
            return "Z5 (110–120%)"
        else:
            return "Z6 (>120%)"

    segs["zone"] = segs["ratio"].apply(zone_label)

    # HR средно за сегмента – вече имаме hr_mean
    # (в build_segments сме го сметнали)

    # --- общо за всички активности
    total_time = segs["dt_seg"].sum()
    zone_group = segs.groupby("zone").agg(
        T_s=("dt_seg", "sum"),
        V_eff_mean=("V_eff", lambda x: np.average(x, weights=segs.loc[x.index, "dt_seg"])),
        HR_mean=("hr_mean", lambda x: np.average(x.dropna(), weights=segs.loc[x.index, "dt_seg"].loc[~x.isna()]) if (~x.isna()).any() else np.nan),
    ).reset_index()

    zone_group["T_min"] = zone_group["T_s"] / 60.0
    zone_group["P_time_%"] = 100.0 * zone_group["T_s"] / total_time

    zone_total_table = zone_group[["zone", "T_s", "T_min", "P_time_%", "V_eff_mean", "HR_mean"]].sort_values("zone")

    # --- по отделни активности
    rows = []
    for (act_id, zone), dfZ in segs.groupby(["activity_id", "zone"]):
        T_s = dfZ["dt_seg"].sum()
        V_eff_mean = np.average(dfZ["V_eff"], weights=dfZ["dt_seg"])
        # HR
        if dfZ["hr_mean"].notna().any():
            HR_mean = np.average(dfZ["hr_mean"].dropna(), weights=dfZ["dt_seg"].loc[dfZ["hr_mean"].notna()])
        else:
            HR_mean = np.nan
        rows.append(dict(activity_id=act_id, zone=zone, T_s=T_s, V_eff_mean=V_eff_mean, HR_mean=HR_mean))

    zone_act_df = pd.DataFrame(rows)
    if not zone_act_df.empty:
        total_by_act = zone_act_df.groupby("activity_id")["T_s"].transform("sum")
        zone_act_df["T_min"] = zone_act_df["T_s"] / 60.0
        zone_act_df["P_time_%"] = 100.0 * zone_act_df["T_s"] / total_by_act

    zone_act_df = zone_act_df.sort_values(["activity_id", "zone"])

    return zone_total_table, zone_act_df


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="onFlows – Ski Glide + Slope + CS Zones", layout="wide")

st.title("onFlows – Ski Glide + Slope + CS Zones")
st.write("Комбиниран модел: плъзгаемост → наклон → CS зони + пулс.")

# --- Sidebar ---
st.sidebar.header("Входни данни и параметри")

uploaded_files = st.sidebar.file_uploader(
    "Качи един или повече TCX файла",
    type=["tcx"],
    accept_multiple_files=True,
)

alpha_glide = st.sidebar.slider(
    "Омекотяване на плъзгаемостта (α_glide)",
    min_value=0.0,
    max_value=1.0,
    value=ALPHA_GLIDE_DEFAULT,
    step=0.05,
)

CS_kmh = st.sidebar.number_input(
    "Критична скорост (CS, km/h)",
    min_value=0.0,
    max_value=50.0,
    value=18.0,
    step=0.5,
)

z1_high = st.sidebar.slider(
    "Горна граница на Z1 (ratio от CS)",
    min_value=0.6,
    max_value=0.95,
    value=Z1_HIGH_DEFAULT,
    step=0.01,
)

st.sidebar.markdown("---")
st.sidebar.write("Сегменти:", f"{T_SEG:.0f} s, валидни сегменти: ≥{MIN_POINTS_SEG} точки, ≥{MIN_SEG_DISTANCE} m, ≥{MIN_SEG_DURATION} s, |slope| ≤ {MAX_ABS_SLOPE}%.")

if not uploaded_files:
    st.info("Качи поне един TCX файл от менюто вляво.")
    st.stop()

# =========================
# ОБРАБОТКА НА ФАЙЛОВЕТЕ
# =========================
all_points = []
for i, f in enumerate(uploaded_files):
    act_id = f"{i+1}: {f.name}"
    df_tcx = parse_tcx_file(f, act_id)
    if df_tcx.empty:
        st.warning(f"⚠ Файлът {f.name} не съдържа валидни Trackpoint данни.")
        continue
    all_points.append(df_tcx)

if not all_points:
    st.error("Няма валидни данни от нито един TCX.")
    st.stop()

df_points_raw = pd.concat(all_points, ignore_index=True)
df_points = preprocess_points(df_points_raw)
if df_points.empty:
    st.error("След филтрирането не останаха валидни точки.")
    st.stop()

segments = build_segments(df_points)
if segments.empty:
    st.error("Не успях да създам валидни сегменти. Може би трябва да намалим изискванията.")
    st.stop()

st.success(f"Успешно заредени {len(uploaded_files)} активности, {len(segments)} валидни сегмента.")

# =========================
# МОДЕЛ 1 – ПЛЪЗГАЕМОСТ
# =========================
st.header("1. Модел за плъзгаемост (Glide)")

segments_glide, glide_summary = fit_glide_model(segments, alpha_glide=alpha_glide)

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Таблица по активност (Модел 1)")
    if glide_summary.empty:
        st.write("Няма достатъчно downhill сегменти за оценка на модела. Корекция по плъзгаемост не се прилага.")
    else:
        display_cols = [
            "activity_id",
            "n_downhill",
            "mean_down_slope_pct",
            "mean_down_V_real",
            "V_down_model",
            "K_glide_raw",
            "K_glide_soft",
            "V_overall_real",
            "V_overall_glide",
        ]
        st.dataframe(glide_summary[display_cols].round(3))

with col2:
    st.subheader("V срещу наклон (downhill сегменти)")
    down = segments_glide[segments_glide["downhill_use"]].copy()
    if down.empty:
        st.write("Няма downhill сегменти за визуализация.")
    else:
        import altair as alt

        chart = (
            alt.Chart(down)
            .mark_circle(size=30, opacity=0.5)
            .encode(
                x=alt.X("slope_pct", title="Наклон [%]"),
                y=alt.Y("V_seg_kmh", title="Сегментна скорост [km/h]"),
                color="activity_id",
                tooltip=["activity_id", "slope_pct", "V_seg_kmh"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

st.markdown("---")

# =========================
# МОДЕЛ 2 – НАКЛОН
# =========================
st.header("2. Модел за влияние на наклона")

segments_final, slope_summary = fit_slope_model(segments_glide)

col3, col4 = st.columns([2, 3])

with col3:
    st.subheader("Таблица по активност (Модел 2)")
    if slope_summary.empty:
        st.write("Няма достатъчно сегменти за ΔV% модел. Корекция по наклон не се прилага.")
    else:
        display_cols = [
            "activity_id",
            "V_flat",
            "n_slope_segments",
            "mean_slope_model",
            "V_overall_real",
            "V_overall_glide",
            "V_overall_final",
        ]
        st.dataframe(slope_summary[display_cols].round(3))

with col4:
    st.subheader("ΔV% срещу наклон (обучаващ облак)")
    M_mask = (
        (segments_final["slope_pct"] > SLOPE_MODEL_MIN)
        & (segments_final["slope_pct"] < SLOPE_MODEL_MAX)
        & (segments_final["slope_pct"].abs() > FLAT_SLOPE_ABS)
        & (~segments_final["V_flat"].isna())
        & (segments_final["V_flat"] > 0)
    )
    M_plot = segments_final[M_mask].copy()
    if M_plot.empty:
        st.write("Няма достатъчно сегменти за визуализация на ΔV%.")
    else:
        M_plot["DeltaV_real_pct"] = 100.0 * (M_plot["V_glide"] - M_plot["V_flat"]) / M_plot["V_flat"]
        import altair as alt

        chart2 = (
            alt.Chart(M_plot)
            .mark_circle(size=30, opacity=0.5)
            .encode(
                x=alt.X("slope_pct", title="Наклон [%]"),
                y=alt.Y("DeltaV_real_pct", title="ΔV_real [% спрямо V_flat на активността]"),
                color="activity_id",
                tooltip=["activity_id", "slope_pct", "DeltaV_real_pct"],
            )
        )
        st.altair_chart(chart2, use_container_width=True)

st.markdown("---")

# =========================
# МОДЕЛ 3 – CS ЗОНИ + ПУЛС
# =========================
st.header("3. Разпределение по CS зони + пулс (Модел 3)")

zone_total_table, zone_act_df = compute_cs_zones(segments_final, CS_kmh=CS_kmh, z1_high=z1_high)

if zone_total_table.empty:
    st.write("Не може да се изчисли разпределение по зони. Проверете дали CS > 0 и има сегменти.")
else:
    col5, col6 = st.columns([2, 3])

    with col5:
        st.subheader("Всички активности – общо")
        st.dataframe(zone_total_table.round(3))

    with col6:
        st.subheader("Избор на конкретна активност")
        act_ids = sorted(segments_final["activity_id"].unique())
        selected_act = st.selectbox("Избери активност за детайлно зониране:", act_ids)
        df_act = zone_act_df[zone_act_df["activity_id"] == selected_act].copy()
        if df_act.empty:
            st.write("За тази активност няма валидни сегменти за зони.")
        else:
            st.dataframe(
                df_act[["activity_id", "zone", "T_s", "T_min", "P_time_%", "V_eff_mean", "HR_mean"]].round(3)
            )

    # малка бар-диаграма за общото разпределение
    import altair as alt

    st.subheader("Общо разпределение на времето по зони")
    chart3 = (
        alt.Chart(zone_total_table)
        .mark_bar()
        .encode(
            x=alt.X("zone", sort=None, title="Зона"),
            y=alt.Y("P_time_%", title="% време"),
            tooltip=["zone", "P_time_%", "T_min", "V_eff_mean", "HR_mean"],
        )
    )
    st.altair_chart(chart3, use_container_width=True)

st.markdown("---")
st.write(
    """
**Обобщение на изходите:**

- Модел 1 (плъзгаемост): реална скорост, брой downhill сегменти, средна downhill скорост и наклон, 
  модифицирана скорост по плъзгаемост (V_overall_glide).
- Модел 2 (наклон): влияние на наклона, реална средна скорост, модифицирана по плъзгаемост и финална 
  скорост (V_overall_final) приравнена към равен терен.
- Модел 3 (CS зони): време в зона, % време, средна модулирана скорост и съответстващ пулс по зони –
  както общо за всички активности, така и за конкретна активност.
"""
)
