import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import altair as alt

# -----------------------------
# Настройки по подразбиране
# -----------------------------
T_SEG = 5.0              # дължина на сегмента [s]
MIN_POINTS_SEG = 2       # минимум точки в сегмента
MIN_D_SEG = 5.0          # минимум хоризонтална дистанция [m]
MIN_T_SEG = 3.0          # минимум продължителност [s]
MAX_ABS_SLOPE = 30.0     # макс. наклон [%]
V_MAX_KMH = 60.0         # оставено за диагностика, не филтрираме по него

DOWN_MIN = -15.0         # долна граница за downhill [%] за Glide модела
DOWN_MAX = -5.0          # горна граница за downhill [%]

SLOPE_MODEL_MIN = -3.0   # диапазон за ΔV% модела
SLOPE_MODEL_MAX = 10.0

FLAT_BAND = 1.0          # |slope| <= 1% се счита за "равно"

DOWNHILL_ZONE1_THRESH = -5.0  # всичко под -5% -> Z1, "без усилие"


# --------------------------------------------------------------------------------
# Помощни функции
# --------------------------------------------------------------------------------
def parse_tcx(file: BytesIO, activity_id: str) -> pd.DataFrame:
    """
    Парсира TCX файл и връща DataFrame с колони:
    ['activity_id', 'time', 'dist', 'alt', 'hr']
    """
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {}
    if root.tag[0] == "{":
        uri = root.tag[1:].split("}")[0]
        ns["tcx"] = uri
    else:
        ns["tcx"] = ""

    trackpoints = root.findall(".//tcx:Trackpoint", ns)

    rows = []
    for tp in trackpoints:
        time_el = tp.find("tcx:Time", ns)
        dist_el = tp.find("tcx:DistanceMeters", ns)
        alt_el = tp.find("tcx:AltitudeMeters", ns)
        hr_el = tp.find("tcx:HeartRateBpm/tcx:Value", ns)

        if time_el is None or dist_el is None or alt_el is None:
            continue

        t_str = time_el.text
        try:
            t = pd.to_datetime(t_str)
        except Exception:
            try:
                t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                try:
                    t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    continue

        try:
            dist = float(dist_el.text)
            alt = float(alt_el.text)
        except (TypeError, ValueError):
            continue

        if hr_el is not None and hr_el.text is not None:
            try:
                hr = float(hr_el.text)
            except (TypeError, ValueError):
                hr = np.nan
        else:
            hr = np.nan

        rows.append(
            {
                "activity_id": activity_id,
                "time": t,
                "dist": dist,
                "alt": alt,
                "hr": hr,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["activity_id", "time", "dist", "alt", "hr"])

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def preprocess_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    НЯМА филтриране – само:
    - сортиране по време
    - time_sec за сегментация
    - alt_smooth = alt (работим със суровата височина)
    """
    if df.empty:
        return df

    df = df.sort_values("time").reset_index(drop=True)
    df["alt_smooth"] = df["alt"]          # сурови данни
    df["time_sec"] = df["time"].astype("int64") / 1e9
    return df


def build_segments(df: pd.DataFrame, activity_id: str, t_seg: float = T_SEG) -> pd.DataFrame:
    """
    Прави 5-секундни сегменти без припокриване върху СУРОВИТЕ данни.
    Взима само първа и последна точка в сегмента за:
    - Δh, Δd, slope, V
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "activity_id",
                "seg_id",
                "t_start",
                "t_end",
                "duration_s",
                "D_m",
                "dh_m",
                "slope_pct",
                "V_kmh",
                "hr_mean",
            ]
        )

    t0 = df["time_sec"].iloc[0]
    df["t_rel"] = df["time_sec"] - t0
    df["seg_id"] = (df["t_rel"] / t_seg).astype(int)

    seg_rows = []
    for seg_id, g in df.groupby("seg_id"):
        g = g.sort_values("time")
        if len(g) < MIN_POINTS_SEG:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration = (t_end - t_start).total_seconds()
        if duration < MIN_T_SEG:
            continue

        dist_start = g["dist"].iloc[0]
        dist_end = g["dist"].iloc[-1]
        D = dist_end - dist_start
        if D < MIN_D_SEG:
            continue

        alt_start = g["alt_smooth"].iloc[0]
        alt_end = g["alt_smooth"].iloc[-1]
        dh = alt_end - alt_start
        slope = (dh / D) * 100.0 if D > 0 else np.nan
        if np.isnan(slope) or abs(slope) > MAX_ABS_SLOPE:
            continue

        V_kmh = (D / duration) * 3.6
        hr_mean = g["hr"].mean()

        seg_rows.append(
            {
                "activity_id": activity_id,
                "seg_id": int(seg_id),
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration,
                "D_m": D,
                "dh_m": dh,
                "slope_pct": slope,
                "V_kmh": V_kmh,
                "hr_mean": hr_mean,
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    return seg_df


# --------------------------------------------------------------------------------
# Модел 1 – Плъзгаемост (Glide)
# --------------------------------------------------------------------------------
def compute_glide_model(segments: pd.DataFrame, alpha_glide: float, deg_glide: int = 2):
    """
    Връща:
    - segments с добавени колони ['is_downhill','V_glide','K_glide_raw','K_glide_soft']
    - таблица с обобщения по активност
    - параметри на глобалния Glide модел (np.poly1d glide_poly или None)
    """
    seg = segments.copy()
    if seg.empty:
        return seg, pd.DataFrame(), None

    seg = seg.sort_values(["activity_id", "seg_id"]).reset_index(drop=True)

    # downhill сегмент = наклон между DOWN_MIN и DOWN_MAX
    seg["is_downhill"] = (seg["slope_pct"] >= DOWN_MIN) & (seg["slope_pct"] <= DOWN_MAX)

    # изискваме и предходният сегмент да е бил downhill (по-стабилни условия)
    seg["is_prev_downhill"] = False
    for aid, g in seg.groupby("activity_id"):
        idx = g.index
        prev_down = g["is_downhill"].shift(1).fillna(False).values
        seg.loc[idx, "is_prev_downhill"] = prev_down

    downhill_mask = seg["is_downhill"] & seg["is_prev_downhill"]
    down_df = seg.loc[downhill_mask].copy()

    # ако няма достатъчно downhill сегменти – без модел
    if len(down_df) < 20:
        seg["K_glide_raw"] = 1.0
        seg["K_glide_soft"] = 1.0
        seg["V_glide"] = seg["V_kmh"]
        summary = (
            seg.groupby("activity_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "activity_id": g["activity_id"].iloc[0],
                        "n_downhill": 0,
                        "mean_down_slope": np.nan,
                        "mean_down_V_real": np.nan,
                        "V_down_model": np.nan,
                        "K_glide_raw": 1.0,
                        "K_glide_soft": 1.0,
                        "V_overall_real": (g["V_kmh"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                        "V_overall_glide": (g["V_kmh"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                    }
                )
            )
            .reset_index(drop=True)
        )
        return seg, summary, None

    # Глобален Glide модел V = f(slope)
    x = down_df["slope_pct"].values
    y = down_df["V_kmh"].values
    try:
        coeffs = np.polyfit(x, y, deg=deg_glide)
        glide_poly = np.poly1d(coeffs)
    except Exception:
        glide_poly = None

    activity_rows = []
    seg["K_glide_raw"] = 1.0
    seg["K_glide_soft"] = 1.0

    for aid, g in seg.groupby("activity_id"):
        g_down = down_df[down_df["activity_id"] == aid]
        if glide_poly is None or len(g_down) < 5:
            K_raw = 1.0
            K_soft = 1.0
            mean_down_slope = np.nan
            mean_down_V_real = np.nan
            V_down_model = np.nan
            n_down = len(g_down)
        else:
            w = g_down["duration_s"].values
            mean_down_slope = np.average(g_down["slope_pct"].values, weights=w)
            mean_down_V_real = np.average(g_down["V_kmh"].values, weights=w)
            V_down_model = float(glide_poly(mean_down_slope))
            if V_down_model <= 0:
                K_raw = 1.0
            else:
                K_raw = mean_down_V_real / V_down_model
            # безопасни граници за K_raw
            if not np.isfinite(K_raw) or K_raw <= 0.3 or K_raw >= 1.7:
                K_raw = 1.0
            K_soft = 1.0 + alpha_glide * (K_raw - 1.0)
            n_down = len(g_down)

        seg.loc[seg["activity_id"] == aid, "K_glide_raw"] = K_raw
        seg.loc[seg["activity_id"] == aid, "K_glide_soft"] = K_soft

        V_overall_real_seg = (g["V_kmh"] * g["duration_s"]).sum() / g["duration_s"].sum()
        V_overall_glide_seg = (
            (g["V_kmh"] / K_soft) * g["duration_s"]
        ).sum() / g["duration_s"].sum()

        activity_rows.append(
            {
                "activity_id": aid,
                "n_downhill": n_down,
                "mean_down_slope": mean_down_slope,
                "mean_down_V_real": mean_down_V_real,
                "V_down_model": V_down_model,
                "K_glide_raw": K_raw,
                "K_glide_soft": K_soft,
                "V_overall_real": V_overall_real_seg,
                "V_overall_glide": V_overall_glide_seg,
            }
        )

    seg["V_glide"] = seg["V_kmh"] / seg["K_glide_soft"]
    summary_df = pd.DataFrame(activity_rows)

    return seg, summary_df, glide_poly


# --------------------------------------------------------------------------------
# Модел 2 – Наклон (ГЛОБАЛЕН ΔV%(slope), ПО СЕГМЕНТИ)
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Модел 2 – Наклон (ГЛОБАЛЕН ΔV%(slope), ПО СЕГМЕНТИ)
# --------------------------------------------------------------------------------
def compute_slope_model(
    segments_glide: pd.DataFrame,
    alpha_slope: float = 0.5,
    dv_clip: float = 40.0,
):
    """
    Глобален ΔV%(slope) модел, прилаган ПО СЕГМЕНТИ:

    1) За всяка активност A:
       - намираме V_flat,A от сегментите с |slope| <= FLAT_BAND;
       - за всички нейни сегменти смятаме ΔV_real,S = 100 * (V_glide,S - V_flat,A) / V_flat,A.

    2) Всички (s_S, ΔV_real,S) от всички активности влизат в общ обучаващ набор
       за глобален квадратичен модел ΔV_model_raw(s).

    3) За всеки сегмент S:
       - взимаме ΔV_model_raw(s_S) от полинома,
       - омекотяваме: ΔV_model = alpha_slope * ΔV_model_raw,
       - клипваме в [-dv_clip, +dv_clip],
       - ако сегментът е извън диапазона на модела (s <= SLOPE_MODEL_MIN,
         s >= SLOPE_MODEL_MAX или |s| <= FLAT_BAND) → ΔV_model = 0.
       Получаваме:
         coef_slope(s) = 1 - ΔV_model/100
         V_final = V_glide * coef_slope(s)
    """
    seg = segments_glide.copy()
    if seg.empty:
        return seg, pd.DataFrame(), None

    # 1) V_flat по активност (от V_glide)
    vflat_map = {}
    for aid, g in seg.groupby("activity_id"):
        flat = g[abs(g["slope_pct"]) <= FLAT_BAND]
        if len(flat) < 5:
            flat = g
        V_flat = (flat["V_glide"] * flat["duration_s"]).sum() / flat["duration_s"].sum()
        vflat_map[aid] = V_flat

    seg["V_flat_A"] = seg["activity_id"].map(vflat_map)
    seg["DeltaV_real"] = 100.0 * (seg["V_glide"] - seg["V_flat_A"]) / seg["V_flat_A"]

    # 2) Обучаващ набор за ΔV%(slope)
    train = seg[
        (seg["slope_pct"] > SLOPE_MODEL_MIN)
        & (seg["slope_pct"] < SLOPE_MODEL_MAX)
        & seg["V_flat_A"].notna()
    ].copy()

    if len(train) < 20:
        # fallback – няма достатъчно данни за наклонов модел
        seg["DeltaV_model_raw"] = 0.0
        seg["DeltaV_model"] = 0.0
        seg["coef_slope"] = 1.0
        seg["V_final"] = seg["V_glide"]

        summary = (
            seg.groupby("activity_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "activity_id": g["activity_id"].iloc[0],
                        "n_slope_segments": 0,
                        "mean_slope_model": np.nan,
                        "mean_DeltaV_real": np.nan,
                        "V_overall_real": (g["V_kmh"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                        "V_overall_glide": (g["V_glide"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                        "V_overall_final": (g["V_glide"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                    }
                )
            )
            .reset_index(drop=True)
        )
        return seg, summary, None

    x = train["slope_pct"].values
    y = train["DeltaV_real"].values
    try:
        coeffs = np.polyfit(x, y, deg=2)
        slope_poly = np.poly1d(coeffs)
    except Exception:
        slope_poly = None

    # Ако няма стабилен полином
    if slope_poly is None:
        seg["DeltaV_model_raw"] = 0.0
        seg["DeltaV_model"] = 0.0
        seg["coef_slope"] = 1.0
        seg["V_final"] = seg["V_glide"]

        summary = (
            seg.groupby("activity_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "activity_id": g["activity_id"].iloc[0],
                        "n_slope_segments": len(
                            g[
                                (g["slope_pct"] > SLOPE_MODEL_MIN)
                                & (g["slope_pct"] < SLOPE_MODEL_MAX)
                            ]
                        ),
                        "mean_slope_model": np.nan,
                        "mean_DeltaV_real": np.nan,
                        "V_overall_real": (g["V_kmh"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                        "V_overall_glide": (g["V_glide"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                        "V_overall_final": (g["V_glide"] * g["duration_s"]).sum()
                        / g["duration_s"].sum(),
                    }
                )
            )
            .reset_index(drop=True)
        )
        return seg, summary, None

    # 3) Прилагаме глобалния модел по сегмент
    seg["DeltaV_model_raw"] = slope_poly(seg["slope_pct"])

    # омекотяване
    seg["DeltaV_model"] = alpha_slope * seg["DeltaV_model_raw"]

    # извън диапазона на модела или около равно -> без корекция
    outside_mask = (
        (seg["slope_pct"] <= SLOPE_MODEL_MIN)
        | (seg["slope_pct"] >= SLOPE_MODEL_MAX)
        | (abs(seg["slope_pct"]) <= FLAT_BAND)
    )
    seg.loc[outside_mask, "DeltaV_model"] = 0.0

    # клипване в безопасен диапазон
    seg["DeltaV_model"] = seg["DeltaV_model"].clip(-dv_clip, dv_clip)

    # КОЕФИЦИЕНТ ПО СЕГМЕНТИ (както го описваш):
    # ако ΔV_model = -25% (изкачване, по-бавно от равно) → coef = 1 - (-0.25) = 1.25
    # ако ΔV_model = +25% (спускане, по-бързо от равно) → coef = 1 - 0.25   = 0.75
    seg["coef_slope"] = 1.0 - seg["DeltaV_model"] / 100.0

    # защита – да не стане 0 или отрицателно
    seg.loc[seg["coef_slope"] < 0.2, "coef_slope"] = 0.2

    # крайна скорост: V_glide, модулирана по наклон
    seg["V_final"] = seg["V_glide"] * seg["coef_slope"]

    # 4) Обобщение по активност
    activity_rows = []
    for aid, g in seg.groupby("activity_id"):
        g_train = g[
            (g["slope_pct"] > SLOPE_MODEL_MIN)
            & (g["slope_pct"] < SLOPE_MODEL_MAX)
        ]
        if len(g_train) > 0:
            w = g_train["duration_s"].values
            mean_slope_model = np.average(g_train["slope_pct"].values, weights=w)
            mean_DeltaV_real = np.average(g_train["DeltaV_real"].values, weights=w)
            n_slope_segments = len(g_train)
        else:
            mean_slope_model = np.nan
            mean_DeltaV_real = np.nan
            n_slope_segments = 0

        V_overall_real_seg = (g["V_kmh"] * g["duration_s"]).sum() / g["duration_s"].sum()
        V_overall_glide_seg = (g["V_glide"] * g["duration_s"]).sum() / g["duration_s"].sum()
        V_overall_final_seg = (g["V_final"] * g["duration_s"]).sum() / g["duration_s"].sum()

        activity_rows.append(
            {
                "activity_id": aid,
                "n_slope_segments": n_slope_segments,
                "mean_slope_model": mean_slope_model,
                "mean_DeltaV_real": mean_DeltaV_real,
                "V_overall_real": V_overall_real_seg,
                "V_overall_glide": V_overall_glide_seg,
                "V_overall_final": V_overall_final_seg,
            }
        )

    summary_df = pd.DataFrame(activity_rows)
    return seg, summary_df, slope_poly

# --------------------------------------------------------------------------------
# Модел 3 – Зони + пулс
# --------------------------------------------------------------------------------
def assign_zones(df: pd.DataFrame, cs: float, z_bounds: dict, speed_min_zone: float) -> pd.DataFrame:
    """
    - Зоните се базират на V_final (двойно модулирана скорост).
    - В зоните влизат само сегменти с V_kmh >= speed_min_zone.
    - Спусканията (slope <= -5%) при движение винаги са Z1,
      като им задаваме фиксирана "Z1 скорост" = горната граница на Z1 (напр. 0.8*CS).
    - Сегментите с V_kmh < speed_min_zone (стрелба/почивки) не влизат в зоните.
    """
    d = df.copy()
    if d.empty or cs <= 0:
        d["V_eff"] = np.nan
        d["ratio"] = np.nan
        d["zone"] = None
        return d

    move_mask = d["V_kmh"] >= speed_min_zone
    down_mask = (d["slope_pct"] <= DOWNHILL_ZONE1_THRESH) & move_mask
    flat_up_mask = (d["slope_pct"] > DOWNHILL_ZONE1_THRESH) & move_mask

    # Z1 – ГОРНА граница на диапазона
    z1_lo, z1_hi = z_bounds["Z1"]
    z1_ratio = z1_hi   # напр. 0.80 → 16 km/h при CS=20

    d["V_eff"] = np.nan
    d["ratio"] = np.nan
    d["zone"] = None

    # 1) сегменти без спускания – V_eff = V_final
    d.loc[flat_up_mask, "V_eff"] = d.loc[flat_up_mask, "V_final"]
    d.loc[flat_up_mask, "ratio"] = d.loc[flat_up_mask, "V_eff"] / cs

    def get_zone(r):
        for z_name, (lo, hi) in z_bounds.items():
            if lo <= r < hi:
                return z_name
        return "Z6+"

    d.loc[flat_up_mask, "zone"] = d.loc[flat_up_mask, "ratio"].apply(get_zone)

    # 2) спускания – „без усилие“, но с ratio = горна граница на Z1
    d.loc[down_mask, "ratio"] = z1_ratio
    d.loc[down_mask, "V_eff"] = z1_ratio * cs
    d.loc[down_mask, "zone"] = "Z1"

    # 3) много бавни сегменти (стрелба/почивки) не влизат в зоните
    return d


def zone_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Прави таблица по зони (без стрелба/почивки, които са zone=None):
    - време [min]
    - % от времето
    - средна V_eff [km/h]
    - среден HR
    """
    d = df[df["zone"].notna()].copy()
    if d.empty:
        return pd.DataFrame(
            columns=[
                "zone",
                "time_min",
                "time_percent",
                "V_eff_mean",
                "HR_mean",
            ]
        )

    total_time = d["duration_s"].sum()
    if total_time <= 0:
        total_time = 1.0

    rows = []
    for zone, g in d.groupby("zone"):
        T = g["duration_s"].sum()
        time_min = T / 60.0
        time_percent = 100.0 * T / total_time
        V_eff_mean = (g["V_eff"] * g["duration_s"]).sum() / T if T > 0 else np.nan
        if g["hr_mean"].notna().any():
            HR_mean = (g["hr_mean"] * g["duration_s"]).sum() / T
        else:
            HR_mean = np.nan

        rows.append(
            {
                "zone": zone,
                "time_min": time_min,
                "time_percent": time_percent,
                "V_eff_mean": V_eff_mean,
                "HR_mean": HR_mean,
            }
        )

    order = ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z6+"]
    df_z = pd.DataFrame(rows)
    df_z["zone"] = pd.Categorical(df_z["zone"], categories=order, ordered=True)
    df_z = df_z.sort_values("zone").reset_index(drop=True)
    return df_z


# --------------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------------
st.set_page_config(page_title="onFlows – Ski Glide + Slope + CS Zones", layout="wide")

st.title("onFlows – Ski Glide + Slope + CS Zones")
st.markdown(
    """
Модел за:
- **плъзгаемост (Glide)**  
- **влияние на наклона** (скорост → еквивалентна на равен терен)  
- **разпределение на натоварването по зони + пулс**  

Работи с няколко TCX файла едновременно.
"""
)

# --- Sidebar ---
st.sidebar.header("Настройки на модела")

alpha_glide = st.sidebar.slider(
    "Омекотяване на плъзгаемостта (α_glide)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = игнориране на плъзгаемостта, 1 = пълно влияние",
)

deg_glide = st.sidebar.selectbox(
    "Степен на модела за плъзгаемост V=f(slope)",
    options=[1, 2],
    index=1,
    help="1 = линеен, 2 = квадратичен модел",
)

alpha_slope = st.sidebar.slider(
    "Омекотяване на модела за наклона (α_slope)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0 = игнориране на наклона, 1 = пълно влияние на ΔV%(slope)",
)

cs_default = 20.0
cs = st.sidebar.number_input(
    "Критична скорост (CS) [km/h]",
    min_value=1.0,
    max_value=40.0,
    value=cs_default,
    step=0.5,
)

speed_min_zone = st.sidebar.number_input(
    "Мин. скорост за зони [km/h]",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="Сегменти с по-ниска скорост (стрелба, почивка) не влизат в зоните."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Зони по CS (ratio = V_eff / CS):**")
z_bounds = {
    "Z1": (0.00, 0.80),
    "Z2": (0.80, 0.90),
    "Z3": (0.90, 1.00),
    "Z4": (1.00, 1.10),
    "Z5": (1.10, 1.20),
    "Z6": (1.20, 999.0),
}
for z_name, (lo, hi) in z_bounds.items():
    st.sidebar.write(f"{z_name}: {lo:.2f} – {hi:.2f}")

# --- Качване на файлове ---
uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла", type=["tcx"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Моля, качи поне един TCX файл, за да започнем анализа.")
    st.stop()

# --------------------------------------------------------------------------------
# Парсване и сегментиране на всички активности
# --------------------------------------------------------------------------------
all_segments = []
info_rows = []

for i, f in enumerate(uploaded_files, start=1):
    activity_id = f"{i}: {f.name}"
    st.write(f"Обработка на **{activity_id}** ...")

    df_points_raw = parse_tcx(f, activity_id)
    if df_points_raw.empty:
        st.warning(f"⚠ Няма валидни Trackpoint-и в файла: {f.name}")
        continue

    # 1) Реална скорост – директно от суровите данни (без модулации)
    df_raw = df_points_raw.sort_values("time").reset_index(drop=True)
    t_start = df_raw["time"].iloc[0]
    t_end = df_raw["time"].iloc[-1]
    total_time_s_raw = (t_end - t_start).total_seconds()
    total_dist_m_raw = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
    if total_time_s_raw > 0:
        V_overall_real_raw = (total_dist_m_raw / total_time_s_raw) * 3.6
    else:
        V_overall_real_raw = np.nan

    # 2) Сегменти за моделите – ВЪРХУ СУРОВИТЕ ДАННИ
    df_clean = preprocess_points(df_points_raw)
    seg_df = build_segments(df_clean, activity_id, T_SEG)
    if seg_df.empty:
        st.warning(f"⚠ Няма валидни сегменти в активност: {activity_id}")
        continue

    all_segments.append(seg_df)

    info_rows.append(
        {
            "activity_id": activity_id,
            "n_segments": len(seg_df),
            "time_min": total_time_s_raw / 60.0,
            "dist_km": total_dist_m_raw / 1000.0,
            "V_overall_real": V_overall_real_raw,
        }
    )

if not all_segments:
    st.error("Не успях да извадя валидни сегменти от нито един файл.")
    st.stop()

segments = pd.concat(all_segments, ignore_index=True)
info_df = pd.DataFrame(info_rows)

st.subheader("Базова информация по активност (реални данни)")
st.dataframe(
    info_df.style.format(
        {
            "time_min": "{:.1f}",
            "dist_km": "{:.2f}",
            "V_overall_real": "{:.2f}",
        }
    )
)

# --------------------------------------------------------------------------------
# Модел 1 – Плъзгаемост
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Модел 1 – Плъзгаемост (Glide)")

segments_glide, glide_summary, glide_poly = compute_glide_model(
    segments, alpha_glide=alpha_glide, deg_glide=deg_glide
)

if glide_summary.empty:
    st.warning("Няма достатъчно downhill сегменти за модел на плъзгаемостта.")
else:
    # заменяме реалната скорост с константата от суровия TCX
    real_map = info_df.set_index("activity_id")["V_overall_real"]
    glide_summary["V_overall_real"] = glide_summary["activity_id"].map(real_map)
    glide_summary["V_overall_glide"] = (
        glide_summary["V_overall_real"] / glide_summary["K_glide_soft"]
    )

    st.markdown("**Обобщение по активност (реална vs. коригирана по плъзгаемост скорост):**")
    st.dataframe(
        glide_summary.style.format(
            {
                "mean_down_slope": "{:.2f}",
                "mean_down_V_real": "{:.2f}",
                "V_down_model": "{:.2f}",
                "K_glide_raw": "{:.3f}",
                "K_glide_soft": "{:.3f}",
                "V_overall_real": "{:.2f}",
                "V_overall_glide": "{:.2f}",
            }
        )
    )

    down_plot_df = segments_glide[
        segments_glide["is_downhill"] & segments_glide["is_prev_downhill"]
    ].copy()

    if not down_plot_df.empty and glide_poly is not None:
        st.markdown("**Графика: зависимост между наклон и скорост (Glide модел)**")

        scatter = (
            alt.Chart(down_plot_df)
            .mark_circle(size=30, opacity=0.4)
            .encode(
                x=alt.X("slope_pct", title="Наклон [%]"),
                y=alt.Y("V_kmh", title="Скорост [km/h]"),
                color=alt.Color("activity_id", title="Активност"),
                tooltip=["activity_id", "slope_pct", "V_kmh"],
            )
        )

        x_min = float(down_plot_df["slope_pct"].min())
        x_max = float(down_plot_df["slope_pct"].max())
        x_line = np.linspace(x_min, x_max, 100)
        y_line = glide_poly(x_line)
        line_df = pd.DataFrame({"slope_pct": x_line, "V_model": y_line})

        line = (
            alt.Chart(line_df)
            .mark_line()
            .encode(
                x="slope_pct",
                y=alt.Y("V_model", title="Скорост [km/h]"),
                color=alt.value("black"),
            )
        )

        st.altair_chart(scatter + line, use_container_width=True)
    else:
        st.info("Няма достатъчно downhill сегменти за визуализация на Glide модела.")

# --------------------------------------------------------------------------------
# Модел 2 – Наклон (ГЛОБАЛЕН ΔV%(slope) ПО СЕГМЕНТИ)
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Модел 2 – Влияние на наклона (ΔV%) – глобален модел по сегменти")

segments_slope, slope_summary, slope_poly = compute_slope_model(
    segments_glide,
    alpha_slope=alpha_slope,
    dv_clip=40.0,
)

if slope_summary.empty or slope_summary["n_slope_segments"].sum() == 0:
    st.warning("Няма достатъчно сегменти за модел на наклона.")
else:
    # коефициент на релеф от сегментните средни
    slope_summary["terrain_factor"] = (
        slope_summary["V_overall_final"] / slope_summary["V_overall_glide"]
    ).replace([np.inf, -np.inf], np.nan)

    real_map = info_df.set_index("activity_id")["V_overall_real"]
    slope_summary["V_overall_real"] = slope_summary["activity_id"].map(real_map)

    glide_map = glide_summary.set_index("activity_id")["V_overall_glide"]
    slope_summary["V_overall_glide"] = slope_summary["activity_id"].map(glide_map)

    st.markdown(
        """
В този модел за всеки сегмент използваме **V_glide** и го сравняваме със
**средната скорост при почти равен наклон (-1..+1%) за същата активност**.
От всички сегменти обучаваме глобален модел **ΔV%(slope)** (квадратичен полином).

След това за всеки сегмент с наклон *s*:
- взимаме ΔV_model(s) от полинома,
- омекотяваме го чрез α_slope и клипваме (напр. в диапазон [-40%, +40%]),
- образуваме множител *f_poly(s) = 1 + ΔV_model/100*,
- и приравняваме към равно чрез коефициент **coef_slope(s) = 1 / f_poly(s)**:

\\[
V_{final} = V_{glide} \\,\\cdot\\, coef\\_slope(s).
\\]

Т.е. **глайдът** дава един глобален коефициент за активността,  
а **наклонът** – сегментен коефициент, който зависи от наклона на сегмента.
"""
    )

    st.markdown("**Обобщение по активност:**")
    st.dataframe(
        slope_summary[[
            "activity_id",
            "n_slope_segments",
            "mean_slope_model",
            "mean_DeltaV_real",
            "V_overall_real",
            "V_overall_glide",
            "V_overall_final",
        ]].style.format(
            {
                "mean_slope_model": "{:.2f}",
                "mean_DeltaV_real": "{:.2f}",
                "V_overall_real": "{:.2f}",
                "V_overall_glide": "{:.2f}",
                "V_overall_final": "{:.2f}",
            }
        )
    )

    train_plot = segments_slope[
        (segments_slope["slope_pct"] > SLOPE_MODEL_MIN)
        & (segments_slope["slope_pct"] < SLOPE_MODEL_MAX)
    ].copy()

    if not train_plot.empty and slope_poly is not None:
        st.markdown(
            "**Графика: ΔV_real% спрямо наклон + глобален квадратичен модел ΔV_model%**"
        )

        scatter2 = (
            alt.Chart(train_plot)
            .mark_circle(size=30, opacity=0.4)
            .encode(
                x=alt.X("slope_pct", title="Наклон [%]"),
                y=alt.Y("DeltaV_real", title="ΔV_real [%]"),
                color=alt.Color("activity_id", title="Активност"),
                tooltip=["activity_id", "slope_pct", "DeltaV_real"],
            )
        )

        x_min2 = float(train_plot["slope_pct"].min())
        x_max2 = float(train_plot["slope_pct"].max())
        x_line2 = np.linspace(x_min2, x_max2, 100)
        y_line2 = slope_poly(x_line2)
        line2_df = pd.DataFrame({"slope_pct": x_line2, "DeltaV_model": y_line2})

        line2 = (
            alt.Chart(line2_df)
            .mark_line()
            .encode(
                x="slope_pct",
                y=alt.Y("DeltaV_model", title="ΔV_model [%]"),
                color=alt.value("black"),
            )
        )

        st.altair_chart(scatter2 + line2, use_container_width=True)
    else:
        st.info("Няма достатъчно данни за визуализация на глобалния ΔV% модел.")

# --------------------------------------------------------------------------------
# Модел 3 – Зони + пулс
# --------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Модел 3 – Разпределение по зони + пулс")

activity_options = ["Всички активности"] + sorted(segments_slope["activity_id"].unique())
selected_activity = st.selectbox("Избери активност за зонирането", activity_options)

if selected_activity == "Всички активности":
    seg_for_zones = segments_slope.copy()
else:
    seg_for_zones = segments_slope[segments_slope["activity_id"] == selected_activity].copy()

seg_zoned = assign_zones(
    seg_for_zones,
    cs=cs,
    z_bounds=z_bounds,
    speed_min_zone=speed_min_zone,
)
zones_table = zone_summary(seg_zoned)

st.markdown(
    """
Зонирането се прави върху **крайната модулирана скорост** `V_final`  
(коригирана по плъзгаемост и глобален наклонов модел), приравнена към равен терен
и референтна плъзгаемост.

В зоните влизат само движещите се сегменти (V ≥ праг),  
спусканията със **slope ≤ -5%** при движение се считат за **без усилие**
и се наливат директно в Z1 с ефективна скорост = горната граница на Z1 (напр. 0.8·CS).
"""
)

st.dataframe(
    zones_table.style.format(
        {
            "time_min": "{:.1f}",
            "time_percent": "{:.1f}",
            "V_eff_mean": "{:.2f}",
            "HR_mean": "{:.0f}",
        }
    )
)

if not zones_table.empty:
    st.markdown("**Графика: % време по зони**")
    chart = (
        alt.Chart(zones_table)
        .mark_bar()
        .encode(
            x=alt.X("zone", title="Зона"),
            y=alt.Y("time_percent", title="% време"),
            tooltip=["zone", "time_min", "time_percent", "V_eff_mean", "HR_mean"],
        )
    )
    st.altair_chart(chart, use_container_width=True)

st.success(
    "Glide моделът дава един глобален коефициент за активността, "
    "а наклоновият модел вече се прилага по сегменти чрез coef_slope(s), "
    "с омекотяване (α_slope) и безопасно клипване на ΔV%(slope), за да няма „излитане“ при крайни случаи."
)
