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

SLOPE_MODEL_MIN = -3.0   # долна граница за наклонов модел (изкачвания и леко надолу)
SLOPE_MODEL_MAX = 10.0   # горна граница за наклонов модел

FLAT_BAND = 1.0          # |slope| <= 1% се счита за "равно"

DOWNHILL_ZONE1_THRESH = -3.0  # всичко под -3% -> Z1, "без усилие"


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
    НЯМА тежко филтриране – само:
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
# Модел 1 – Плъзгаемост (Glide) с асиметрично, плавно омекотяване
# --------------------------------------------------------------------------------
def compute_glide_model(segments: pd.DataFrame, alpha_glide: float, deg_glide: int = 2):
    """
    Модел за плъзгаемост (Glide) с асиметрично, плавно омекотяване на K_raw.

    Стъпки:
    1) От downhill сегментите (между DOWN_MIN и DOWN_MAX, предхождани от downhill)
       строим глобален полином V = poly(slope).

    2) За всяка активност A:
       - среден наклон на нейните downhill сегменти: s̄_A
       - средна реална скорост върху тях: V̄_real,A
       - моделна скорост: V_down_model = poly(s̄_A)
       - суров коефициент на плъзгаемост: K_raw = V̄_real,A / V_down_model

    3) If K_raw е извън [0.2, 2.0] → считаме го за счупен и го правим 1.0.

    4) Асиметрично омекотяване на Δ = K_raw - 1:
       - при бързи ски (Δ>0): плавно омекотяване от +10% до +15%, после макс.
       - при бавни ски (Δ<0): плавно от -20% до -30%.

    5) Alpha_glide контролира силата на ефекта:
       K_soft = 1 + alpha_glide * (K_mod - 1)

    6) За всеки сегмент:
       V_glide = V_kmh / K_soft
    """

    # вътрешна помощна функция – асиметрично, плавно омекотяване
    def soften_delta_asym(delta: float) -> float:
        """
        Асиметрично, плавно омекотяване на Δ = K_raw - 1.

        - За бързи ски (Δ > 0):
            * до +10% (0.10) -> без омекотяване
            * между +10% и +15% -> плавен преход (smoothstep)
            * над +15% -> максимум омекотяване (M_MIN_POS)
        - За бавни ски (Δ < 0):
            * до -20% (0.20) -> без омекотяване
            * между -20% и -30% -> плавен преход
            * под -30% -> максимум омекотяване (M_MIN_NEG)
        """
        if not np.isfinite(delta):
            return 0.0
        if delta == 0.0:
            return 0.0

        sign = 1.0 if delta > 0 else -1.0
        a = abs(delta)

        if sign > 0:
            # БЪРЗИ СКИ (K_raw > 1)
            DELTA0 = 0.10   # до +10% -> вярваме напълно
            DELTA1 = 0.15   # от +15% нагоре -> крайност
            M_MIN  = 0.4    # в опашката ползваме 40% от отклонението
        else:
            # БАВНИ СКИ (K_raw < 1)
            DELTA0 = 0.20   # до -20% -> вярваме напълно
            DELTA1 = 0.30   # от -30% надолу -> крайност
            M_MIN  = 0.7    # в опашката ползваме 70% от отклонението

        # Централна зона – без омекотяване
        if a <= DELTA0:
            return delta

        # Опашка – максимално омекотяване
        if a >= DELTA1:
            return sign * a * M_MIN

        # Преходна зона – smoothstep между 1.0 и M_MIN
        t = (a - DELTA0) / (DELTA1 - DELTA0)  # t ∈ (0,1)
        h = 3.0 * t * t - 2.0 * t * t * t     # smoothstep
        m = 1.0 - (1.0 - M_MIN) * h
        return sign * a * m

    seg = segments.copy()
    if seg.empty:
        return seg, pd.DataFrame(), None

    seg = seg.sort_values(["activity_id", "seg_id"]).reset_index(drop=True)

    # 1) маркираме downhill сегментите
    seg["is_downhill"] = (seg["slope_pct"] >= DOWN_MIN) & (seg["slope_pct"] <= DOWN_MAX)

    seg["is_prev_downhill"] = False
    for aid, g in seg.groupby("activity_id"):
        idx = g.index
        prev_down = g["is_downhill"].shift(1).fillna(False).values
        seg.loc[idx, "is_prev_downhill"] = prev_down

    downhill_mask = seg["is_downhill"] & seg["is_prev_downhill"]
    down_df = seg.loc[downhill_mask].copy()

    # ако няма достатъчно downhill → без модел
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

    # 2) глобален полином V = poly(slope)
    x = down_df["slope_pct"].values
    y = down_df["V_kmh"].values
    try:
        coeffs = np.polyfit(x, y, deg=deg_glide)
        glide_poly = np.poly1d(coeffs)
    except Exception:
        glide_poly = None

    # 3) K_raw по активност + омекотяване
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

            # сурово K_raw
            if V_down_model <= 0:
                K_raw = 1.0
            else:
                K_raw = mean_down_V_real / V_down_model

            # груба защита
            if (not np.isfinite(K_raw)) or (K_raw <= 0.2) or (K_raw >= 2.0):
                K_raw = 1.0
            else:
                # асиметрично, плавно омекотяване
                delta = K_raw - 1.0
                delta_soft = soften_delta_asym(delta)
                K_raw = 1.0 + delta_soft

            # финално омекотяване
            K_soft = 1.0 + alpha_glide * (K_raw - 1.0)
            n_down = len(g_down)

        seg.loc[seg["activity_id"] == aid, "K_glide_raw"] = K_raw
        seg.loc[seg["activity_id"] == aid, "K_glide_soft"] = K_soft

        V_overall_real = (g["V_kmh"] * g["duration_s"]).sum() / g["duration_s"].sum()
        V_overall_glide = ((g["V_kmh"] / K_soft) * g["duration_s"]).sum() / g["duration_s"].sum()

        activity_rows.append(
            {
                "activity_id": aid,
                "n_downhill": n_down,
                "mean_down_slope": mean_down_slope,
                "mean_down_V_real": mean_down_V_real,
                "V_down_model": V_down_model,
                "K_glide_raw": K_raw,
                "K_glide_soft": K_soft,
                "V_overall_real": V_overall_real,
                "V_overall_glide": V_overall_glide,
            }
        )

    seg["V_glide"] = seg["V_kmh"] / seg["K_glide_soft"]
    summary_df = pd.DataFrame(activity_rows)

    return seg, summary_df, glide_poly


# --------------------------------------------------------------------------------
# Модел 2 – Влияние на наклона (полином върху r = V_glide / V_flat,A)
# --------------------------------------------------------------------------------
def compute_slope_model(segments_glide: pd.DataFrame):
    """
    Прост полиномен модел за влияние на наклона върху скоростта (след Glide).

    1) За всяка активност A:
       - намираме V_flat,A = средна V_glide при почти равен наклон (|slope| <= FLAT_BAND).
       - за всеки сегмент S с наклон s_S в диапазона (SLOPE_MODEL_MIN, SLOPE_MODEL_MAX):
           r_S = V_glide,S / V_flat,A  (относителна скорост спрямо равно)

    2) Обединяваме всички (s_S, r_S) от всички активности и обучаваме глобален полином:
           r_model(s) = a0 + a1*s + a2*s^2

    3) Приравняваме към равно за всеки сегмент:
           f_slope(s) = 1 / r_model(s)
           V_final_raw = V_glide * f_slope(s)

       Ограничаваме r_model в [0.5, 1.3] и f_slope в [0.7, 1.5], за да няма екстремни стойности.

    4) Допълнителна защита по активност:
       - за всяка активност гледаме средното отношение
           mean_factor_A = mean(V_final_raw) / mean(V_glide)
       - ограничаваме го в [0.9, 1.1] (±10% глобален ефект от наклона)
         чрез пренормализиране на f_slope и V_final_raw.
    """

    seg = segments_glide.copy()
    if seg.empty:
        return seg, pd.DataFrame(), None

    # 1) V_flat по активност
    vflat_map = {}
    for aid, g in seg.groupby("activity_id"):
        flat = g[g["slope_pct"].abs() <= FLAT_BAND]
        if len(flat) < 5:
            flat = g[
                (g["slope_pct"] > SLOPE_MODEL_MIN)
                & (g["slope_pct"] < SLOPE_MODEL_MAX)
            ]
            if flat.empty:
                flat = g

        V_flat_A = (flat["V_glide"] * flat["duration_s"]).sum() / flat["duration_s"].sum()
        vflat_map[aid] = V_flat_A

    seg["V_flat_A"] = seg["activity_id"].map(vflat_map)
    seg["r_rel"] = seg["V_glide"] / seg["V_flat_A"]
    seg["DeltaV_real_pct"] = 100.0 * (seg["r_rel"] - 1.0)

    # 2) Обучаващ набор
    train = seg[
        (seg["slope_pct"] > SLOPE_MODEL_MIN)
        & (seg["slope_pct"] < SLOPE_MODEL_MAX)
        & seg["V_flat_A"].notna()
        & np.isfinite(seg["r_rel"])
    ].copy()

    train = train[(train["r_rel"] >= 0.4) & (train["r_rel"] <= 1.6)]

    if len(train) < 20:
        seg["DeltaV_model"] = 0.0
        seg["f_slope"] = 1.0
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
    y = train["r_rel"].values
    try:
        coeffs = np.polyfit(x, y, deg=2)
        slope_poly = np.poly1d(coeffs)
    except Exception:
        slope_poly = None

    if slope_poly is None:
        seg["DeltaV_model"] = 0.0
        seg["f_slope"] = 1.0
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
                        "V_overall_final": (g["V_gl
