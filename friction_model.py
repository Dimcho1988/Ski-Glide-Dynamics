import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

G = 9.81  # m/s^2


def parse_tcx_to_df(uploaded_file) -> Optional[pd.DataFrame]:
    """Парсва TCX до DataFrame с индекс време и колони altitude, distance."""
    try:
        content = uploaded_file.read()
        uploaded_file.seek(0)
        tree = ET.parse(BytesIO(content))
        root = tree.getroot()
    except Exception:
        return None

    ns = {}
    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns["tcx"] = uri

    times, alts, dists = [], [], []

    for tp in root.findall(".//tcx:Trackpoint", ns):
        time_el = tp.find("tcx:Time", ns)
        alt_el = tp.find("tcx:AltitudeMeters", ns)
        dist_el = tp.find("tcx:DistanceMeters", ns)

        if time_el is None or alt_el is None or dist_el is None:
            continue

        try:
            t = pd.to_datetime(time_el.text)
            alt = float(alt_el.text)
            dist = float(dist_el.text)
        except Exception:
            continue

        times.append(t)
        alts.append(alt)
        dists.append(dist)

    if not times:
        return None

    df = pd.DataFrame({"time": times, "altitude": alts, "distance": dists})
    df = df.sort_values("time").drop_duplicates(subset=["time"]).set_index("time")

    # Ресемплиране до 1 Hz
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1S")
    df = df.reindex(full_index)

    # Интерполация
    df["altitude"] = df["altitude"].interpolate(method="time", limit_direction="both")
    df["distance"] = df["distance"].interpolate(method="time", limit_direction="both")
    df["distance"] = df["distance"].cummax()

    return df


def _compute_trend_and_kinematics(
    df: pd.DataFrame,
    trend_window: int,
    slope_window: int = 11,
) -> pd.DataFrame:
    """
    1) Напълно изглаждаме (trend) височина и дистанция.
    2) Изчисляваме:
       - реална скорост v_real и ускорение a_real;
       - наклон (от тренда);
       - теоретично ускорение a_theor (само гравитация, без триене);
       - теоретична скорост v_theor чрез интегриране на a_theor.
    """
    out = df.copy()

    # Изглаждане (trend) – колкото по-голям прозорец, толкова по-"Гармин"
    Wt = max(9, int(trend_window))
    if Wt % 2 == 0:
        Wt += 1

    out["h_trend"] = out["altitude"].rolling(Wt, center=True, min_periods=1).mean()
    out["d_trend"] = out["distance"].rolling(Wt, center=True, min_periods=1).mean()

    # Наклон върху по-дълъг прозорец (slope_window ~ 10–11 s)
    Ws = max(5, int(slope_window))
    if Ws % 2 == 0:
        Ws += 1
    half = Ws // 2

    dh = out["h_trend"].shift(-half) - out["h_trend"].shift(half)
    dd = out["d_trend"].shift(-half) - out["d_trend"].shift(half)

    # Ако почти няма движение по дистанция → няма смислен наклон
    small_dd = dd.abs() < 0.5
    dh[small_dd] = np.nan
    dd[small_dd] = np.nan

    slope = (dh / dd).clip(-0.3, 0.3)  # -30% до +30%
    out["slope"] = slope
    out["slope_pct"] = slope * 100.0

    # Реална скорост v_real = d(d_trend)/dt (dt = 1 s) + леко изглаждане
    v_real = out["d_trend"].diff().fillna(0.0)
    v_real = v_real.rolling(3, center=True, min_periods=1).mean()
    out["v"] = v_real  # ще я използваме и като "оригинална" скорост в апа

    # Реално ускорение a_real = dv_real/dt + леко изглаждане
    a_real = (v_real.shift(-1) - v_real.shift(1)) / 2.0
    a_real.iloc[0] = 0.0
    a_real.iloc[-1] = 0.0
    a_real = a_real.rolling(3, center=True, min_periods=1).mean()
    out["a_real"] = a_real

    # Теоретично ускорение (без триене): a_theor = g * sin(theta)
    theta = np.arctan(slope)
    a_theor = G * np.sin(theta)
    # Нисходящ наклон → a_theor > 0; ако не е така, няма ускорение от гравитация
    a_theor[a_theor <= 0] = np.nan
    out["a_theor"] = a_theor

    # Теоретична скорост v_theor чрез интеграция на a_theor
    v_theor = np.zeros(len(out))
    # стартова скорост ~ реалната в началото (ако е положителна)
    v_theor[0] = max(v_real.iloc[0], 0.0)

    for i in range(1, len(out)):
        prev = v_theor[i - 1]
        a_th_prev = a_theor.iloc[i - 1]
        if np.isfinite(a_th_prev):
            v_theor[i] = max(0.0, prev + a_th_prev * 1.0)  # dt=1 s
        else:
            # ако нямаме теоретично ускорение, просто "носим" предишната скорост
            v_theor[i] = prev

    out["v_theor"] = v_theor

    return out


def _mark_free_glide_segments(
    df: pd.DataFrame,
    slope_thr: float,
    v_min: float,
    t_cut: int,
    t_min_free: int,
    a_max: float,
) -> pd.DataFrame:
    """
    Маркира free-glide секунди:
    - достатъчно стръмен наклон (slope_pct <= slope_thr)
    - достатъчно висока скорост (v > v_min)
    - ниско реално ускорение (почти без оттласкване): |a_real| <= a_max
    - поне t_min_free секунди след изрязване на първите t_cut
    """
    out = df.copy()

    is_desc = (out["slope_pct"] <= slope_thr) & (out["v"] > v_min)
    low_acc = out["a_real"].abs() <= a_max

    candidate = is_desc & low_acc
    out["free_glide"] = False

    if not candidate.any():
        return out

    shift = candidate.shift(1, fill_value=False)
    starts = out.index[(~shift) & candidate]
    ends = out.index[(shift) & (~candidate)]

    if candidate.iloc[-1]:
        ends = ends.append(pd.Index([out.index[-1]]))

    if len(ends) < len(starts):
        ends = ends.append(pd.Index([out.index[-1]]))

    for s, e in zip(starts, ends):
        seg = out.loc[s:e].index
        if len(seg) <= t_cut:
            continue
        seg = seg[t_cut:]
        if len(seg) < t_min_free:
            continue
        out.loc[seg, "free_glide"] = True

    return out


def compute_session_friction(
    df: pd.DataFrame,
    slope_thr: float = -5.0,
    v_min: float = 2.0,
    trend_window: int = 31,
    t_cut: int = 5,
    t_min_free: int = 10,
    a_max: float = 0.4,
    mu_min: float = 0.0,   # запазени параметри за съвместимост (не ползваме директно μ)
    mu_max: float = 0.2,
    **kwargs,
) -> Dict[str, Any]:
    """
    НОВА ЛОГИКА:
    - работим върху изгладения тренд;
    - сравняваме реално и теоретично ускорение/скорост;
    - извеждаме C_session – коефициент на мапване между реални и идеални условия.

    За съвместимост C_session се връща като 'mu_session', за да не чупим UI-то.
    """
    out = _compute_trend_and_kinematics(df, trend_window=trend_window, slope_window=11)
    out = _mark_free_glide_segments(out, slope_thr, v_min, t_cut, t_min_free, a_max)

    free = out["free_glide"].fillna(False)
    idx = out.index[free]

    out["ratio_a"] = np.nan
    out["ratio_v"] = np.nan
    out["C_point"] = np.nan  # локален коефициент

    C_vals = []

    for t in idx:
        row = out.loc[t]
        a_r = row["a_real"]
        a_th = row["a_theor"]
        v_r = row["v"]
        v_th = row["v_theor"]

        # Имаме смисъл само ако теоретичните величини са положителни
        if not (np.isfinite(a_r) and np.isfinite(a_th) and a_th > 0):
            continue
        if not (np.isfinite(v_r) and np.isfinite(v_th) and v_th > 0):
            continue

        ratio_a = a_r / a_th
        ratio_v = v_r / v_th

        # Физически разумни граници, за да режем екстремите
        if not (0.0 < ratio_a < 2.0 and 0.0 < ratio_v < 2.0):
            continue

        C_point = 0.5 * (ratio_a + ratio_v)

        out.at[t, "ratio_a"] = ratio_a
        out.at[t, "ratio_v"] = ratio_v
        out.at[t, "C_point"] = C_point

        C_vals.append(C_point)

    C_session = float(np.median(C_vals)) if C_vals else np.nan
    valid_seconds = len(C_vals)

    # Връщаме C_session под ключ 'mu_session', за да не чупим streamlit_app.py
    return {
        "df": out,
        "mu_session": C_session,
        "valid_seconds": valid_seconds,
    }


def apply_reference_and_modulation(
    activity: Dict[str, Any],
    mu_ref: float,
    delta_max_up: float = 0.20,
    delta_max_down: float = 0.15,
) -> None:
    """
    Модулиране на скоростта на база C_session:

    - C_session (тук в променлива mu_sess) – колко близо сме до идеални условия.
      По-висока стойност = по-добро плъзгане.
    - За да мапнем към референцията:
        K_raw = C_ref / C_session  (mu_ref / mu_sess)
    - Ограничаваме K с delta_max_up/down и множим скоростта.
    """
    df = activity["df"]
    mu_sess = activity.get("mu_session", np.nan)  # реално това е C_session

    if not np.isfinite(mu_ref) or not np.isfinite(mu_sess) or mu_ref <= 0 or mu_sess <= 0:
        activity["FI"] = np.nan
        activity["K"] = np.nan
        df["v_mod"] = np.nan
        return

    # Индекс FI – отношение между текущи и референтни условия
    FI = mu_sess / mu_ref

    # Скалинг за скоростта – мапваме към референтните условия
    K_raw = mu_ref / mu_sess  # ако C_session < C_ref → K_raw > 1 → увеличаваме скоростта

    delta = K_raw - 1.0
    delta = np.clip(delta, -delta_max_down, delta_max_up)

    K = 1.0 + delta
    df["v_mod"] = df["v"] * K

    activity["FI"] = float(FI)
    activity["K"] = float(K)
