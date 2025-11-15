"""
friction_model.py  (v6 — trend smoothing + segment μ)

Имплементация на модела за коефициент на триене и модулиране на скоростта
с фиксирана референтна активност.

Новото във v6:
- Добавена е "плаваща" тренд линия за височина и дистанция:
  * h_trend = rolling mean на h_smooth с по-широк прозорец (примерно 31 s)
  * d_trend = rolling mean на d_smooth със същия прозорец
- slope и v се изчисляват от h_trend / d_trend вместо от h_smooth / d_smooth.
- Продължаваме да използваме сегментен подход (v(t) регресия → a_seg → μ_seg).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

G = 9.81  # гравитационно ускорение, m/s^2


# --------------------------------------------------------------
#  PARSING
# --------------------------------------------------------------

def _load_tcx(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """Парсва TCX файл и връща DataFrame: time_s, h, d."""
    try:
        content = file_obj.read()
        file_obj.seek(0)
        root = ET.fromstring(content)
    except Exception as e:
        raise ValueError(f"Грешка при парсване на TCX ({filename}): {e}")

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    tps = root.findall(".//tcx:Trackpoint", ns)

    times, alts, dists = [], [], []
    for tp in tps:
        t_el = tp.find("tcx:Time", ns)
        a_el = tp.find("tcx:AltitudeMeters", ns)
        d_el = tp.find("tcx:DistanceMeters", ns)
        if t_el is None or a_el is None or d_el is None:
            continue
        try:
            times.append(pd.to_datetime(t_el.text))
            alts.append(float(a_el.text))
            dists.append(float(d_el.text))
        except Exception:
            continue

    if not times:
        raise ValueError("Няма валидни Trackpoint в TCX файла.")

    t0 = times[0]
    time_s = np.array([(t - t0).total_seconds() for t in times], float)

    return pd.DataFrame({"time_s": time_s, "h": alts, "d": dists})


def _load_csv(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """Парсва CSV файл към: time_s, h, d."""
    try:
        df_raw = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Грешка при CSV ({filename}): {e}")

    cols = {c.lower(): c for c in df_raw.columns}

    # време
    for k in ["time", "seconds", "t"]:
        if k in cols:
            time_col = cols[k]
            break
    else:
        raise ValueError("CSV трябва да има time/seconds/t")

    # височина
    for k in ["elevation", "altitude", "alt", "h"]:
        if k in cols:
            h_col = cols[k]
            break
    else:
        raise ValueError("CSV трябва да има elevation/altitude/alt/h")

    # дистанция
    for k in ["distance", "dist", "d"]:
        if k in cols:
            d_col = cols[k]
            break
    else:
        raise ValueError("CSV трябва да има distance/dist/d")

    tvals = df_raw[time_col].values

    # време → секунди
    if not np.issubdtype(df_raw[time_col].dtype, np.number):
        times = pd.to_datetime(df_raw[time_col])
        t0 = times.iloc[0]
        time_s = (times - t0).dt.total_seconds().astype(float).values
    else:
        time_s = tvals.astype(float)
        time_s -= time_s[0]

    return pd.DataFrame(
        {
            "time_s": time_s,
            "h": df_raw[h_col].astype(float).values,
            "d": df_raw[d_col].astype(float).values,
        }
    )


def load_activity(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """Unified loader."""
    name = filename.lower()
    if name.endswith(".tcx"):
        return _load_tcx(file_obj, filename)
    if name.endswith(".csv"):
        return _load_csv(file_obj, filename)
    raise ValueError("Поддържани формати: .tcx, .csv")


# --------------------------------------------------------------
#  RESAMPLING & KINEMATICS (с trend линия)
# --------------------------------------------------------------

def _resample_and_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ресемплиране на 1 Hz + изглаждане + trend линия.

    - h_smooth: 5 s rolling mean (локално изглаждане)
    - d_smooth: 3-точково rolling mean
    - h_trend: 31 s rolling mean на h_smooth (trend линия, подобно на Garmin)
    - d_trend: 31 s rolling mean на d_smooth
    """
    df = df.sort_values("time_s").reset_index(drop=True)
    t_min, t_max = df["time_s"].min(), df["time_s"].max()

    t_grid = np.arange(0, int(np.round(t_max - t_min)) + 1, 1.0)
    base = pd.DataFrame({"time_s": t_grid}).set_index("time_s")

    df_interp = (
        df.set_index("time_s")[["h", "d"]]
        .reindex(base.index)
        .interpolate("linear", limit_direction="both")
        .reset_index()
    )

    # локално изглаждане
    df_interp["h_smooth"] = df_interp["h"].rolling(5, center=True, min_periods=1).mean()
    df_interp["d_smooth"] = df_interp["d"].rolling(3, center=True, min_periods=1).mean()

    # trend линия (по-широк прозорец)
    trend_window = 31  # ~30 s прозорец
    df_interp["h_trend"] = (
        df_interp["h_smooth"].rolling(trend_window, center=True, min_periods=1).mean()
    )
    df_interp["d_trend"] = (
        df_interp["d_smooth"].rolling(trend_window, center=True, min_periods=1).mean()
    )

    return df_interp


def _compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Смята slope, v, a, използвайки trend линия.

    - slope = Δh_trend / Δd_trend
    - v = Δd_trend / Δt  (тук Δt = 1 s)
    - a = централна разлика на v
    """
    df = df.copy()

    dh_trend = df["h_trend"].diff()
    dd_trend = df["d_trend"].diff()

    df["slope"] = dh_trend / dd_trend.replace(0, np.nan)
    df["v"] = dd_trend  # защото dt = 1 s
    df["a"] = (df["v"].shift(-1) - df["v"].shift(1)) / 2.0

    return df


# --------------------------------------------------------------
#  FREE-GLIDE DETECTION
# --------------------------------------------------------------

def _detect_free_glide_mask(
    df: pd.DataFrame,
    S_thr_percent: float,
    v_min: float,
    T_cut: int,
    Tmin_free: int,
) -> pd.Series:
    """Създава маска за free-glide секунди."""
    S_thr = S_thr_percent / 100.0
    is_desc = (df["slope"] < S_thr) & (df["v"] > v_min)

    is_free = pd.Series(False, index=df.index)
    in_block, s_idx = False, None

    for i, flag in enumerate(is_desc.values):
        if flag and not in_block:
            in_block = True
            s_idx = i
        elif not flag and in_block:
            e_idx = i - 1
            length = e_idx - s_idx + 1
            if length > T_cut + Tmin_free:
                rs, re = s_idx + T_cut, e_idx
                is_free.iloc[rs : re + 1] = True
            in_block = False

    # ако завършва вътре в блок
    if in_block:
        e_idx = len(is_desc) - 1
        length = e_idx - s_idx + 1
        if length > T_cut + Tmin_free:
            rs, re = s_idx + T_cut, e_idx
            is_free.iloc[rs : re + 1] = True

    return is_free


def _extract_segments(is_free: pd.Series, Tmin_seg: int) -> List[Tuple[int, int]]:
    """Връща списък от (start, end) free-glide сегменти."""
    segs: List[Tuple[int, int]] = []
    in_seg, s_idx = False, None

    for i, fl in enumerate(is_free.values):
        if fl and not in_seg:
            in_seg = True
            s_idx = i
        elif (not fl or i == len(is_free) - 1) and in_seg:
            e_idx = i - 1 if not fl else i
            if e_idx - s_idx + 1 >= Tmin_seg:
                segs.append((s_idx, e_idx))
            in_seg = False

    return segs


# --------------------------------------------------------------
#  SEGMENT μ CALCULATION
# --------------------------------------------------------------

def _estimate_mu_segments(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    mu_min: float,
    mu_max: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    За всеки сегмент:
    - регресия v(t) → a_seg
    - slope_seg = медиана(slope)
    - μ_seg = (sinθ − a_seg/g) / cosθ
    Филтрираме само по μ_min <= μ_seg <= μ_max.
    """
    mu_seg_label = pd.Series(np.nan, index=df.index)
    rec = []

    for idx, (start, end) in enumerate(segments):
        length = end - start + 1
        if length < 2:
            continue

        t = np.arange(length, float)
        v = df["v"].iloc[start : end + 1].values

        try:
            coeffs = np.polyfit(t, v, 1)
        except Exception:
            continue

        a_seg = coeffs[0]
        slope_seg = float(df["slope"].iloc[start : end + 1].median())

        theta = np.arctan(slope_seg)
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)

        if cos_t == 0:
            continue
        if not np.isfinite(a_seg) or not np.isfinite(slope_seg):
            continue

        mu_seg = (sin_t - a_seg / G) / cos_t

        if not np.isfinite(mu_seg):
            continue
        if mu_seg < mu_min or mu_seg > mu_max:
            continue

        mu_seg_label.iloc[start : end + 1] = mu_seg

        rec.append(
            {
                "seg_id": idx,
                "start": start,
                "end": end,
                "duration_s": length,
                "a_seg": float(a_seg),
                "slope_seg": slope_seg,
                "mu_seg": float(mu_seg),
            }
        )

    return mu_seg_label, pd.DataFrame(rec)


# --------------------------------------------------------------
#  MAIN PROCESS
# --------------------------------------------------------------

def process_activity_file(
    file_obj,
    filename: str,
    S_thr_percent: float = -2.0,
    v_min: float = 2.0,
    T_cut: int = 5,
    Tmin_free: int = 5,
    mu_min: float = 0.0,
    mu_max: float = 0.2,
) -> Dict[str, Any]:
    """
    Обработва цяла активност и връща:
    {
        "name": filename,
        "df": df (с всички колони),
        "segments": seg_df,
        "mu_session": медиана(μ_seg),
        "n_valid": обща продължителност на валидните сегменти (s),
        "FI": None,
        "K": None,
    }
    """
    # четем файла в buffer за повече от едно ползване
    bytes_data = file_obj.read()
    file_obj.seek(0)
    buffer = io.BytesIO(bytes_data)

    df_raw = load_activity(buffer, filename)
    df_raw = df_raw.sort_values("time_s").drop_duplicates("time_s").reset_index(
        drop=True
    )

    df = _resample_and_smooth(df_raw)
    df = _compute_kinematics(df)

    is_free = _detect_free_glide_mask(
        df,
        S_thr_percent=S_thr_percent,
        v_min=v_min,
        T_cut=T_cut,
        Tmin_free=Tmin_free,
    )
    segments = _extract_segments(is_free, Tmin_free)

    mu_seg_label, seg_df = _estimate_mu_segments(
        df,
        segments=segments,
        mu_min=mu_min,
        mu_max=mu_max,
    )

    df["is_free_glide"] = is_free
    df["mu_seg_label"] = mu_seg_label

    if seg_df.empty:
        raise ValueError("Няма валидни free-glide сегменти за μ (провери параметрите).")

    mu_session = float(seg_df["mu_seg"].median())
    n_valid = int(seg_df["duration_s"].sum())

    return {
        "name": filename,
        "df": df,
        "segments": seg_df,
        "mu_session": mu_session,
        "n_valid": n_valid,
        "FI": None,
        "K": None,
    }


# --------------------------------------------------------------
#  MODULATION
# --------------------------------------------------------------

def compute_friction_indices_and_modulation(
    activities: Dict[str, Dict[str, Any]],
    ref_name: str,
    delta_up: float,
    delta_down: float,
) -> None:
    """
    Изчислява Friction Index (FI) и модулатор K за всяка активност спрямо
    избраната референтна, след което добавя колоната v_mod в df.

    - FI = mu_session / mu_ref
    - K_raw = mu_session / mu_ref
      * mu_session > mu_ref  -> K_raw > 1 (по-тежки условия -> повишаваме скоростта)
      * mu_session < mu_ref  -> K_raw < 1 (по-бързи условия -> намаляваме скоростта)
    - K се ограничава в [1 - delta_down, 1 + delta_up].
    """
    if ref_name not in activities:
        raise ValueError("Референтната активност не е налична.")

    mu_ref = activities[ref_name]["mu_session"]

    for name, act in activities.items():
        mu_ses = act["mu_session"]

        FI = mu_ses / mu_ref
        K_raw = mu_ses / mu_ref

        K_min = 1.0 - delta_down
        K_max = 1.0 + delta_up
        K = max(K_min, min(K_raw, K_max))

        df = act["df"].copy()
        df["v_mod"] = df["v"] * K

        act["FI"] = float(FI)
        act["K"] = float(K)
        act["df"] = df
