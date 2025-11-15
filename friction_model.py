"""
friction_model.py  (segment v4)

Имплементация на модела за коефициент на триене и модулиране на скоростта
с фиксирана референтна активност.

Новото в segment v4:
- Вместо да оценяваме μ_eff по секунди, работим по СЕГМЕНТИ:
  * намираме free-glide сегменти (както и преди);
  * за всеки сегмент правим линейна регресия v(t) -> a_seg;
  * изчисляваме един μ_seg за целия сегмент;
  * μ_session = медиана от всички валидни μ_seg.
- Премахваме дублирани времеви точки (time_s).
- Логика за модулация: K_raw = mu_session / mu_ref.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

G = 9.81  # гравитационно ускорение, m/s^2


# ---------------- ПАРСВАНЕ НА ФАЙЛОВЕ ---------------- #


def _load_tcx(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Минимален парсър за TCX:
    - Time (ISO 8601)
    - AltitudeMeters
    - DistanceMeters

    Връща DataFrame с колони:
    time_s (s от началото), h (m), d (m)
    """
    try:
        content = file_obj.read()
        file_obj.seek(0)
        root = ET.fromstring(content)
    except Exception as e:
        raise ValueError(f"Грешка при парсване на TCX ({filename}): {e}")

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}
    trackpoints = root.findall(".//tcx:Trackpoint", ns)

    times = []
    alts = []
    dists = []

    for tp in trackpoints:
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
        raise ValueError("Не бяха намерени валидни Trackpoint в TCX файла.")

    t0 = times[0]
    time_s = np.array([(t - t0).total_seconds() for t in times], dtype=float)

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "h": np.array(alts, dtype=float),
            "d": np.array(dists, dtype=float),
        }
    )

    return df


def _load_csv(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Очакван CSV формат:

    - Колона "time" (s) или "Time" / "seconds"
    - Колона "elevation" или "altitude" (m)
    - Колона "distance" или "dist" (m)

    Връща DataFrame с колони:
    time_s (s от началото), h (m), d (m)
    """
    try:
        df_raw = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Грешка при четене на CSV ({filename}): {e}")

    cols = {c.lower(): c for c in df_raw.columns}

    # време
    time_col = None
    for key in ["time", "seconds", "t"]:
        if key in cols:
            time_col = cols[key]
            break
    if time_col is None:
        raise ValueError("CSV трябва да съдържа колона 'time' или 'seconds' (s).")

    # височина
    elev_col = None
    for key in ["elevation", "altitude", "alt", "h"]:
        if key in cols:
            elev_col = cols[key]
            break
    if elev_col is None:
        raise ValueError("CSV трябва да съдържа колона 'elevation' или 'altitude' (m).")

    # дистанция
    dist_col = None
    for key in ["distance", "dist", "d"]:
        if key in cols:
            dist_col = cols[key]
            break
    if dist_col is None:
        raise ValueError("CSV трябва да съдържа колона 'distance' или 'dist' (m).")

    time_values = df_raw[time_col].values

    # Ако времето не е числово, опитваме да го парснем като datetime
    if not np.issubdtype(df_raw[time_col].dtype, np.number):
        times = pd.to_datetime(df_raw[time_col])
        t0 = times.iloc[0]
        time_s = (times - t0).dt.total_seconds().astype(float).values
    else:
        time_s = time_values.astype(float)
        # нормализираме така, че да започва от 0
        time_s = time_s - time_s[0]

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "h": df_raw[elev_col].astype(float).values,
            "d": df_raw[dist_col].astype(float).values,
        }
    )

    return df


def load_activity(file_obj: io.BytesIO, filename: str) -> pd.DataFrame:
    """
    Генеричен loader за TCX или CSV. Връща DataFrame с time_s, h, d.
    """
    name = filename.lower()
    if name.endswith(".tcx"):
        return _load_tcx(file_obj, filename)
    elif name.endswith(".csv"):
        return _load_csv(file_obj, filename)
    else:
        raise ValueError("Неподдържан формат. Използвай .tcx или .csv.")


# ---------------- ЯДРО НА МОДЕЛА ---------------- #


def _resample_and_smooth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ресемплиране до 1 Hz и изглаждане:
    - h_smooth: 5-секундно подвижно средно (centered)
    - d_smooth: 3-точково подвижно средно
    """
    df = df.sort_values("time_s").reset_index(drop=True)

    t_min = df["time_s"].min()
    t_max = df["time_s"].max()
    t_grid = np.arange(0, int(np.round(t_max - t_min)) + 1, 1.0, dtype=float)

    base = pd.DataFrame({"time_s": t_grid})
    base = base.set_index("time_s")

    df_interp = df.set_index("time_s")[["h", "d"]].reindex(
        base.index
    ).interpolate(method="linear", limit_direction="both")

    df_interp = df_interp.reset_index()

    # изглаждане
    df_interp["h_smooth"] = (
        df_interp["h"].rolling(window=5, center=True, min_periods=1).mean()
    )
    df_interp["d_smooth"] = (
        df_interp["d"].rolling(window=3, center=True, min_periods=1).mean()
    )

    return df_interp


def _compute_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Изчислява slope, скорост v и ускорение a (централна разлика).
    """
    df = df.copy()
    dt = 1.0

    # прирасти
    dh = df["h_smooth"].diff()
    dd = df["d_smooth"].diff()

    # slope = Δh / Δd
    slope = dh / dd.replace(0, np.nan)
    df["slope"] = slope

    # скорост по трасето
    v = dd / dt
    df["v"] = v

    # ускорение (централна разлика)
    v_forward = df["v"].shift(-1)
    v_backward = df["v"].shift(1)
    a = (v_forward - v_backward) / (2 * dt)
    df["a"] = a

    return df


def _detect_free_glide_mask(
    df: pd.DataFrame,
    S_thr_percent: float,
    v_min: float,
    T_cut: int,
    Tmin_free: int,
) -> pd.Series:
    """
    Детекция на "свободно плъзгане" (free glide) на ниво секунди.

    Връща булева Series is_free със същата дължина като df.

    Стъпки:
    - slope < S_thr (в %)
    - v > v_min
    - събираме последователни блокове (спускания)
    - режем първите T_cut секунди от всеки блок
    - запазваме само ако остатъкът е дълъг поне Tmin_free
    """
    df = df.copy()

    # S_thr е в %, slope е безразмерен => сравняваме със S_thr/100
    S_thr = S_thr_percent / 100.0

    is_desc = (df["slope"] < S_thr) & (df["v"] > v_min)

    is_free = pd.Series(False, index=df.index)

    in_block = False
    start_idx = None

    for i, val in enumerate(is_desc.values):
        if val and not in_block:
            in_block = True
            start_idx = i
        elif not val and in_block:
            end_idx = i - 1
            length = end_idx - start_idx + 1
            if length > T_cut + Tmin_free:
                real_start = start_idx + T_cut
                real_end = end_idx
                is_free.iloc[real_start : real_end + 1] = True
            in_block = False

    # ако свършва с блок
    if in_block:
        end_idx = len(is_desc) - 1
        length = end_idx - start_idx + 1
        if length > T_cut + Tmin_free:
            real_start = start_idx + T_cut
            real_end = end_idx
            is_free.iloc[real_start : real_end + 1] = True

    return is_free


def _extract_segments_from_mask(
    is_free: pd.Series,
    Tmin_seg: int,
) -> List[Tuple[int, int]]:
    """
    Връща списък от сегменти (start_idx, end_idx), където is_free == True
    и дължината на сегмента е поне Tmin_seg.
    """
    segments: List[Tuple[int, int]] = []

    in_seg = False
    s_idx = None

    for i, val in enumerate(is_free.values):
        if val and not in_seg:
            in_seg = True
            s_idx = i
        elif (not val or i == len(is_free) - 1) and in_seg:
            e_idx = i - 1 if not val else i
            length = e_idx - s_idx + 1
            if length >= Tmin_seg:
                segments.append((s_idx, e_idx))
            in_seg = False

    return segments


def _estimate_mu_per_segment(
    df: pd.DataFrame,
    segments: List[Tuple[int, int]],
    mu_min: float,
    mu_max: float,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    За всеки сегмент (start_idx, end_idx):
    - правим линейна регресия v(t) -> a_seg
    - взимаме медиана на slope в сегмента -> slope_seg
    - изчисляваме μ_seg по физичния модел
    - филтрираме:
        * a_seg трябва да е отрицателно (очакваме забавяне при триене)
        * mu_min <= μ_seg <= mu_max

    Връща:
    - Series mu_seg_label със същата дължина като df (μ_seg за секунда / NaN)
    - DataFrame segments_df с информация за сегментите
    """
    mu_seg_label = pd.Series(np.nan, index=df.index)

    seg_records = []

    for idx, (start, end) in enumerate(segments):
        length = end - start + 1
        if length < 2:
            continue

        # време в сегмента (0,1,2,...)
        t = np.arange(length, dtype=float)
        v = df["v"].iloc[start : end + 1].values

        # линейна регресия v(t) = a_seg * t + b
        # polyfit връща [slope, intercept]
        try:
            coeffs = np.polyfit(t, v, 1)
        except Exception:
            continue

        a_seg = coeffs[0]  # m/s^2
        # slope на сегмента (медиана)
        slope_seg = float(df["slope"].iloc[start : end + 1].median())

        # конвертираме в ъгъл
        theta = np.arctan(slope_seg)

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # очакваме a_seg да е отрицателно (намаляваща скорост)
        if not np.isfinite(a_seg) or a_seg >= 0:
            continue
        if not np.isfinite(slope_seg) or cos_theta == 0:
            continue

        # μ_seg = (sinθ - a/g) / cosθ
        mu_seg = (sin_theta - a_seg / G) / cos_theta

        if not np.isfinite(mu_seg):
            continue
        if mu_seg < mu_min or mu_seg > mu_max:
            continue

        # валиден сегмент -> попълваме
        mu_seg_label.iloc[start : end + 1] = mu_seg

        seg_records.append(
            {
                "seg_id": idx,
                "start_idx": start,
                "end_idx": end,
                "duration_s": length,
                "start_time_s": float(df["time_s"].iloc[start]),
                "end_time_s": float(df["time_s"].iloc[end]),
                "a_seg": float(a_seg),
                "slope_seg": float(slope_seg),
                "mu_seg": float(mu_seg),
            }
        )

    segments_df = pd.DataFrame(seg_records)

    return mu_seg_label, segments_df


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
    Високо ниво: взима качен файл, връща dict с:
    {
        "name": filename,
        "df": DataFrame със всички колони,
        "mu_session": float,
        "n_valid": int (обща продължителност на валидните free-glide сегменти),
        "segments": DataFrame с информация за сегментите,
        "FI": None (ще се изчисли по-късно),
        "K": None (ще се изчисли по-късно),
    }

    Стъпки:
    - зареждане (TCX/CSV) и унифициране (time_s, h, d);
    - премахване на дублирани time_s;
    - ресемплиране и изглаждане;
    - кинематика (slope, v, a);
    - маска за free glide;
    - сегменти от маската;
    - оценка на μ_seg за всеки сегмент и медиана μ_session.
    """
    # Нужно е да можем да четем файла повече от веднъж
    bytes_data = file_obj.read()
    file_obj.seek(0)
    buffer = io.BytesIO(bytes_data)

    df_raw = load_activity(buffer, filename)

    # фиксиране на дублирани времеви точки
    df_raw = df_raw.sort_values("time_s")
    df_raw = df_raw.drop_duplicates(subset=["time_s"], keep="first").reset_index(
        drop=True
    )

    df = _resample_and_smooth(df_raw)
    df = _compute_kinematics(df)

    # маска за free-glide секунди
    is_free = _detect_free_glide_mask(
        df,
        S_thr_percent=S_thr_percent,
        v_min=v_min,
        T_cut=T_cut,
        Tmin_free=Tmin_free,
    )

    # сегменти от маската (със същия праг за дължина)
    segments = _extract_segments_from_mask(
        is_free=is_free,
        Tmin_seg=Tmin_free,
    )

    mu_seg_label, segments_df = _estimate_mu_per_segment(
        df=df,
        segments=segments,
        mu_min=mu_min,
        mu_max=mu_max,
    )

    df["is_free_glide"] = is_free
    df["mu_seg_label"] = mu_seg_label

    if segments_df.empty:
        raise ValueError(
            "Няма валидни free-glide сегменти за изчисляване на μ (провери параметрите)."
        )

    # медиана на μ_seg по сегменти
    mu_session = float(segments_df["mu_seg"].median())
    n_valid_seconds = int(segments_df["duration_s"].sum())

    result = {
        "name": filename,
        "df": df,
        "mu_session": mu_session,
        "n_valid": n_valid_seconds,
        "segments": segments_df,
        "FI": None,
        "K": None,
    }

    return result


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
      * mu_session > mu_ref  -> K_raw > 1  (по-тежки условия -> повишаваме скоростта)
      * mu_session < mu_ref  -> K_raw < 1  (по-бързи условия -> намаляваме скоростта)
    - K се ограничава в [1 - delta_down, 1 + delta_up]
    """
    if ref_name not in activities:
        raise ValueError("Референтната активност не е налична.")

    mu_ref = activities[ref_name]["mu_session"]

    for name, act in activities.items():
        mu_session = act["mu_session"]

        # Friction Index
        FI = mu_session / mu_ref

        # теоретичен коефициент за модулация
        K_raw = mu_session / mu_ref

        # Ограничаване
        K_min = 1.0 - delta_down
        K_max = 1.0 + delta_up
        K = max(K_min, min(K_raw, K_max))

        df = act["df"].copy()
        df["v_mod"] = df["v"] * K

        act["FI"] = float(FI)
        act["K"] = float(K)
        act["df"] = df
