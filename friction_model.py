"""
friction_model.py  (v3)

Имплементация на модела за коефициент на триене и модулиране на скоростта
с фиксирана референтна активност.

Новото във v3:
- Премахване на дублирани времеви точки (time_s), за да няма
  грешка 'cannot reindex on an axis with duplicate labels'.
- Коригирана логика за модулация на скоростта:
  K_raw = mu_session / mu_ref
  (по-високо триене -> по-бавни условия -> скоростта се увеличава при мапване
   към референтни; по-ниско триене -> скоростта се намалява).
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from typing import Dict, Any

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


def _detect_free_glide(
    df: pd.DataFrame,
    S_thr_percent: float,
    v_min: float,
    T_cut: int,
    Tmin_free: int,
) -> pd.Series:
    """
    Детекция на "свободно плъзгане".

    Връща булева Series is_free_glide със същата дължина като df.
    """
    df = df.copy()

    # S_thr е в %, slope е безразмерен => сравняваме със S_thr/100
    S_thr = S_thr_percent / 100.0

    is_desc = (df["slope"] < S_thr) & (df["v"] > v_min)

    is_free = pd.Series(False, index=df.index)

    # намираме сегменти
    in_block = False
    start_idx = None

    for i, val in enumerate(is_desc.values):
        if val and not in_block:
            in_block = True
            start_idx = i
        elif not val and in_block:
            end_idx = i - 1
            # обработваме сегмента [start_idx, end_idx]
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


def _compute_mu_eff_for_free_glide(
    df: pd.DataFrame,
    is_free_glide: pd.Series,
    mu_min: float,
    mu_max: float,
) -> pd.Series:
    """
    Изчислява μ_eff само за секундите на свободно плъзгане.
    """
    df = df.copy()

    # ограничаваме slope за стабилност
    slope = df["slope"].clip(lower=-0.5, upper=0.0)

    theta = np.arctan(slope.values)
    a = df["a"].values

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # μ_eff = (sinθ - a/g) / cosθ
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_eff = (sin_theta - a / G) / cos_theta

    mu_eff_series = pd.Series(mu_eff, index=df.index)

    # филтрация
    mu_eff_series[~is_free_glide] = np.nan  # само в free_glide
    mu_eff_series[(mu_eff_series < mu_min) | (mu_eff_series > mu_max)] = np.nan

    return mu_eff_series


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
        "n_valid": int,
        "FI": None (ще се изчисли по-късно),
        "K": None (ще се изчисли по-късно),
    }

    В тази v3 версия са добавени:
    - сортиране по time_s
    - премахване на дублирани time_s (duplicate timestamps)
    - коригирана логика за K_raw = mu_session / mu_ref
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

    is_free = _detect_free_glide(
        df,
        S_thr_percent=S_thr_percent,
        v_min=v_min,
        T_cut=T_cut,
        Tmin_free=Tmin_free,
    )

    mu_eff = _compute_mu_eff_for_free_glide(
        df,
        is_free_glide=is_free,
        mu_min=mu_min,
        mu_max=mu_max,
    )

    df["is_free_glide"] = is_free
    df["mu_eff"] = mu_eff

    valid_mu = mu_eff.dropna()
    n_valid = int(valid_mu.shape[0])

    if n_valid == 0:
        raise ValueError(
            "Няма валидни секунди за изчисляване на μ_eff (провери параметрите)."
        )

    mu_session = float(valid_mu.median())

    result = {
        "name": filename,
        "df": df,
        "mu_session": mu_session,
        "n_valid": n_valid,
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
