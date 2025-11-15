
import math
import xml.etree.ElementTree as ET
from typing import Tuple, Dict

import numpy as np
import pandas as pd


G = 9.80665  # гравитация


def _smooth_series(x: pd.Series, win_med: int = 11, win_mean: int = 11) -> pd.Series:
    """
    Двойно изглаждане: медианен филтър + подвижна средна.
    Подобно на това, което виждаш в Garmin (хващаме тренда).
    """
    med = x.rolling(window=win_med, center=True, min_periods=1).median()
    smooth = med.rolling(window=win_mean, center=True, min_periods=1).mean()
    return smooth


def process_tcx_file(
    file_obj,
    mass: float = 72.0,
    cda: float = 0.35,
    rho: float = 1.2,
    min_down_grade: float = -0.05,
    min_segment_time: float = 15.0,
    cut_head_time: float = 5.0,
    min_remaining_time: float = 10.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Чете TCX, изчислява наклон, скорост, ускорения и ефективен μ.
    Връща:
      - DataFrame с колоните:
        time, t_sec, dist, alt_smooth, grade, speed_smooth,
        a_real, a_grav, a_air, a_fric, mu_eff, is_valid_down
      - summary dict с индексите по активност.
    """
    # --- парсване на TCX ---
    tree = ET.parse(file_obj)
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    rows = []
    for tp in root.findall(".//tcx:Trackpoint", ns):
        time_el = tp.find("tcx:Time", ns)
        alt_el = tp.find("tcx:AltitudeMeters", ns)
        dist_el = tp.find("tcx:DistanceMeters", ns)

        if time_el is None:
            continue

        t = pd.to_datetime(time_el.text)
        alt = float(alt_el.text) if alt_el is not None else np.nan
        dist = float(dist_el.text) if dist_el is not None else np.nan

        rows.append({"time": t, "alt": alt, "dist": dist})

    if not rows:
        raise ValueError("TCX файлът не съдържа Trackpoint елементи.")

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    # --- базови разлики ---
    df["dt"] = df["time"].diff().dt.total_seconds()
    # отстраняваме нулеви или отрицателни интервали
    df.loc[df["dt"] <= 0, "dt"] = np.nan

    df["ddist"] = df["dist"].diff()
    df["dalt"] = df["alt"].diff()

    # време от началото (секунди)
    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    # --- изглаждане на височината и скоростта ---
    df["alt_smooth"] = _smooth_series(df["alt"])

    df["dalt_s"] = df["alt_smooth"].diff()
    df["ddist_s"] = df["dist"].diff()

    # наклон (относителен, не в %)
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = df["dalt_s"] / df["ddist_s"]
    df["grade"] = grade.replace([np.inf, -np.inf], np.nan)

    # скорост (m/s)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = df["ddist_s"] / df["dt"]
    df["speed"] = speed.replace([np.inf, -np.inf], np.nan)
    df["speed_smooth"] = _smooth_series(df["speed"], win_med=5, win_mean=5)

    # --- ускорения ---
    df["dv"] = df["speed_smooth"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        a_real = df["dv"] / df["dt"]
    df["a_real"] = a_real.replace([np.inf, -np.inf], np.nan)

    # гравитация по наклона
    theta = np.arctan(df["grade"])
    df["theta"] = theta
    df["a_grav"] = -G * np.sin(theta)

    # въздушно съпротивление ~ v^2
    k_air = 0.5 * rho * cda / mass
    df["a_air"] = -k_air * (df["speed_smooth"] ** 2)

    # остатъчно "фрикционно" ускорение
    df["a_fric"] = df["a_real"] - df["a_grav"] - df["a_air"]

    # ефективен μ
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_eff = -df["a_fric"] / (G * np.cos(theta))
    df["mu_eff"] = mu_eff.replace([np.inf, -np.inf], np.nan)

    # --- намиране на сегменти за спускане ---
    df["is_down"] = df["grade"] <= min_down_grade

    segments = []
    current = []
    for idx, isd in enumerate(df["is_down"]):
        if bool(isd):
            current.append(idx)
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        segments.append(current)

    df["is_valid_down"] = False

    for seg in segments:
        seg_df = df.loc[seg]

        seg_dt = seg_df["dt"].fillna(0).sum()
        if seg_dt < min_segment_time:
            continue

        # кумулативно време в сегмента
        cum = seg_df["dt"].fillna(0).cumsum()
        total = cum.iloc[-1]
        if total < min_segment_time:
            continue

        # отрязваме първите cut_head_time секунди
        keep_mask = cum > cut_head_time
        remaining_time = total - cut_head_time
        if remaining_time < min_remaining_time:
            continue

        kept_indices = seg_df.index[keep_mask]
        df.loc[kept_indices, "is_valid_down"] = True

    # --- обобщение за валидните точки ---
    valid = df[df["is_valid_down"]].copy()

    if not valid.empty:
        median_mu_valid = float(valid["mu_eff"].median())
        mean_grade_valid = float(valid["grade"].mean())
        n_valid_down = int(valid.shape[0])
    else:
        median_mu_valid = float("nan")
        mean_grade_valid = float("nan")
        n_valid_down = 0

    summary = {
        "median_mu_valid": median_mu_valid,
        "mean_grade_valid": mean_grade_valid,
        "n_valid_down": n_valid_down,
    }

    return df, summary
