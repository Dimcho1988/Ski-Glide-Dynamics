import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

G = 9.81  # m/s^2


def parse_tcx_to_df(uploaded_file) -> Optional[pd.DataFrame]:
    """Parse a TCX file (UploadedFile) into a DataFrame with time, altitude, distance.

    Returns
    -------
    df : pd.DataFrame with DateTimeIndex and columns ['altitude', 'distance']
    """
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

    # Search for Trackpoints
    times = []
    alts = []
    dists = []

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

    df = pd.DataFrame(
        {"time": times, "altitude": alts, "distance": dists}
    ).dropna(subset=["time"])

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"])
    df = df.set_index("time")

    # Reindex to 1 Hz
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1S")
    df = df.reindex(full_index)

    # Interpolate altitude and distance
    for col in ["altitude", "distance"]:
        df[col] = df[col].interpolate(method="time", limit_direction="both")

    # Distance should be non-decreasing
    df["distance"] = df["distance"].cummax()

    return df


def _compute_kinematics(
    df: pd.DataFrame,
    slope_thr: float,
    v_min: float,
    trend_window: int,
    slope_window: int = 11,
) -> pd.DataFrame:
    """Compute smoothed height, trend line, slope, velocity, acceleration."""
    out = df.copy()

    # Basic rolling smoothing
    out["h_smooth"] = out["altitude"].rolling(window=5, center=True, min_periods=1).mean()
    out["d_smooth"] = out["distance"].rolling(window=3, center=True, min_periods=1).mean()

    # Trend line on altitude (bigger window)
    W_trend = max(5, int(trend_window))
    if W_trend % 2 == 0:
        W_trend += 1  # make it odd for symmetry
    out["h_trend"] = out["h_smooth"].rolling(window=W_trend, center=True, min_periods=1).mean()

    # --- СКЛОН ВЪРХУ ПО-ДЪЛЪГ ПРОЗОРЕЦ (примерно 10–11 s) ---
    W_slope = max(3, int(slope_window))
    if W_slope % 2 == 0:
        W_slope += 1
    half = W_slope // 2

    dh = out["h_trend"].shift(-half) - out["h_trend"].shift(half)
    dd = out["d_smooth"].shift(-half) - out["d_smooth"].shift(half)

    # Guard against tiny dd (почти няма движение)
    small_dd = dd.abs() < 0.5  # 0.5 m за ~10 s = много бавна промяна
    dh[small_dd] = np.nan
    dd[small_dd] = np.nan

    slope = dh / dd  # [-] (rise/run)
    slope = slope.clip(-0.3, 0.3)  # -30% до +30% за здрав разум
    slope_pct = slope * 100.0

    out["slope"] = slope
    out["slope_pct"] = slope_pct

    # Velocity (m/s) – 1 s diff + леко изглаждане
    v = out["d_smooth"].diff()
    v.iloc[0] = 0.0
    v = v.rolling(window=3, center=True, min_periods=1).mean()
    out["v"] = v

    # Acceleration (m/s^2) – централна разлика + изглаждане
    a = (out["v"].shift(-1) - out["v"].shift(1)) / 2.0
    a.iloc[0] = 0.0
    a.iloc[-1] = 0.0
    a = a.rolling(window=3, center=True, min_periods=1).mean()
    out["a"] = a

    # Геометрични кандидати за спускане
    is_desc_geom = (out["slope_pct"] <= slope_thr) & (out["v"] > v_min)
    out["is_desc_raw"] = is_desc_geom

    return out


def _mark_free_glide_segments(
    df: pd.DataFrame,
    t_cut: int,
    t_min_free: int,
    a_max: float,
) -> pd.DataFrame:
    """Mark free-glide seconds based on consecutive candidate segments.

    Parameters
    ----------
    t_cut : int
        Seconds to cut from the beginning of each candidate segment.
    t_min_free : int
        Minimal remaining length (in seconds) to keep a segment.
    a_max : float
        Maximal absolute acceleration (m/s^2) to be considered "free glide".
    """
    out = df.copy()

    # Кандидатите са тези със спускане + ниско ускорение (почти без оттласкване)
    is_desc = out["is_desc_raw"].fillna(False).astype(bool)
    low_acc = out["a"].abs() <= a_max
    is_candidate = is_desc & low_acc

    out["free_glide"] = False

    if not is_candidate.any():
        return out

    # Find segments of consecutive True values
    is_shift = is_candidate.shift(1, fill_value=False)
    segment_start = (~is_shift & is_candidate)
    segment_end = (is_shift & ~is_candidate)

    starts = list(out.index[segment_start])
    ends = list(out.index[segment_end])

    # If candidate extends to the last row
    if is_candidate.iloc[-1]:
        ends.append(out.index[-1])

    if len(ends) < len(starts):
        ends.append(out.index[-1])

    for s, e in zip(starts, ends):
        segment_idx = out.loc[s:e].index

        # Cut first t_cut seconds
        if len(segment_idx) <= t_cut:
            continue
        remaining = segment_idx[t_cut:]

        # Require at least t_min_free seconds after cutting
        if len(remaining) < t_min_free:
            continue

        out.loc[remaining, "free_glide"] = True

    return out


def compute_session_friction(
    df: pd.DataFrame,
    slope_thr: float = -5.0,
    v_min: float = 2.0,
    trend_window: int = 31,
    t_cut: int = 5,
    t_min_free: int = 10,
    a_max: float = 0.4,
    mu_min: float = 0.0,
    mu_max: float = 0.2,
    **kwargs,
) -> Dict[str, Any]:
    """Compute μ_session for a given activity.

    Returns
    -------
    result : dict
        {
            "df": DataFrame with additional columns,
            "mu_session": float or np.nan,
            "valid_seconds": int
        }
    """
    out = _compute_kinematics(
        df,
        slope_thr=slope_thr,
        v_min=v_min,
        trend_window=trend_window,
        slope_window=11,  # ~10 s прозорец за наклон
    )
    out = _mark_free_glide_segments(
        out,
        t_cut=t_cut,
        t_min_free=t_min_free,
        a_max=a_max,
    )

    free = out["free_glide"].fillna(False)
    idx_free = out.index[free]

    mu_vals = []
    out["mu_point"] = np.nan  # ще попълваме само за валидните точки

    for t in idx_free:
        row = out.loc[t]
        a = row["a"]
        slope = row["slope"]  # dimensionless (tan(theta))

        if not np.isfinite(a) or not np.isfinite(slope):
            continue

        # Convert slope to angle
        theta = np.arctan(slope)

        cos_theta = np.cos(theta)
        if np.isclose(cos_theta, 0.0):
            continue

        # a = g (sin(theta) - μ cos(theta))  ->  μ = (g sin(theta) - a) / (g cos(theta))
        mu = (G * np.sin(theta) - a) / (G * cos_theta)

        if not np.isfinite(mu):
            continue

        # Physical filter
        if mu <= mu_min or mu > mu_max:
            continue

        mu_vals.append(mu)
        out.at[t, "mu_point"] = mu

    mu_session = float(np.median(mu_vals)) if mu_vals else np.nan
    valid_seconds = len(mu_vals)

    return {
        "df": out,
        "mu_session": mu_session,
        "valid_seconds": valid_seconds,
    }


def apply_reference_and_modulation(
    activity: Dict[str, Any],
    mu_ref: float,
    delta_max_up: float = 0.20,
    delta_max_down: float = 0.15,
) -> None:
    """Given an activity dict (with 'df' and 'mu_session') and μ_ref,
    compute FI and K, and add modulated speed to df as 'v_mod'.
    """
    mu_sess = activity.get("mu_session", np.nan)
    df = activity["df"]

    if not np.isfinite(mu_ref) or not np.isfinite(mu_sess) or mu_ref <= 0:
        activity["FI"] = np.nan
        activity["K"] = np.nan
        df["v_mod"] = np.nan
        return

    FI = mu_sess / mu_ref
    # Raw scaling factor for speed: >1 → по-бърза скорост отчитаме, <1 → по-бавна
    K_raw = FI

    # Ограничаваме спрямо референтната скорост
    delta = K_raw - 1.0
    if delta > delta_max_up:
        delta = delta_max_up
    if delta < -delta_max_down:
        delta = -delta_max_down

    K = 1.0 + delta

    df["v_mod"] = df["v"] * K

    activity["FI"] = float(FI)
    activity["K"] = float(K)
