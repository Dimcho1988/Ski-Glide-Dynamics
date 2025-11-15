import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

G = 9.81  # m/s^2


def parse_tcx_to_df(uploaded_file) -> Optional[pd.DataFrame]:
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

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="1S")
    df = df.reindex(full_index)

    df["altitude"] = df["altitude"].interpolate(method="time", limit_direction="both")
    df["distance"] = df["distance"].interpolate(method="time", limit_direction="both")
    df["distance"] = df["distance"].cummax()

    return df


def _compute_kinematics(df, slope_thr, v_min, trend_window, slope_window=11):
    out = df.copy()

    out["h_smooth"] = out["altitude"].rolling(5, center=True, min_periods=1).mean()
    out["d_smooth"] = out["distance"].rolling(3, center=True, min_periods=1).mean()

    Wt = max(5, int(trend_window))
    if Wt % 2 == 0:
        Wt += 1
    out["h_trend"] = out["h_smooth"].rolling(Wt, center=True, min_periods=1).mean()

    Ws = max(3, int(slope_window))
    if Ws % 2 == 0:
        Ws += 1
    half = Ws // 2

    dh = out["h_trend"].shift(-half) - out["h_trend"].shift(half)
    dd = out["d_smooth"].shift(-half) - out["d_smooth"].shift(half)

    mask = dd.abs() < 0.5
    dh[mask] = np.nan
    dd[mask] = np.nan

    slope = (dh / dd).clip(-0.3, 0.3)
    out["slope"] = slope
    out["slope_pct"] = slope * 100

    v = out["d_smooth"].diff().fillna(0)
    v = v.rolling(3, center=True, min_periods=1).mean()
    out["v"] = v

    a = (out["v"].shift(-1) - out["v"].shift(1)) / 2
    a = a.fillna(0).rolling(3, center=True, min_periods=1).mean()
    out["a"] = a

    out["is_desc_raw"] = (out["slope_pct"] <= slope_thr) & (out["v"] > v_min)

    return out


def _mark_free_glide_segments(df, t_cut, t_min_free, a_max):
    out = df.copy()

    is_desc = out["is_desc_raw"].fillna(False)
    low_acc = (out["a"].abs() <= a_max)

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
    df,
    slope_thr=-5.0,
    v_min=2.0,
    trend_window=31,
    t_cut=5,
    t_min_free=10,
    a_max=0.4,
    mu_min=0.0,
    mu_max=0.2,
    **kwargs
):
    out = _compute_kinematics(df, slope_thr, v_min, trend_window, slope_window=11)
    out = _mark_free_glide_segments(out, t_cut, t_min_free, a_max)

    free = out["free_glide"].fillna(False)
    idx = out.index[free]

    out["mu_point"] = np.nan
    mu_vals = []

    for t in idx:
        row = out.loc[t]
        a = row["a"]
        slope = row["slope"]

        if not np.isfinite(a) or not np.isfinite(slope):
            continue

        theta = np.arctan(slope)
        cos_t = np.cos(theta)
        if np.isclose(cos_t, 0):
            continue

        mu = (G * np.sin(theta) - a) / (G * cos_t)

        if mu <= mu_min or mu > mu_max or not np.isfinite(mu):
            continue

        mu_vals.append(mu)
        out.at[t, "mu_point"] = mu

    mu_session = float(np.median(mu_vals)) if mu_vals else np.nan

    return {
        "df": out,
        "mu_session": mu_session,
        "valid_seconds": len(mu_vals),
    }


def apply_reference_and_modulation(activity, mu_ref, delta_max_up=0.20, delta_max_down=0.15):
    df = activity["df"]
    mu_sess = activity["mu_session"]

    if not np.isfinite(mu_ref) or not np.isfinite(mu_sess) or mu_ref <= 0:
        activity["FI"] = np.nan
        activity["K"] = np.nan
        df["v_mod"] = np.nan
        return

    FI = mu_sess / mu_ref
    K_raw = FI

    delta = K_raw - 1
    delta = np.clip(delta, -delta_max_down, delta_max_up)

    K = 1 + delta
    df["v_mod"] = df["v"] * K

    activity["FI"] = float(FI)
    activity["K"] = float(K)
