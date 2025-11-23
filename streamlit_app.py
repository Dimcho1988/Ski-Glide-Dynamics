import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import altair as alt
from datetime import datetime

# -----------------------
# НАСТРОЙКИ (КОНСТАНТИ)
# -----------------------

# Сегментация
SEGMENT_LENGTH_SEC = 5.0
MIN_SEG_POINTS = 2          # минимум точки в сегмент
MIN_SEG_DURATION_S = 1.0    # минимум реално време
MIN_SEG_DIST_M = 3.0        # минимум хоризонтална дистанция

# Филтри за явно нереалистични данни
MAX_ABS_SLOPE = 40.0        # режем сегменти с |slope| > 40%
MAX_SPEED_KMH = 70.0        # режем сегменти със скорост > 70 km/h

# Модел 1 – плъзгаемост
DOWNHILL_MIN_SLOPE = -15.0
DOWNHILL_MAX_SLOPE = -5.0
MIN_DOWNHILL_SEGMENTS = 10  # минимум сегменти за модел на плъзгаемостта
RATIO_TRIM_Q = 0.05         # trimming по V/|slope|

# Модел 2 – влияние на наклона
FLAT_REF_SLOPE_MAX = 1.0
SLOPE_MODEL_MIN = -3.0
SLOPE_MODEL_MAX = 10.0
MIN_SLOPE_SEGMENTS = 10

# Модел 3 – зони по критична скорост
DOWNHILL_RELAX_SLOPE = -5.0  # всичко под -5% -> Z1
ZONES = [
    (0.00, 0.80, "Z1"),
    (0.80, 0.90, "Z2"),
    (0.90, 1.00, "Z3"),
    (1.00, 1.10, "Z4"),
    (1.10, 1.20, "Z5"),
    (1.20, 99.0, "Z6"),
]

# -----------------------
# ОБЩИ ФУНКЦИИ
# -----------------------

def parse_tcx(file) -> pd.DataFrame:
    """
    Парсира TCX файл и връща:
    time, dist (m), alt (m), hr, elapsed_s
    """
    try:
        tree = ET.parse(file)
    except Exception as e:
        st.error(f"Грешка при парсване на TCX ({file.name}): {e}")
        return pd.DataFrame()

    root = tree.getroot()
    rows = []

    def get_child_with_tag(parent, tag_suffix):
        for child in parent:
            if child.tag.endswith(tag_suffix):
                return child
        return None

    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        t_el = get_child_with_tag(tp, "Time")
        alt_el = get_child_with_tag(tp, "AltitudeMeters")
        dist_el = get_child_with_tag(tp, "DistanceMeters")
        hr_parent = get_child_with_tag(tp, "HeartRateBpm")
        hr_el = get_child_with_tag(hr_parent, "Value") if hr_parent is not None else None

        if t_el is None or alt_el is None or dist_el is None:
            continue

        try:
            t = pd.to_datetime(t_el.text)
            alt = float(alt_el.text)
            dist = float(dist_el.text)
        except Exception:
            continue

        if hr_el is not None and hr_el.text is not None:
            try:
                hr = float(hr_el.text)
            except Exception:
                hr = np.nan
        else:
            hr = np.nan

        rows.append({"time": t, "alt": alt, "dist": dist, "hr": hr})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    5-секундни сегменти без припокриване, със сурови данни.
    Филтрите са само:
    - минимум точки, време, дистанция
    - режем |slope| > MAX_ABS_SLOPE
    - режем скорост > MAX_SPEED_KMH
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    df = df.copy()
    df["seg_id"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    seg_rows = []
    for seg_id, g in df.groupby("seg_id"):
        g = g.sort_values("time")
        if len(g) < MIN_SEG_POINTS:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration = (t_end - t_start).total_seconds()
        if duration < MIN_SEG_DURATION_S:
            continue

        alt_start = g["alt"].iloc[0]
        alt_end = g["alt"].iloc[-1]
        dist_start = g["dist"].iloc[0]
        dist_end = g["dist"].iloc[-1]
        horiz_dist = dist_end - dist_start

        if horiz_dist < MIN_SEG_DIST_M:
            continue

        delta_alt = alt_end - alt_start
        slope_percent = (delta_alt / horiz_dist) * 100.0
        speed_kmh = (horiz_dist / duration) * 3.6

        if abs(slope_percent) > MAX_ABS_SLOPE:
            continue
        if speed_kmh > MAX_SPEED_KMH:
            continue

        hr_mean = g["hr"].mean()

        seg_rows.append(
            {
                "seg_id": seg_id,
                "start_time": t_start,
                "end_time": t_end,
                "duration_s": duration,
                "alt_start": alt_start,
                "alt_end": alt_end,
                "dist_start": dist_start,
                "dist_end": dist_end,
                "horiz_dist_m": horiz_dist,
                "slope_percent": slope_percent,
                "speed_kmh": speed_kmh,
                "hr_mean": hr_mean,
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    return seg_df

# -----------------------
# МОДЕЛ 1 – ПЛЪЗГАЕМОСТ
# -----------------------

def filter_downhill_with_predecessor(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Downhill сегменти: -15% до -5% и предхождани от downhill.
    (минимални логически филтри, както в стария код)
    """
    if seg_df.empty:
        return seg_df

    base_mask = (
        (seg_df["slope_percent"] >= DOWNHILL_MIN_SLOPE)
        & (seg_df["slope_percent"] <= DOWNHILL_MAX_SLOPE)
    )
    seg_df = seg_df[base_mask].copy()
    if seg_df.empty:
        return seg_df

    seg_df = seg_df.sort_values("seg_id").reset_index(drop=True)
    slope_by_id = dict(zip(seg_df["seg_id"], seg_df["slope_percent"]))

    valid = []
    for _, row in seg_df.iterrows():
        sid = row["seg_id"]
        prev_slope = slope_by_id.get(sid - 1, None)
        if prev_slope is None:
            valid.append(False)
        else:
            valid.append(DOWNHILL_MIN_SLOPE <= prev_slope <= DOWNHILL_MAX_SLOPE)

    seg_df["valid"] = valid
    seg_df = seg_df[seg_df["valid"]].drop(columns=["valid"])
    return seg_df


def trim_by_speed_slope_ratio(all_segments: pd.DataFrame, q: float = RATIO_TRIM_Q):
    """Лек outlier trim по R = V/|slope|."""
    if all_segments.empty or len(all_segments) < MIN_DOWNHILL_SEGMENTS:
        return all_segments, None, None

    segs = all_segments.copy()
    segs["R"] = segs["speed_kmh"] / segs["slope_percent"].abs()
    r = segs["R"].values
    r_low = np.quantile(r, q)
    r_high = np.quantile(r, 1 - q)
    mask = (segs["R"] >= r_low) & (segs["R"] <= r_high)
    segs = segs[mask].copy()
    return segs, r_low, r_high


def fit_linear_speed_slope(all_segments: pd.DataFrame):
    """Линеен модел V = a*slope + b за downhill."""
    if all_segments.empty or len(all_segments) < 5:
        return None, None
    x = all_segments["slope_percent"].values
    y = all_segments["speed_kmh"].values
    try:
        a, b = np.polyfit(x, y, 1)
        return a, b
    except Exception:
        return None, None


def compute_glide_model(downhill_segments_list, raw_by_activity):
    """
    Модел за плъзгаемост:
    - общ линеен модел V = a*slope + b,
    - Glide index по активност,
    - модулирана средна скорост.
    """
    if not downhill_segments_list:
        return {}, None, None, pd.DataFrame(), pd.DataFrame()

    down = pd.concat(downhill_segments_list, ignore_index=True)

    down_trimmed, _, _ = trim_by_speed_slope_ratio(down)
    a, b = fit_linear_speed_slope(down_trimmed)

    if a is None:
        return {}, None, None, down_trimmed, pd.DataFrame()

    glide_index_by_activity = {}
    summary_rows = []

    for act_name, df_raw in raw_by_activity.items():
        if df_raw.empty:
            continue

        total_dist = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
        total_time = df_raw["elapsed_s"].iloc[-1] - df_raw["elapsed_s"].iloc[0]
        overall_speed = (total_dist / total_time) * 3.6 if total_time > 0 else np.nan

        seg_act = down_trimmed[down_trimmed["activity"] == act_name]
        if seg_act.empty:
            glide_index_by_activity[act_name] = np.nan
            summary_rows.append(
                {
                    "activity": act_name,
                    "n_segments": 0,
                    "mean_slope": np.nan,
                    "mean_speed": np.nan,
                    "model_speed_at_mean_slope": np.nan,
                    "glide_index": np.nan,
                    "overall_speed": overall_speed,
                    "modulated_overall_speed": np.nan,
                }
            )
            continue

        w = seg_act["duration_s"].values
        mean_slope = np.average(seg_act["slope_percent"].values, weights=w)
        mean_speed = np.average(seg_act["speed_kmh"].values, weights=w)
        model_speed = a * mean_slope + b

        if model_speed <= 0:
            glide_index = np.nan
            mod_overall = np.nan
        else:
            glide_index = mean_speed / model_speed
            mod_overall = overall_speed / glide_index

        glide_index_by_activity[act_name] = glide_index
        summary_rows.append(
            {
                "activity": act_name,
                "n_segments": int(len(seg_act)),
                "mean_slope": mean_slope,
                "mean_speed": mean_speed,
                "model_speed_at_mean_slope": model_speed,
                "glide_index": glide_index,
                "overall_speed": overall_speed,
                "modulated_overall_speed": mod_overall,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return glide_index_by_activity, a, b, down_trimmed, summary_df


def show_glide_view(a, b, down_trimmed, summary_df):
    st.subheader("Модел 1 – Плъзгаемост (Ski Glide)")

    if a is None or down_trimmed.empty:
        st.warning("Няма достатъчно downhill данни за модела.")
        return

    st.markdown(
        f"""
        **Линеен модел при спускане:**

        V = a · slope% + b  

        a = **{a:.3f} km/h на 1% наклон**  
        b = **{b:.3f} km/h при 0% наклон**
        """
    )

    down_trimmed = down_trimmed.copy()
    down_trimmed["v_model"] = a * down_trimmed["slope_percent"] + b
    down_trimmed["residual"] = down_trimmed["speed_kmh"] - down_trimmed["v_model"]

    x_min = down_trimmed["slope_percent"].min()
    x_max = down_trimmed["slope_percent"].max()
    x_line = np.linspace(x_min, x_max, 100)
    y_line = a * x_line + b
    line_df = pd.DataFrame({"slope_percent": x_line, "v_model": y_line})

    scatter = alt.Chart(down_trimmed).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("slope_percent", title="Наклон (%)"),
        y=alt.Y("speed_kmh", title="Скорост (km/h)"),
        color=alt.Color("activity", title="Активност"),
        tooltip=["activity", "slope_percent", "speed_kmh", "v_model", "residual"],
    )

    line = alt.Chart(line_df).mark_line().encode(
        x="slope_percent",
        y="v_model",
    )

    st.altair_chart(scatter + line, use_container_width=True)

    st.markdown("**Обобщение по активности (плъзгаемост)**")
    if not summary_df.empty:
        st.dataframe(
            summary_df.style.format(
                {
                    "mean_slope": "{:.2f}",
                    "mean_speed": "{:.2f}",
                    "model_speed_at_mean_slope": "{:.2f}",
                    "glide_index": "{:.3f}",
                    "overall_speed": "{:.2f}",
                    "modulated_overall_speed": "{:.2f}",
                }
            )
        )


# -----------------------
# МОДЕЛ 2 – НАКЛОН (ΔV%)
# -----------------------

def compute_glide_corrected_segments(all_segments: pd.DataFrame, glide_idx_by_activity):
    seg = all_segments.copy()
    ks = [glide_idx_by_activity.get(act, 1.0) for act in seg["activity"]]
    seg["glide_index"] = ks
    seg["speed_glide_kmh"] = np.where(
        (seg["glide_index"] > 0) & (~np.isnan(seg["glide_index"])),
        seg["speed_kmh"] / seg["glide_index"],
        seg["speed_kmh"],
    )
    return seg


def compute_slope_model(all_segments_glide: pd.DataFrame):
    seg = all_segments_glide.copy()
    if seg.empty:
        return None, None, pd.DataFrame()

    flat_mask = seg["slope_percent"].abs() <= FLAT_REF_SLOPE_MAX
    flat_segments = seg[flat_mask]
    if flat_segments.empty:
        return None, None, pd.DataFrame()

    v_flat = (flat_segments["speed_glide_kmh"] * flat_segments["duration_s"]).sum() / flat_segments["duration_s"].sum()

    slope_mask = (
        (seg["slope_percent"] > SLOPE_MODEL_MIN)
        & (seg["slope_percent"] < SLOPE_MODEL_MAX)
        & (~flat_mask)
        & (seg["slope_percent"].abs() <= MAX_ABS_SLOPE)
    )
    slope_df = seg[slope_mask].copy()
    if slope_df.empty or len(slope_df) < MIN_SLOPE_SEGMENTS:
        return v_flat, None, slope_df

    slope_df["delta_v_pct"] = 100.0 * (slope_df["speed_glide_kmh"] - v_flat) / v_flat

    x = slope_df["slope_percent"].values
    y = slope_df["delta_v_pct"].values
    try:
        coeffs = np.polyfit(x, y, 2)
    except Exception:
        coeffs = None

    return v_flat, coeffs, slope_df


def apply_slope_correction(all_segments_glide: pd.DataFrame, v_flat, coeffs):
    seg = all_segments_glide.copy()
    seg["slope_factor"] = 1.0

    if v_flat is not None and coeffs is not None:
        c2, c1, c0 = coeffs

        def f_slope(s):
            if abs(s) <= FLAT_REF_SLOPE_MAX:
                return 1.0
            if SLOPE_MODEL_MIN < s < SLOPE_MODEL_MAX:
                delta = c2 * s**2 + c1 * s + c0
                return 1.0 + delta / 100.0
            return 1.0

        seg["slope_factor"] = seg["slope_percent"].apply(f_slope)

    f = seg["slope_factor"].replace(0, 1.0)
    seg["speed_slope_kmh"] = seg["speed_kmh"] / f
    seg["speed_flat_kmh"] = seg["speed_glide_kmh"] / f
    return seg


def show_slope_view(v_flat, coeffs, all_corr: pd.DataFrame):
    st.subheader("Модел 2 – Влияние на наклона (ΔV%)")

    if v_flat is None or coeffs is None:
        st.warning("Недостатъчно данни за стабилен модел ΔV% = f(% наклон).")
        return

    c2, c1, c0 = coeffs

    st.write(
        f"Референтна скорост на равното (|наклон| ≤ {FLAT_REF_SLOPE_MAX:.1f}%): "
        f"**{v_flat:.2f} km/h**"
    )

    st.markdown(
        f"""
        ΔV% = c₂ · slope² + c₁ · slope + c₀  

        c₂ = **{c2:.4f}**, c₁ = **{c1:.4f}**, c₀ = **{c0:.4f}**
        """
    )

    flat_mask = all_corr["slope_percent"].abs() <= FLAT_REF_SLOPE_MAX
    slope_mask = (
        (all_corr["slope_percent"] > SLOPE_MODEL_MIN)
        & (all_corr["slope_percent"] < SLOPE_MODEL_MAX)
        & (~flat_mask)
    )
    slope_df = all_corr[slope_mask].copy()
    if slope_df.empty:
        st.warning("Няма достатъчно сегменти за визуализация.")
        return

    slope_df["delta_real_pct"] = 100.0 * (slope_df["speed_glide_kmh"] - v_flat) / v_flat
    slope_df["delta_model_pct"] = (
        c2 * slope_df["slope_percent"]**2 + c1 * slope_df["slope_percent"] + c0
    )

    x_min = slope_df["slope_percent"].min()
    x_max = slope_df["slope_percent"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = c2 * x_line**2 + c1 * x_line + c0
    line_df = pd.DataFrame({"slope_percent": x_line, "delta_model": y_line})

    scatter = alt.Chart(slope_df).mark_circle(size=40, opacity=0.5).encode(
        x=alt.X("slope_percent", title="Наклон (%)"),
        y=alt.Y("delta_real_pct", title="ΔV% (реално)"),
        color=alt.Color("activity", title="Активност"),
        tooltip=[
            "activity",
            "slope_percent",
            "speed_kmh",
            "speed_glide_kmh",
            "speed_flat_kmh",
            "delta_real_pct",
            "delta_model_pct",
        ],
    )

    line = alt.Chart(line_df).mark_line().encode(
        x="slope_percent",
        y="delta_model",
    )

    st.altair_chart(scatter + line, use_container_width=True)

    st.markdown("**Обобщение по активности (реална и модулирана скорост)**")
    per_act = []
    for act, g in slope_df.groupby("activity"):
        w = g["duration_s"].values
        per_act.append(
            {
                "activity": act,
                "n_segments": len(g),
                "mean_slope": np.average(g["slope_percent"], weights=w),
                "mean_speed_real": np.average(g["speed_kmh"], weights=w),
                "mean_speed_glide": np.average(g["speed_glide_kmh"], weights=w),
                "mean_speed_final": np.average(g["speed_flat_kmh"], weights=w),
            }
        )

    if per_act:
        df = pd.DataFrame(per_act)
        st.dataframe(
            df.style.format(
                {
                    "mean_slope": "{:.2f}",
                    "mean_speed_real": "{:.2f}",
                    "mean_speed_glide": "{:.2f}",
                    "mean_speed_final": "{:.2f}",
                }
            )
        )


# -----------------------
# МОДЕЛ 3 – ЗОНИ + PULS
# -----------------------

def assign_zone(speed_eff_kmh: float, slope_percent: float, cs: float) -> str:
    if cs <= 0 or np.isnan(speed_eff_kmh):
        return "NA"
    if slope_percent <= DOWNHILL_RELAX_SLOPE:
        ratio = ZONES[0][1]  # горна граница на Z1
    else:
        ratio = speed_eff_kmh / cs
    for lo, hi, name in ZONES:
        if lo <= ratio < hi:
            return name
    return ZONES[-1][2]


def show_zones_view(all_corr: pd.DataFrame):
    st.subheader("Модел 3 – Разпределение по зони + пулс")

    if all_corr.empty:
        st.warning("Няма валидни сегменти.")
        return

    cs = st.number_input(
        "Критична скорост (CS) [km/h]",
        min_value=0.0,
        value=20.0,
        step=0.5,
    )

    speed_min_zone = st.number_input(
        "Мин. скорост за включване в зони [km/h]",
        min_value=0.0,
        value=2.0,
        step=0.5,
        help="Сегменти под този праг (стрелба, почивка) не влизат в зоните.",
    )

    seg = all_corr.copy()
    seg = seg[seg["speed_kmh"] >= speed_min_zone].copy()
    if seg.empty:
        st.warning("След прага за скорост не останаха сегменти.")
        return

    seg["zone"] = seg.apply(
        lambda r: assign_zone(r["speed_flat_kmh"], r["slope_percent"], cs),
        axis=1,
    )

    total_time = seg["duration_s"].sum()
    rows = []
    for lo, hi, name in ZONES:
        g = seg[seg["zone"] == name]
        if g.empty:
            rows.append(
                {"zone": name, "time_min": 0.0, "time_pct": 0.0, "V_eff_mean": np.nan, "HR_mean": np.nan}
            )
        else:
            T = g["duration_s"].sum()
            time_min = T / 60.0
            time_pct = 100.0 * T / total_time
            V_eff = (g["speed_flat_kmh"] * g["duration_s"]).sum() / T
            HR = (g["hr_mean"] * g["duration_s"]).sum() / T if g["hr_mean"].notna().any() else np.nan
            rows.append(
                {"zone": name, "time_min": time_min, "time_pct": time_pct, "V_eff_mean": V_eff, "HR_mean": HR}
            )

    zone_df = pd.DataFrame(rows)
    st.dataframe(
        zone_df.style.format(
            {
                "time_min": "{:.1f}",
                "time_pct": "{:.1f}",
                "V_eff_mean": "{:.2f}",
                "HR_mean": "{:.0f}",
            }
        )
    )

# -----------------------
# MAIN APP
# -----------------------

def main():
    st.title("onFlows – Ski Glide + Slope + CS Zones (без агресивни филтри)")

    uploaded_files = st.file_uploader(
        "Качи един или повече TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Качи TCX файлове, за да започнем.")
        return

    mode = st.sidebar.radio(
        "Избери изглед:",
        ("Модел 1 – Плъзгаемост", "Модел 2 – Наклон", "Модел 3 – Зони"),
    )

    all_segments_list = []
    downhill_segments_list = []
    raw_by_activity = {}

    for f in uploaded_files:
        df_raw = parse_tcx(f)
        if df_raw.empty:
            continue
        raw_by_activity[f.name] = df_raw

        seg_df = build_segments(df_raw)
        if seg_df.empty:
            continue
        seg_df["activity"] = f.name
        all_segments_list.append(seg_df)

        seg_down = filter_downhill_with_predecessor(seg_df)
        if not seg_down.empty:
            seg_down["activity"] = f.name
            downhill_segments_list.append(seg_down)

    if not all_segments_list:
        st.error("Няма валидни сегменти.")
        return

    all_segments = pd.concat(all_segments_list, ignore_index=True)

    # Модел 1
    glide_idx_by_act, a, b, down_trimmed, glide_summary = compute_glide_model(
        downhill_segments_list, raw_by_activity
    )

    # Glide корекция
    all_glide = compute_glide_corrected_segments(all_segments, glide_idx_by_act)

    # Модел 2
    v_flat, coeffs, _ = compute_slope_model(all_glide)

    # Крайна корекция
    all_corr = apply_slope_correction(all_glide, v_flat, coeffs)

    if mode == "Модел 1 – Плъзгаемост":
        show_glide_view(a, b, down_trimmed, glide_summary)
    elif mode == "Модел 2 – Наклон":
        show_slope_view(v_flat, coeffs, all_corr)
    else:
        show_zones_view(all_corr)


if __name__ == "__main__":
    main()
