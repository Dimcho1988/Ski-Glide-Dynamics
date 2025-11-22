import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import altair as alt

# -----------------------
# НАСТРОЙКИ (КОНСТАНТИ)
# -----------------------

# Обща сегментация
SEGMENT_LENGTH_SEC = 5.0       # 5-секундни сегменти
MIN_SEGMENT_DISTANCE_M = 5.0   # мин. хоризонтална дистанция в сегмент

# Модел 1: Ski Glide Dynamics – V = f(% наклон) (само спускания)
DOWNHILL_MIN_SLOPE = -15.0     # минимален наклон за сегментите в модела
DOWNHILL_MAX_SLOPE = -5.0      # максимален наклон за сегментите в модела
RATIO_TRIM_Q = 0.05            # 5% отрязване на екстремни V/|slope|

# Модел 2: Влияние на наклона при еднакво усилие (ΔV%)
FLAT_REF_SLOPE_MAX = 1.0       # |slope| <= 1% = "равен терен"
SLOPE_MODEL_MIN = -3.0         # долна граница за наклон в модела ΔV%
SLOPE_MODEL_MAX = 10.0         # горна граница за наклон в модела ΔV%
GLOBAL_SLOPE_LIMIT = 30.0      # режем екстреми |slope| > 30%

# Модел 3: Зони спрямо критична скорост (CS)
DOWNHILL_RELAX_SLOPE = -3.0    # slope <= -3% -> натоварване в горна граница на Z1

# Зони като кратни на критичната скорост (V / CS)
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
    Парсва TCX файл и връща DataFrame:
    time, alt (m), dist (m), elapsed_s (sec)
    """
    try:
        tree = ET.parse(file)
    except Exception as e:
        st.error(f"Грешка при парсване на TCX ({file.name}): {e}")
        return pd.DataFrame()

    root = tree.getroot()
    records = []

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

        if t_el is None or alt_el is None or dist_el is None:
            continue

        try:
            t = pd.to_datetime(t_el.text)
            alt = float(alt_el.text)
            dist = float(dist_el.text)
        except Exception:
            continue

        records.append({"time": t, "alt": alt, "dist": dist})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("time").reset_index(drop=True)
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разделя активността на 5-секундни сегменти (последователни, без припокриване)
    и изчислява среден наклон и скорост за всеки сегмент.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    df = df.copy()
    df["seg_id"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    seg_rows = []
    for seg_id, g in df.groupby("seg_id"):
        g = g.sort_values("time")
        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration = (t_end - t_start).total_seconds()
        if duration <= 0:
            continue

        alt_start = g["alt"].iloc[0]
        alt_end = g["alt"].iloc[-1]
        dist_start = g["dist"].iloc[0]
        dist_end = g["dist"].iloc[-1]
        horiz_dist = dist_end - dist_start  # m

        if horiz_dist < MIN_SEGMENT_DISTANCE_M:
            continue

        delta_alt = alt_end - alt_start
        slope_percent = (delta_alt / horiz_dist) * 100.0  # %
        speed_kmh = (horiz_dist / duration) * 3.6         # km/h

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
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    return seg_df


# -----------------------
# МОДЕЛ 1: SKI GLIDE – DOWNHILL
# -----------------------

def filter_downhill_with_predecessor(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    За Ski Glide модела:
    1) Сегменти с наклон между DOWNHILL_MIN_SLOPE и DOWNHILL_MAX_SLOPE.
    2) Всеки сегмент трябва да е предхождан от сегмент със същия диапазон.
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

    valid_mask = []
    for _, row in seg_df.iterrows():
        sid = row["seg_id"]
        prev_id = sid - 1
        prev_slope = slope_by_id.get(prev_id, None)

        if prev_slope is None:
            valid_mask.append(False)
            continue

        if DOWNHILL_MIN_SLOPE <= prev_slope <= DOWNHILL_MAX_SLOPE:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    seg_df["valid"] = valid_mask
    seg_df = seg_df[seg_df["valid"]].drop(columns=["valid"])
    return seg_df


def trim_by_speed_slope_ratio(all_segments: pd.DataFrame, q: float = RATIO_TRIM_Q):
    """
    За downhill сегменти: режем екстремни R = V / |slope|.
    Оставяме сегментите между q и (1-q) персентил.
    """
    if all_segments.empty or len(all_segments) < 10:
        return all_segments, None, None

    segs = all_segments.copy()
    segs["speed_per_abs_slope"] = segs["speed_kmh"] / segs["slope_percent"].abs()

    r = segs["speed_per_abs_slope"].values
    r_low = np.quantile(r, q)
    r_high = np.quantile(r, 1 - q)

    mask = (segs["speed_per_abs_slope"] >= r_low) & (segs["speed_per_abs_slope"] <= r_high)
    segs = segs[mask].copy()
    return segs, r_low, r_high


def fit_linear_speed_slope(all_segments: pd.DataFrame):
    """Линейна регресия V = a * slope_percent + b."""
    if all_segments.empty or len(all_segments) < 5:
        return None, None

    x = all_segments["slope_percent"].values
    y = all_segments["speed_kmh"].values
    try:
        a, b = np.polyfit(x, y, 1)
        return a, b
    except Exception:
        return None, None


def show_ski_glide_model(downhill_segments_list, raw_by_activity):
    """Показва модел 1: Ski Glide Dynamics + обобщителна таблица по активности."""
    st.subheader("Модел 1: Ski Glide Dynamics – V = f(% наклон) (спускане)")

    if not downhill_segments_list:
        st.warning("Няма достатъчно downhill сегменти (−15% до −5%) за модела.")
        return {}, None, None

    down = pd.concat(downhill_segments_list, ignore_index=True)

    # Филтър по V/|slope|
    down_trimmed, r_low, r_high = trim_by_speed_slope_ratio(down)

    if r_low is not None:
        st.write(
            f"Downhill сегменти преди филтъра: {len(down)}, "
            f"след филтъра: {len(down_trimmed)}"
        )
        st.write(
            f"Допуснат диапазон V/|slope|: {r_low:.2f} – {r_high:.2f} km/h на 1% наклон."
        )

    # Линеен модел
    a, b = fit_linear_speed_slope(down_trimmed)
    if a is None:
        st.error("Не успях да построя линейния модел V = a*slope + b.")
        return {}, None, None

    st.markdown(
        f"""
        **Линеен модел за скорост при спускане:**

        V = a · slope% + b  

        a = **{a:.3f} km/h на 1% наклон**  
        b = **{b:.3f} km/h при 0% наклон**
        """
    )

    # Резидуали = индикатор за плъзгаемост
    down_trimmed = down_trimmed.copy()
    down_trimmed["v_model"] = a * down_trimmed["slope_percent"] + b
    down_trimmed["residual"] = down_trimmed["speed_kmh"] - down_trimmed["v_model"]
    down_trimmed["glide_label"] = np.where(
        down_trimmed["residual"] >= 0,
        "По-добра плъзгаемост (V > модел)",
        "По-лоша плъзгаемост (V < модел)",
    )

    st.markdown("**Разпределение на скоростта спрямо наклона (downhill сегменти)**")

    x_min = down_trimmed["slope_percent"].min()
    x_max = down_trimmed["slope_percent"].max()
    x_line = np.linspace(x_min, x_max, 100)
    y_line = a * x_line + b
    line_df = pd.DataFrame({"slope_percent": x_line, "v_model": y_line})

    scatter = alt.Chart(down_trimmed).mark_circle(size=40, opacity=0.6).encode(
        x=alt.X("slope_percent", title="Наклон (%)"),
        y=alt.Y("speed_kmh", title="Скорост (km/h)"),
        color=alt.Color("glide_label", title="Плъзгаемост"),
        tooltip=["activity", "slope_percent", "speed_kmh", "v_model", "residual"],
    )

    line = alt.Chart(line_df).mark_line().encode(
        x="slope_percent",
        y="v_model",
    )

    st.altair_chart(scatter + line, use_container_width=True)

    # Обобщителна таблица по активности
    st.markdown("**Обобщение по активности (плъзгаемост)**")

    per_act = []
    glide_index_by_activity = {}

    for act_name, df_raw in raw_by_activity.items():
        # обща средна скорост на активността
        if df_raw.empty:
            continue

        total_dist = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
        total_time = df_raw["elapsed_s"].iloc[-1] - df_raw["elapsed_s"].iloc[0]
        overall_speed = (total_dist / total_time) * 3.6 if total_time > 0 else np.nan

        # downhill сегментите след всички филтри за тази активност
        seg_act = down_trimmed[down_trimmed["activity"] == act_name]
        if seg_act.empty:
            per_act.append(
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
            glide_index_by_activity[act_name] = np.nan
            continue

        mean_slope = seg_act["slope_percent"].mean()
        mean_speed = np.average(seg_act["speed_kmh"], weights=seg_act["duration_s"])
        model_speed = a * mean_slope + b

        if model_speed <= 0:
            glide_index = np.nan
            mod_overall = np.nan
        else:
            glide_index = mean_speed / model_speed
            mod_overall = overall_speed / glide_index if glide_index not in (0, np.nan) else np.nan

        per_act.append(
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
        glide_index_by_activity[act_name] = glide_index

    if per_act:
        summary_df = pd.DataFrame(per_act)
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

    return glide_index_by_activity, a, b


# -----------------------
# МОДЕЛ 2: ΔV% СПРЯМО РАВНОТО
# -----------------------

def fit_delta_v_model(slope: np.ndarray, delta_v_pct: np.ndarray):
    """Квадратична регресия ΔV% = c2*s^2 + c1*s + c0."""
    if len(slope) < 5:
        return None
    try:
        coeffs = np.polyfit(slope, delta_v_pct, 2)
        return coeffs  # [c2, c1, c0]
    except Exception:
        return None


def compute_flat_speed_and_delta_model(all_segments: pd.DataFrame):
    """
    Обща функция:
    - намира V_flat (|slope| <= 1%)
    - строи ΔV% модел за сегментите -3% < slope < 10%
    Връща: v_flat, coeffs, slope_df (с реални ΔV%).
    """
    if all_segments.empty:
        return None, None, pd.DataFrame()

    flat_mask = all_segments["slope_percent"].abs() <= FLAT_REF_SLOPE_MAX
    flat_segments = all_segments[flat_mask]

    if flat_segments.empty:
        return None, None, pd.DataFrame()

    v_flat = flat_segments["speed_kmh"].mean()

    slope_mask = (
        (all_segments["slope_percent"] > SLOPE_MODEL_MIN) &
        (all_segments["slope_percent"] < SLOPE_MODEL_MAX) &
        (~flat_mask) &
        (all_segments["slope_percent"].abs() <= GLOBAL_SLOPE_LIMIT)
    )
    slope_df = all_segments[slope_mask].copy()

    if slope_df.empty or len(slope_df) < 10:
        return v_flat, None, slope_df

    slope_df["delta_v_pct"] = 100.0 * (slope_df["speed_kmh"] - v_flat) / v_flat
    coeffs = fit_delta_v_model(
        slope_df["slope_percent"].values,
        slope_df["delta_v_pct"].values,
    )
    return v_flat, coeffs, slope_df


def show_slope_effect_model(all_segments: pd.DataFrame):
    """Показва модел 2: влияние на наклона при еднакво усилие (ΔV%) + обобщителна таблица."""
    st.subheader("Модел 2: Влияние на наклона върху скоростта при еднакво усилие")

    v_flat, coeffs, slope_df = compute_flat_speed_and_delta_model(all_segments)

    if v_flat is None:
        st.warning("Няма сегменти с |наклон| ≤ 1% за определяне на V_flat.")
        return

    st.write(
        f"Референтна скорост на равното (|наклон| ≤ {FLAT_REF_SLOPE_MAX:.1f}%): "
        f"**{v_flat:.2f} km/h**"
    )

    if coeffs is None or slope_df.empty:
        st.warning("Недостатъчно данни за стабилен модел ΔV% = f(% наклон).")
        return

    c2, c1, c0 = coeffs

    st.markdown(
        f"""
        **Модел за влияние на наклона (при еднакво усилие):**

        ΔV% = c₂ · slope² + c₁ · slope + c₀  

        c₂ = **{c2:.4f}**  
        c₁ = **{c1:.4f}**  
        c₀ = **{c0:.4f}**
        """
    )

    # Визуализация
    x_min = slope_df["slope_percent"].min()
    x_max = slope_df["slope_percent"].max()
    x_line = np.linspace(x_min, x_max, 200)
    y_line = c2 * x_line**2 + c1 * x_line + c0
    line_df = pd.DataFrame({"slope_percent": x_line, "delta_model": y_line})

    scatter = alt.Chart(slope_df).mark_circle(size=40, opacity=0.5).encode(
        x=alt.X("slope_percent", title="Наклон (%)"),
        y=alt.Y("delta_v_pct", title="ΔV% спрямо равното"),
        color=alt.Color("activity", title="Активност"),
        tooltip=["activity", "slope_percent", "speed_kmh", "delta_v_pct"],
    )

    line = alt.Chart(line_df).mark_line().encode(
        x="slope_percent",
        y="delta_model",
    )

    st.altair_chart(scatter + line, use_container_width=True)

    # Обобщителна таблица по активности
    st.markdown("**Обобщение по активности (влияние на наклона)**")

    per_act = []
    for act_name, act_df in slope_df.groupby("activity"):
        n_seg = len(act_df)
        mean_slope = np.average(act_df["slope_percent"], weights=act_df["duration_s"])
        mean_speed = np.average(act_df["speed_kmh"], weights=act_df["duration_s"])
        mean_delta_real = np.average(act_df["delta_v_pct"], weights=act_df["duration_s"])

        # моделна ΔV% за всеки сегмент
        delta_model_each = c2 * act_df["slope_percent"]**2 + c1 * act_df["slope_percent"] + c0
        mean_delta_model = np.average(delta_model_each, weights=act_df["duration_s"])

        per_act.append(
            {
                "activity": act_name,
                "n_segments": n_seg,
                "mean_slope": mean_slope,
                "mean_speed": mean_speed,
                "mean_delta_real_pct": mean_delta_real,
                "mean_delta_model_pct": mean_delta_model,
                "mean_residual_pct": mean_delta_real - mean_delta_model,
            }
        )

    if per_act:
        summary_df = pd.DataFrame(per_act)
        st.dataframe(
            summary_df.style.format(
                {
                    "mean_slope": "{:.2f}",
                    "mean_speed": "{:.2f}",
                    "mean_delta_real_pct": "{:.1f}",
                    "mean_delta_model_pct": "{:.1f}",
                    "mean_residual_pct": "{:.1f}",
                }
            )
        )


# -----------------------
# МОДЕЛ 3: CS ЗОНИ (с модулирана скорост по наклон)
# -----------------------

def assign_zone(speed_eff_kmh: float, slope_percent: float, cs: float) -> str:
    """
    Връща име на зона според критичната скорост (CS).
    - ако slope <= -3% -> натоварване в горната граница на Z1;
    - иначе – според ratio = speed_eff / CS.
    """
    if cs <= 0 or np.isnan(speed_eff_kmh):
        return "NA"

    if slope_percent <= DOWNHILL_RELAX_SLOPE:
        ratio = ZONES[0][1]  # горна граница на Z1
    else:
        ratio = speed_eff_kmh / cs

    for low, high, name in ZONES:
        if low <= ratio < high:
            return name
    return ZONES[-1][2]


def show_cs_zones_model(all_segments: pd.DataFrame):
    """
    Показва модел 3: разпределение по зони спрямо критична скорост.
    Скоростта първо се модулира спрямо наклона (ΔV% модел),
    след това се ползва за зоновете.
    """
    st.subheader("Модел 3: Разпределение по зони спрямо критична скорост (CS)")

    if all_segments.empty:
        st.warning("Няма валидни сегменти за анализ.")
        return

    # 1) Модулираме скоростта по наклон (използваме същия модел като в Модел 2)
    v_flat, coeffs, slope_df = compute_flat_speed_and_delta_model(all_segments)

    seg_eff = all_segments.copy()
    seg_eff["speed_mod_kmh"] = seg_eff["speed_kmh"]  # по подразбиране – без промяна

    if v_flat is not None and coeffs is not None:
        c2, c1, c0 = coeffs
        for idx, row in seg_eff.iterrows():
            s = row["slope_percent"]
            v = row["speed_kmh"]

            # около равното – почти няма нужда от корекция
            if abs(s) <= FLAT_REF_SLOPE_MAX:
                seg_eff.at[idx, "speed_mod_kmh"] = v  # или v_flat – предпочитам v
            # в диапазона на модела – корекция по ΔV%
            elif SLOPE_MODEL_MIN < s < SLOPE_MODEL_MAX:
                delta_model = c2 * s**2 + c1 * s + c0
                factor = 1.0 + delta_model / 100.0
                if factor != 0:
                    seg_eff.at[idx, "speed_mod_kmh"] = v / factor
            # извън диапазона – оставяме реалната скорост

    # 2) Критична скорост и зони
    cs = st.number_input(
        "Въведи критична скорост (km/h)",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )

    if cs <= 0:
        st.info("Моля, въведи критична скорост, за да видиш зоновете.")
        return

    seg_eff["zone"] = seg_eff.apply(
        lambda row: assign_zone(row["speed_mod_kmh"], row["slope_percent"], cs),
        axis=1,
    )

    total_time = seg_eff["duration_s"].sum()
    if total_time <= 0:
        st.warning("Общото време на сегментите е нула – не мога да направя разпределение.")
        return

    # Обобщение за всички активности заедно
    zone_stats = []
    for low, high, name in ZONES:
        z = seg_eff[seg_eff["zone"] == name]
        if z.empty:
            zone_stats.append(
                {
                    "zone": name,
                    "time_min": 0.0,
                    "time_pct": 0.0,
                    "mean_speed_mod_kmh": np.nan,
                }
            )
        else:
            time_s = z["duration_s"].sum()
            time_min = time_s / 60.0
            time_pct = 100.0 * time_s / total_time
            mean_speed = np.average(z["speed_mod_kmh"], weights=z["duration_s"])
            zone_stats.append(
                {
                    "zone": name,
                    "time_min": time_min,
                    "time_pct": time_pct,
                    "mean_speed_mod_kmh": mean_speed,
                }
            )

    st.markdown(
        f"""
        Скоростта за зонирането е **модулирана за наклон** (на база модела ΔV%),  
        така че да отразява приблизително еднакво усилие на равен терен.

        - `zone` – зона Z1–Z6  
        - `time_min` – минути в зоната  
        - `time_pct` – % от общото време  
        - `mean_speed_mod_kmh` – средна модулирана скорост в зоната (km/h)
        """
    )

    zone_df = pd.DataFrame(zone_stats)
    st.dataframe(
        zone_df.style.format(
            {
                "time_min": "{:.1f}",
                "time_pct": "{:.1f}",
                "mean_speed_mod_kmh": "{:.2f}",
            }
        )
    )

    # 3) Разделителна таблица по активности
    st.markdown("**Разпределение по зони по отделни активности**")

    per_act_rows = []
    for act_name, g in seg_eff.groupby("activity"):
        total_t_act = g["duration_s"].sum()
        for low, high, name in ZONES:
            z = g[g["zone"] == name]
            if z.empty:
                per_act_rows.append(
                    {
                        "activity": act_name,
                        "zone": name,
                        "time_min": 0.0,
                        "time_pct": 0.0,
                        "mean_speed_mod_kmh": np.nan,
                    }
                )
            else:
                t_s = z["duration_s"].sum()
                t_min = t_s / 60.0
                t_pct = 100.0 * t_s / total_t_act if total_t_act > 0 else 0.0
                mean_speed = np.average(z["speed_mod_kmh"], weights=z["duration_s"])
                per_act_rows.append(
                    {
                        "activity": act_name,
                        "zone": name,
                        "time_min": t_min,
                        "time_pct": t_pct,
                        "mean_speed_mod_kmh": mean_speed,
                    }
                )

    if per_act_rows:
        per_act_df = pd.DataFrame(per_act_rows)
        st.dataframe(
            per_act_df.style.format(
                {
                    "time_min": "{:.1f}",
                    "time_pct": "{:.1f}",
                    "mean_speed_mod_kmh": "{:.2f}",
                }
            )
        )


# -----------------------
# MAIN APP
# -----------------------

def main():
    st.sidebar.title("Избор на модел")
    mode = st.sidebar.radio(
        "Модел:",
        (
            "Ski Glide Dynamics (V=f(% наклон))",
            "Влияние на наклона (ΔV% спрямо равното)",
            "CS Зони (разпределение по критична скорост)",
        ),
    )

    st.title("onFlows – Ski Glide Dynamics & Slope Models")

    uploaded_files = st.file_uploader(
        "Качи един или повече TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи TCX файлове, за да започнем анализа.")
        return

    all_segments_list = []
    downhill_segments_list = []
    raw_by_activity = {}

    for file in uploaded_files:
        df_raw = parse_tcx(file)
        if df_raw.empty:
            continue

        raw_by_activity[file.name] = df_raw

        seg_df = build_segments(df_raw)
        if seg_df.empty:
            continue

        seg_df["activity"] = file.name
        all_segments_list.append(seg_df)

        seg_down = filter_downhill_with_predecessor(seg_df)
        if not seg_down.empty:
            seg_down["activity"] = file.name
            downhill_segments_list.append(seg_down)

    if not all_segments_list:
        st.error("Не успях да извлека валидни сегменти от нито един файл.")
        return

    all_segments = pd.concat(all_segments_list, ignore_index=True)
    all_segments = all_segments[
        all_segments["slope_percent"].abs() <= GLOBAL_SLOPE_LIMIT
    ].copy()

    if mode.startswith("Ski Glide Dynamics"):
        show_ski_glide_model(downhill_segments_list, raw_by_activity)

    elif mode.startswith("Влияние на наклона"):
        show_slope_effect_model(all_segments)

    elif mode.startswith("CS Зони"):
        show_cs_zones_model(all_segments)


if __name__ == "__main__":
    main()
