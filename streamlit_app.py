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
DOWNHILL_RELAX_SLOPE = -3.0    # slope <= -3% → натоварване в горна граница на Z1

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
# ПОМОЩНИ ФУНКЦИИ – ОБЩИ
# -----------------------

def parse_tcx(file) -> pd.DataFrame:
    """
    Парсва TCX файл и връща DataFrame с колони:
    time (datetime), alt (m), dist (m), elapsed_s (sec).
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
    2) Всеки сегмент трябва да е предхождан от сегмент със същия диапазон на наклона.
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


def show_ski_glide_model(downhill_segments_list):
    """Показва модел 1: Ski Glide Dynamics."""
    st.subheader("Модел 1: Ski Glide Dynamics – V = f(% наклон) (спускане)")

    if not downhill_segments_list:
        st.warning("Няма достатъчно downhill сегменти (−15% до −5%) за модела.")
        return

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
        return

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


def show_slope_effect_model(all_segments: pd.DataFrame):
    """Показва модел 2: влияние на наклона при еднакво усилие (ΔV%)."""
    st.subheader("Модел 2: Влияние на наклона върху скоростта при еднакво усилие")

    if all_segments.empty:
        st.warning("Няма валидни сегменти за анализ.")
        return

    # 1) Референтна скорост V_flat при |slope| <= 1%
    flat_mask = all_segments["slope_percent"].abs() <= FLAT_REF_SLOPE_MAX
    flat_segments = all_segments[flat_mask]

    if flat_segments.empty:
        st.warning("Няма сегменти с |наклон| ≤ 1% за определяне на V_flat.")
        return

    v_flat = flat_segments["speed_kmh"].mean()
    st.write(f"Референтна скорост на равното (|наклон| ≤ {FLAT_REF_SLOPE_MAX:.1f}%): "
             f"**{v_flat:.2f} km/h**")

    # 2) Сегменти за модела: -3% < slope < 10%, без равните
    slope_mask = (
        (all_segments["slope_percent"] > SLOPE_MODEL_MIN) &
        (all_segments["slope_percent"] < SLOPE_MODEL_MAX) &
        (~flat_mask) &
        (all_segments["slope_percent"].abs() <= GLOBAL_SLOPE_LIMIT)
    )
    slope_df = all_segments[slope_mask].copy()

    if slope_df.empty or len(slope_df) < 10:
        st.warning("Недостатъчно сегменти в диапазона (-3%, +10%) за стабилен модел.")
        return

    slope_df["delta_v_pct"] = 100.0 * (slope_df["speed_kmh"] - v_flat) / v_flat

    coeffs = fit_delta_v_model(
        slope_df["slope_percent"].values,
        slope_df["delta_v_pct"].values,
    )

    if coeffs is None:
        st.warning("Не успях да построя квадратичен модел ΔV% = f(% наклон).")
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

    # Визуализация: scatter + моделна линия
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


# -----------------------
# МОДЕЛ 3: CS ЗОНИ
# -----------------------

def assign_zone(speed_kmh: float, slope_percent: float, cs: float) -> str:
    """
    Връща име на зона според критичната скорост (CS).
    - ако slope <= -3% -> натоварване в горната граница на Z1;
    - иначе – според ratio = speed / CS.
    """
    if cs <= 0 or np.isnan(speed_kmh):
        return "NA"

    if slope_percent <= DOWNHILL_RELAX_SLOPE:
        ratio = ZONES[0][1]  # горна граница на Z1
    else:
        ratio = speed_kmh / cs

    for low, high, name in ZONES:
        if low <= ratio < high:
            return name
    return ZONES[-1][2]


def show_cs_zones_model(all_segments: pd.DataFrame):
    """Показва модел 3: разпределение по зони спрямо критична скорост."""
    st.subheader("Модел 3: Разпределение по зони спрямо критична скорост (CS)")

    if all_segments.empty:
        st.warning("Няма валидни сегменти за анализ.")
        return

    cs = st.number_input(
        "Въведи критична скорост (km/h)",
        min_value=0.0,
        value=0.0,
        step=0.1,
    )

    if cs <= 0:
        st.info("Моля, въведи критична скорост, за да видиш зоновете.")
        return

    seg_zones = all_segments.copy()
    seg_zones["zone"] = seg_zones.apply(
        lambda row: assign_zone(row["speed_kmh"], row["slope_percent"], cs),
        axis=1,
    )

    total_time = seg_zones["duration_s"].sum()
    if total_time <= 0:
        st.warning("Общото време на сегментите е нула – не мога да направя разпределение.")
        return

    zone_stats = []
    for low, high, name in ZONES:
        z = seg_zones[seg_zones["zone"] == name]
        if z.empty:
            zone_stats.append(
                {
                    "zone": name,
                    "time_min": 0.0,
                    "time_pct": 0.0,
                    "mean_speed_kmh": np.nan,
                }
            )
        else:
            time_s = z["duration_s"].sum()
            time_min = time_s / 60.0
            time_pct = 100.0 * time_s / total_time
            mean_speed = np.average(z["speed_kmh"], weights=z["duration_s"])
            zone_stats.append(
                {
                    "zone": name,
                    "time_min": time_min,
                    "time_pct": time_pct,
                    "mean_speed_kmh": mean_speed,
                }
            )

    zone_df = pd.DataFrame(zone_stats)

    st.markdown(
        f"""
        Сегментите с **наклон ≤ {DOWNHILL_RELAX_SLOPE:.1f}%** се отчитат като натоварване
        в **горната граница на Z1**, независимо от реалната скорост.  
        Останалите сегменти се разпределят по зони спрямо съотношението V / CS.
        """
    )

    st.dataframe(
        zone_df.style.format(
            {
                "time_min": "{:.1f}",
                "time_pct": "{:.1f}",
                "mean_speed_kmh": "{:.2f}",
            }
        )
    )


# -----------------------
# MAIN APP
# -----------------------

def main():
    # Sidebar – избор на модел
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

    # Събиране на сегментите от всички файлове
    all_segments_list = []
    downhill_segments_list = []

    for file in uploaded_files:
        df_raw = parse_tcx(file)
        if df_raw.empty:
            continue

        seg_df = build_segments(df_raw)
        if seg_df.empty:
            continue

        seg_df["activity"] = file.name
        all_segments_list.append(seg_df)

        # за Ski Glide модела – downhill сегменти
        seg_down = filter_downhill_with_predecessor(seg_df)
        if not seg_down.empty:
            seg_down["activity"] = file.name
            downhill_segments_list.append(seg_down)

    if not all_segments_list:
        st.error("Не успях да извлека валидни сегменти от нито един файл.")
        return

    all_segments = pd.concat(all_segments_list, ignore_index=True)

    # Филтър за екстремни наклони за общ анализ (за ΔV% и частично за CS)
    all_segments = all_segments[
        all_segments["slope_percent"].abs() <= GLOBAL_SLOPE_LIMIT
    ].copy()

    # Избор на модел
    if mode.startswith("Ski Glide Dynamics"):
        show_ski_glide_model(downhill_segments_list)

    elif mode.startswith("Влияние на наклона"):
        show_slope_effect_model(all_segments)

    elif mode.startswith("CS Зони"):
        show_cs_zones_model(all_segments)


if __name__ == "__main__":
    main()
