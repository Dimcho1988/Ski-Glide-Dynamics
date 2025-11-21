import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime

# -----------------------
# НАСТРОЙКИ НА МОДЕЛА
# -----------------------
SEGMENT_LENGTH_SEC = 5.0       # 5-секундни сегменти
MIN_SLOPE_PERCENT = -15.0      # не допускаме по-стръмно от -15%
MAX_SLOPE_PERCENT = -5.0       # сегментите за модела трябва да са със среден наклон <= -5%
MIN_SEGMENT_DISTANCE_M = 5.0   # минимална хоризонтална дистанция, за да не е шум

# Процент за отрязване на екстремни стойности на V/|slope|
# 0.05 = отрязваме по 5% отдолу и отгоре (общо 10%)
RATIO_TRIM_Q = 0.03


# -----------------------
# ПОМОЩНИ ФУНКЦИИ
# -----------------------
def parse_tcx(file) -> pd.DataFrame:
    """
    Парсва TCX файл и връща DataFrame с:
    time (datetime), alt (m), dist (m), elapsed_s (sec)
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
    Разделя активността на 5-секундни сегменти (последователни, не припокриващи се).
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

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


def filter_segments_with_predecessor(seg_df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Наклон в диапазона [-15%, -5%].
    2) Всеки сегмент трябва да е предхождан от сегмент (seg_id - 1)
       със същия диапазон на наклона.
    """
    if seg_df.empty:
        return seg_df

    base_mask = (
        (seg_df["slope_percent"] >= MIN_SLOPE_PERCENT)
        & (seg_df["slope_percent"] <= MAX_SLOPE_PERCENT)
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

        if MIN_SLOPE_PERCENT <= prev_slope <= MAX_SLOPE_PERCENT:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    seg_df["valid"] = valid_mask
    seg_df = seg_df[seg_df["valid"]].drop(columns=["valid"])
    return seg_df


def filter_by_speed_slope_ratio(all_segments: pd.DataFrame, q: float = RATIO_TRIM_Q):
    """
    Отрязва екстремни сегменти според R = V / |slope|.
    Оставя сегментите между q и (1-q) персентил.
    """
    if all_segments.empty or len(all_segments) < 10:
        return all_segments, None, None  # твърде малко данни – не режем

    segs = all_segments.copy()
    segs["speed_per_abs_slope"] = segs["speed_kmh"] / segs["slope_percent"].abs()

    r = segs["speed_per_abs_slope"].values
    r_low = np.quantile(r, q)
    r_high = np.quantile(r, 1 - q)

    mask = (segs["speed_per_abs_slope"] >= r_low) & (segs["speed_per_abs_slope"] <= r_high)
    segs = segs[mask].copy()

    return segs, r_low, r_high


def fit_speed_vs_slope_model(all_segments: pd.DataFrame):
    """
    Линейна регресия: V = a * slope_percent + b
    """
    if all_segments.empty or len(all_segments) < 5:
        return None, None

    x = all_segments["slope_percent"].values
    y = all_segments["speed_kmh"].values

    try:
        a, b = np.polyfit(x, y, 1)
        return a, b
    except Exception:
        return None, None


def compute_activity_summary(activity_name: str,
                             df_raw: pd.DataFrame,
                             seg_valid: pd.DataFrame,
                             a: float,
                             b: float) -> dict:
    """
    Обобщение за една активност.
    """
    if df_raw.empty:
        return None

    total_dist = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
    total_time = df_raw["elapsed_s"].iloc[-1] - df_raw["elapsed_s"].iloc[0]
    overall_speed = (total_dist / total_time) * 3.6 if total_time > 0 else np.nan

    if seg_valid.empty or a is None or b is None:
        return {
            "activity": activity_name,
            "n_segments": 0,
            "mean_slope_valid": np.nan,
            "mean_speed_valid": np.nan,
            "model_speed_at_mean_slope": np.nan,
            "glide_coeff": np.nan,
            "overall_speed": overall_speed,
            "normalized_overall_speed": np.nan,
        }

    mean_slope = seg_valid["slope_percent"].mean()
    mean_speed = seg_valid["speed_kmh"].mean()
    model_speed = a * mean_slope + b

    if model_speed <= 0:
        glide_coeff = np.nan
        norm_overall = np.nan
    else:
        glide_coeff = mean_speed / model_speed
        norm_overall = overall_speed / glide_coeff if glide_coeff not in (0, np.nan) else np.nan

    return {
        "activity": activity_name,
        "n_segments": int(len(seg_valid)),
        "mean_slope_valid": mean_slope,
        "mean_speed_valid": mean_speed,
        "model_speed_at_mean_slope": model_speed,
        "glide_coeff": glide_coeff,
        "overall_speed": overall_speed,
        "normalized_overall_speed": norm_overall,
    }


# -----------------------
# STREAMLIT UI
# -----------------------
def main():
    st.title("Ski Glide Dynamics – модел V = f(% наклон)")

    st.markdown(
        f"""
        **Логика на модела**

        1. Зареждаш няколко **TCX файла** от ски бягане.
        2. За всяка активност:
           - разделяме на 5-секундни сегменти;
           - избираме само сегменти с наклон между **{MIN_SLOPE_PERCENT}% и {MAX_SLOPE_PERCENT}%**,
             които са предхождани от сегмент със същия диапазон.
        3. От всички валидни сегменти изчисляваме
           \\( R = V / |slope\\_%| \\) и режем най-екстремните
           **{int(RATIO_TRIM_Q*100)}% отдолу и отгоре**, за да не развалят средните.
        4. Строим линейна зависимост:
           \\( V = a \\cdot slope\\_% + b \\).
        5. За всяка активност намираме:
           - реална средна скорост на валидните сегменти,
           - моделна скорост при същия среден наклон,
           - коефициент на плъзгане \\(K = V_{{real}} / V_{{model}}\\),
           - нормализирана средна скорост за цялата активност
             \\( V_{{norm}} = V_{{overall}} / K \\).
        """
    )

    uploaded_files = st.file_uploader(
        "Качи един или повече TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Моля, качи поне един TCX файл.")
        return

    all_valid_segments = []
    raw_by_activity = {}

    # 1) Парсване и първичен филтър
    for file in uploaded_files:
        st.subheader(f"Активност: {file.name}")

        df_raw = parse_tcx(file)
        if df_raw.empty:
            st.warning("Не бяха намерени валидни Trackpoint-и в TCX.")
            continue

        raw_by_activity[file.name] = df_raw

        seg_df = build_segments(df_raw)
        if seg_df.empty:
            st.warning("Неуспешно създаване на сегменти (твърде малко данни).")
            continue

        seg_valid = filter_segments_with_predecessor(seg_df)

        st.write(f"Общо сегменти: {len(seg_df)}, валидни (наклон + предходен): {len(seg_valid)}")

        if not seg_valid.empty:
            seg_valid = seg_valid.assign(activity=file.name)
            all_valid_segments.append(seg_valid)
            st.dataframe(
                seg_valid[["seg_id", "slope_percent", "speed_kmh", "horiz_dist_m"]].head()
            )

    if not all_valid_segments:
        st.error("Няма нито един валиден сегмент, който да отговаря на критериите.")
        return

    all_valid_segments = pd.concat(all_valid_segments, ignore_index=True)

    # 2) Филтър по отношение скорост/наклон
    st.markdown("### Филтър по отношение скорост / |наклон|")
    before_n = len(all_valid_segments)
    all_valid_segments, r_low, r_high = filter_by_speed_slope_ratio(all_valid_segments)

    if r_low is not None:
        after_n = len(all_valid_segments)
        st.write(
            f"Сегменти преди филтъра: **{before_n}**, "
            f"след филтъра: **{after_n}**"
        )
        st.write(
            f"Допускаме сегменти с V/|slope| между "
            f"**{r_low:.2f}** и **{r_high:.2f}** km/h на 1% наклон."
        )
    else:
        st.write("Твърде малко сегменти – не е прилаган допълнителен филтър по V/|slope|.")

    st.markdown("### Обединени валидни сегменти (след всички филтри)")
    st.dataframe(
        all_valid_segments[
            ["activity", "seg_id", "slope_percent", "speed_kmh", "horiz_dist_m"]
        ].head(200)
    )

    # 3) Модел V = f(slope)
    a, b = fit_speed_vs_slope_model(all_valid_segments)
    if a is None or b is None:
        st.error("Неуспех при построяване на модел V = f(наклон). Няма достатъчно данни.")
        return

    st.markdown("### Модел V = f(% наклон)")
    st.write("Форма: **V = a * slope_percent + b**")
    st.write(f"**a = {a:.4f} (km/h на 1% наклон)**")
    st.write(f"**b = {b:.4f} (km/h при 0% наклон по модела)**")

    # 4) Обобщение по активности (използваме само сегментите след всички филтри)
    summaries = []
    for name, df_raw in raw_by_activity.items():
        seg_act = all_valid_segments[all_valid_segments["activity"] == name].copy()
        seg_act = seg_act.drop(columns=["activity"]) if not seg_act.empty else seg_act
        summary = compute_activity_summary(name, df_raw, seg_act, a, b)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        st.error("Не успях да изчисля обобщение за нито една активност.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[
        [
            "activity",
            "n_segments",
            "mean_slope_valid",
            "mean_speed_valid",
            "model_speed_at_mean_slope",
            "glide_coeff",
            "overall_speed",
            "normalized_overall_speed",
        ]
    ]

    st.markdown("## Обобщение по активности")
    st.markdown(
        """
        - **glide_coeff** – коефициент на плъзгане \\(K = V_{real}/V_{model}\\).
        - **normalized_overall_speed** – средна скорост на цялата активност,
          приравнена към „референтна“ плъзгаемост (по модела).
        """
    )

    st.dataframe(summary_df.style.format(
        {
            "mean_slope_valid": "{:.2f}",
            "mean_speed_valid": "{:.2f}",
            "model_speed_at_mean_slope": "{:.2f}",
            "glide_coeff": "{:.3f}",
            "overall_speed": "{:.2f}",
            "normalized_overall_speed": "{:.2f}",
        }
    ))

    # 5) Scatter V vs slope – без matplotlib
    st.markdown("### Разпределение на скоростта спрямо наклона (валидни сегменти)")
    chart_df = all_valid_segments[["slope_percent", "speed_kmh"]]
    st.scatter_chart(chart_df, x="slope_percent", y="speed_kmh")


if __name__ == "__main__":
    main()
