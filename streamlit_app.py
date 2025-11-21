import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime

# -----------------------
# НАСТРОЙКИ НА МОДЕЛА
# -----------------------
SEGMENT_LENGTH_SEC = 5.0      # 5-секундни сегменти
MIN_SLOPE_PERCENT = -15.0     # не допускаме по-стръмно от -15%
MAX_SLOPE_PERCENT = -5.0      # сегментите за модела трябва да са със среден наклон <= -5%
MIN_SEGMENT_DISTANCE_M = 5.0  # минимална хоризонтална дистанция, за да не е шум


# -----------------------
# ПОМОЩНИ ФУНКЦИИ
# -----------------------
def parse_tcx(file) -> pd.DataFrame:
    """
    Парсва TCX файл и връща DataFrame с:
    time (datetime), alt (m), dist (m), elapsed_s (sec)
    Използваме DistanceMeters, ако го има, иначе кумулираме по скорост (ако има).
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

    # TCX: минаваме през всички Trackpoint
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        t_el = get_child_with_tag(tp, "Time")
        alt_el = get_child_with_tag(tp, "AltitudeMeters")
        dist_el = get_child_with_tag(tp, "DistanceMeters")

        if t_el is None or alt_el is None or dist_el is None:
            # ако липсва някое от трите, прескачаме
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

    # време от начало
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    return df


def build_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разделя активността на 5-секундни сегменти (последователни, НЕ припокриващи се).
    За всеки сегмент изчислява:
    - start_time, end_time
    - duration_s
    - alt_start, alt_end
    - dist_start, dist_end
    - horiz_dist_m
    - slope_percent
    - speed_kmh (средна за сегмента)
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    # индекс на сегмента = floor(elapsed / SEGMENT_LENGTH_SEC)
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
            # много малко движение -> шум
            continue

        delta_alt = alt_end - alt_start  # m
        slope_percent = (delta_alt / horiz_dist) * 100.0

        # средна скорост за сегмента (km/h)
        speed_kmh = (horiz_dist / duration) * 3.6

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
    1) Взимаме само сегменти със среден наклон:
       -15% <= slope <= -5% (реалистичен спуск)
    2) Задължително САМИЯТ сегмент И предходният (seg_id - 1)
       да имат наклон <= -5% и >= -15%.
       Така намаляваме ефекта на инерцията.
    """
    if seg_df.empty:
        return seg_df

    # филтър за реалистичен наклон
    base_mask = (
        (seg_df["slope_percent"] >= MIN_SLOPE_PERCENT)
        & (seg_df["slope_percent"] <= MAX_SLOPE_PERCENT)
    )
    seg_df = seg_df[base_mask].copy()

    if seg_df.empty:
        return seg_df

    seg_df = seg_df.sort_values("seg_id").reset_index(drop=True)

    # правим lookup по seg_id
    slope_by_id = dict(zip(seg_df["seg_id"], seg_df["slope_percent"]))

    valid_mask = []
    for _, row in seg_df.iterrows():
        sid = row["seg_id"]
        prev_id = sid - 1

        # сегментът вече е минал филтъра за наклон
        # проверяваме предходния сегмент:
        prev_slope = slope_by_id.get(prev_id, None)
        if prev_slope is None:
            # няма предходен 5-сек сегмент, който да минава филтъра
            valid_mask.append(False)
            continue

        # предходният също трябва да е в същия диапазон
        if MIN_SLOPE_PERCENT <= prev_slope <= MAX_SLOPE_PERCENT:
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    seg_df["valid"] = valid_mask
    seg_df = seg_df[seg_df["valid"]].drop(columns=["valid"])

    return seg_df


def fit_speed_vs_slope_model(all_segments: pd.DataFrame):
    """
    Строим линейна регресия:
        V = a * slope_percent + b
    Връщаме (a, b) или (None, None), ако няма достатъчно данни.
    """
    if all_segments.empty or len(all_segments) < 5:
        return None, None

    x = all_segments["slope_percent"].values
    y = all_segments["speed_kmh"].values

    # линейна регресия (степен 1)
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
    За една активност:
    - среден наклон на валидните сегменти
    - средна скорост на валидните сегменти
    - моделна скорост при този наклон: V_model = a * slope_mean + b
    - коефициент на плъзгане: K = V_real / V_model
    - средна скорост за цялата активност
    - нормализирана средна скорост (моделирана): V_norm = V_overall / K
      (все едно плъзгаемостта е „референтна“ по модела)
    """
    if df_raw.empty:
        return None

    # обща средна скорост за цялата активност
    total_dist = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
    total_time = df_raw["elapsed_s"].iloc[-1] - df_raw["elapsed_s"].iloc[0]
    if total_time <= 0:
        overall_speed = np.nan
    else:
        overall_speed = (total_dist / total_time) * 3.6

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
        # ако K > 1 -> реалната скорост е по-висока (по-добра плъзгаемост)
        # нормализираме цялата активност към „референтна“ плъзгаемост:
        if glide_coeff != 0 and not np.isnan(glide_coeff):
            norm_overall = overall_speed / glide_coeff
        else:
            norm_overall = np.nan

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
        """
        **Логика на модела:**

        1. Зареждаш няколко **TCX файла** от ски бягане.
        2. За всяка активност:
           - Данните се делят на **5-секундни сегменти**.
           - За всеки сегмент се изчисляват:
             - среден наклон (на база начална/крайна височина и хоризонтална дистанция),
             - средна скорост.
           - В модела влизат само сегменти със:
             - наклон между **-15% и -5%**, и
             - всеки сегмент е **предхождан** от сегмент със същия диапазон на наклона
               (намаляваме ефекта на инерцията).
        3. От **всички валидни сегменти от всички активности** строим линейна зависимост:
           \\( V = a \\cdot slope\\_% + b \\).
        4. За всяка активност намираме:
           - реалната средна скорост на валидните сегменти,
           - моделната скорост при средния им наклон,
           - **коефициент на плъзгане** \\(K = V_{real} / V_{model}\\),
           - нормализирана средна скорост за цялата активност:
             \\( V_{norm} = V_{overall} / K \\).
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
    per_activity_valid_segments = {}
    raw_by_activity = {}

    # 1) Парсване и сегментиране на всички активности
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

        per_activity_valid_segments[file.name] = seg_valid

        st.write(f"Общо сегменти: {len(seg_df)}, валидни за модела: {len(seg_valid)}")

        if not seg_valid.empty:
            # малка таблица с първите няколко сегмента
            st.dataframe(
                seg_valid[["seg_id", "slope_percent", "speed_kmh", "horiz_dist_m"]].head()
            )
            all_valid_segments.append(
                seg_valid.assign(activity=file.name)
            )

    if not all_valid_segments:
        st.error("Няма нито един валиден сегмент, който да отговаря на критериите.")
        return

    all_valid_segments = pd.concat(all_valid_segments, ignore_index=True)

    st.markdown("### Обединени валидни сегменти от всички активности")
    st.dataframe(
        all_valid_segments[
            ["activity", "seg_id", "slope_percent", "speed_kmh", "horiz_dist_m"]
        ].head(200)
    )

    # 2) Строим модел V = f(slope)
    a, b = fit_speed_vs_slope_model(all_valid_segments)
    if a is None or b is None:
        st.error("Неуспех при построяване на модел V = f(наклон). Няма достатъчно данни.")
        return

    st.markdown("### Модел V = f(% наклон)")
    st.write(f"Форма: **V = a * slope_percent + b**")
    st.write(f"**a = {a:.4f}  (km/h на 1% наклон)**")
    st.write(f"**b = {b:.4f}  (km/h при 0% наклон по модела)**")

    # 3) Обобщена таблица по активности
    summaries = []
    for name, df_raw in raw_by_activity.items():
        seg_valid = per_activity_valid_segments.get(name, pd.DataFrame())
        summary = compute_activity_summary(name, df_raw, seg_valid, a, b)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        st.error("Не успях да изчисля обобщение за нито една активност.")
        return

    summary_df = pd.DataFrame(summaries)

    # Подреждане на колоните по ясен начин
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
        - **mean_slope_valid** – среден наклон на всички валидни сегменти (в %).
        - **mean_speed_valid** – реална средна скорост на валидните сегменти (km/h).
        - **model_speed_at_mean_slope** – скорост по модела при същия среден наклон.
        - **glide_coeff (K)** – ако K > 1 → реалната скорост е по-висока от моделната (по-добра плъзгаемост).
        - **overall_speed** – средна скорост за цялата активност.
        - **normalized_overall_speed** – „приравнена“ средна скорост, все едно плъзгаемостта е референтна.
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

    # По желание: scatter графика V vs slope
    st.markdown("### Разпределение на скоростта спрямо наклона (валидни сегменти)")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(
        all_valid_segments["slope_percent"],
        all_valid_segments["speed_kmh"],
        alpha=0.4,
    )

    # линия на модела
    x_vals = np.linspace(
        all_valid_segments["slope_percent"].min(),
        all_valid_segments["slope_percent"].max(),
        100
    )
    y_vals = a * x_vals + b
    ax.plot(x_vals, y_vals)

    ax.set_xlabel("Наклон (%)")
    ax.set_ylabel("Скорост (km/h)")
    ax.set_title("Модел V = f(% наклон) и валидни сегменти")

    st.pyplot(fig)


if __name__ == "__main__":
    main()
