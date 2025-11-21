import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
import plotly.express as px

# ------------------------
# НАСТРОЙКИ НА МОДЕЛА
# ------------------------

SEGMENT_LENGTH_SEC = 5.0         # дължина на сегмента (в сек)
MIN_SEGMENT_DURATION_SEC = 4.0   # минимална реална продължителност
MIN_SEGMENT_DISTANCE_M = 5.0     # минимална хоризонтална дистанция
MIN_SLOPE_PERCENT = -5.0         # търсим сегменти със среден наклон < -5%
TRIM_QUANTILE_LOW = 0.10         # долен квантил за скоростта (изрязваме най-бавните)
TRIM_QUANTILE_HIGH = 0.90        # горен квантил за скоростта (изрязваме най-бързите)


# ------------------------
# ПОМОЩНИ ФУНКЦИИ
# ------------------------

def parse_tcx(file) -> pd.DataFrame:
    """
    Парсва TCX файл и връща DataFrame със сурови данни:
    time, altitude_m, distance_m, dt, speed_kmh.

    Работим минимално чисто:
    - сортиране по време
    - drop на дубликати по time
    - dt > 0
    """
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    trackpoints = root.findall(".//{*}Trackpoint")

    records = []
    for tp in trackpoints:
        time_el = tp.find(".//{*}Time")
        alt_el = tp.find(".//{*}AltitudeMeters")
        dist_el = tp.find(".//{*}DistanceMeters")

        if time_el is None or alt_el is None or dist_el is None:
            continue

        try:
            t = pd.to_datetime(time_el.text)
        except Exception:
            continue

        try:
            alt = float(alt_el.text)
        except Exception:
            alt = np.nan

        try:
            dist = float(dist_el.text)
        except Exception:
            dist = np.nan

        records.append(
            {
                "time": t,
                "altitude_m": alt,
                "distance_m": dist,
            }
        )

    if not records:
        return pd.DataFrame(columns=["time", "altitude_m", "distance_m"])

    df = pd.DataFrame(records)
    df = df.sort_values("time").reset_index(drop=True)

    # Премахваме дублирани времена
    df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Изчисляваме dt
    df["dt"] = df["time"].diff().dt.total_seconds()
    df.loc[0, "dt"] = np.nan

    # Попълваме distance_cum
    df["distance_m"] = df["distance_m"].ffill()

    # Разлика по дистанция
    df["d_dist"] = df["distance_m"].diff()

    # Скорост (km/h) на база d_dist и dt
    df["speed_kmh"] = (df["d_dist"] / df["dt"]) * 3.6

    # Премахваме редове без ключови данни
    df = df.dropna(subset=["time", "altitude_m", "distance_m", "dt", "speed_kmh"]).copy()

    # Само dt > 0
    df = df[df["dt"] > 0].copy()

    df = df.sort_values("time").reset_index(drop=True)
    return df


def make_segments(df: pd.DataFrame, seg_length_sec: float = SEGMENT_LENGTH_SEC) -> pd.DataFrame:
    """
    Разделя активността на 5-секундни сегменти.
    Без сложни условия – само:
      - минимална продължителност
      - минимална дистанция
    После ще филтрираме по наклон и скорост отделно.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame()

    t0 = df["time"].iloc[0]
    df = df.sort_values("time").copy()

    # Индекс на сегмента по време
    df["seg_idx"] = ((df["time"] - t0).dt.total_seconds() // seg_length_sec).astype(int)

    seg_rows = []
    for seg_idx, g in df.groupby("seg_idx"):
        g = g.sort_values("time")

        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration = (t_end - t_start).total_seconds()

        if duration < MIN_SEGMENT_DURATION_SEC:
            continue

        dist_start = g["distance_m"].iloc[0]
        dist_end = g["distance_m"].iloc[-1]
        if pd.isna(dist_start) or pd.isna(dist_end):
            continue

        seg_dist = dist_end - dist_start
        if seg_dist <= MIN_SEGMENT_DISTANCE_M:
            continue

        alt_start = g["altitude_m"].iloc[0]
        alt_end = g["altitude_m"].iloc[-1]
        delta_h = alt_end - alt_start

        mean_speed = (seg_dist / duration) * 3.6
        mean_slope_percent = (delta_h / seg_dist) * 100.0

        seg_rows.append(
            {
                "seg_idx": seg_idx,
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration,
                "distance_m": seg_dist,
                "mean_speed_kmh": mean_speed,
                "mean_slope_percent": mean_slope_percent,
            }
        )

    if not seg_rows:
        return pd.DataFrame()

    seg_df = pd.DataFrame(seg_rows).sort_values("seg_idx").reset_index(drop=True)
    return seg_df


def fit_speed_slope_model(segments: pd.DataFrame):
    """
    Фитва проста линейна регресия:
    mean_speed_kmh ≈ a + b * mean_slope_percent
    върху вече ОФИЛТРИРАНИТЕ сегменти (наклон < -5% и скорост в [Qlow, Qhigh]).

    Връща:
    - segments с колона 'speed_pred_kmh'
    - речник с коефициенти.
    """
    if segments.shape[0] < 3:
        return segments, None

    X = np.column_stack(
        [
            np.ones(len(segments)),
            segments["mean_slope_percent"].values,
        ]
    )
    y = segments["mean_speed_kmh"].values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta

    segments = segments.copy()
    segments["speed_pred_kmh"] = y_pred

    coeffs = {
        "a": beta[0],
        "b_slope": beta[1],
    }

    return segments, coeffs


# ------------------------
# STREAMLIT UI
# ------------------------

st.title("Ski Glide Dynamics – опростен модел (скорост ↔ наклон)")

st.markdown(
    """
**Идея:**
- Вземаме всички 5-секундни сегменти със спускане (**наклон < -5%**).
- Изрязваме крайностите по скорост (най-бавни и най-бързи сегменти).
- Строим проста зависимост **скорост = f(наклон)**.
- За всяка активност оценяваме **колко бърза е спрямо очакваното** за нейните наклони
  → индекс за *плъзгаемост*.
"""
)

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла",
    type=["tcx"],
    accept_multiple_files=True,
)

if uploaded_files:
    all_segments = []

    for f in uploaded_files:
        st.write(f"Файл: **{f.name}**")
        df = parse_tcx(f)

        if df.empty:
            st.warning("Не успях да извадя валидни Trackpoint-и от този файл.")
            continue

        seg_df = make_segments(df, SEGMENT_LENGTH_SEC)
        if seg_df.empty:
            st.warning("Няма 5-сек сегменти с достатъчна продължителност/дистанция.")
            continue

        # Добавяме име на файла
        seg_df["file_name"] = f.name
        all_segments.append(seg_df)

    if not all_segments:
        st.error("Няма нито един валиден сегмент за всички файлове.")
    else:
        segments = pd.concat(all_segments, ignore_index=True)

        # 1) Филтър по наклон < -5%
        segments_downhill = segments[segments["mean_slope_percent"] < MIN_SLOPE_PERCENT].copy()

        if segments_downhill.empty:
            st.error("Няма сегменти със среден наклон под -5% (спускания).")
        else:
            st.subheader("Всички downhill сегменти (наклон < -5%)")
            st.dataframe(
                segments_downhill[
                    [
                        "file_name",
                        "seg_idx",
                        "t_start",
                        "t_end",
                        "duration_s",
                        "distance_m",
                        "mean_speed_kmh",
                        "mean_slope_percent",
                    ]
                ]
            )

            # 2) Тримване по скорост – махаме най-ниските и най-високите
            q_low = segments_downhill["mean_speed_kmh"].quantile(TRIM_QUANTILE_LOW)
            q_high = segments_downhill["mean_speed_kmh"].quantile(TRIM_QUANTILE_HIGH)

            segments_trimmed = segments_downhill[
                (segments_downhill["mean_speed_kmh"] >= q_low)
                & (segments_downhill["mean_speed_kmh"] <= q_high)
            ].copy()

            st.subheader("Downhill сегменти след изрязване на крайностите по скорост")
            st.markdown(
                f"Използваме сегменти със скорост между **{q_low:.2f} km/h** и **{q_high:.2f} km/h**."
            )
            st.dataframe(
                segments_trimmed[
                    [
                        "file_name",
                        "seg_idx",
                        "mean_speed_kmh",
                        "mean_slope_percent",
                    ]
                ]
            )

            # 3) Фитваме модела скорост ~ наклон
            segments_trimmed, coeffs = fit_speed_slope_model(segments_trimmed)

            if coeffs is None:
                st.error("Недостатъчно сегменти за изграждане на модел скорост ~ наклон.")
            else:
                a = coeffs["a"]
                b = coeffs["b_slope"]

                st.subheader("Модел: скорост = f(наклон)")
                st.markdown(
                    f"""
Линеен модел:

\\[
V_{{mean}} \\approx a + b_{{slope}} \\cdot slope
\\]

където:
- \\(V_{{mean}}\\) – средна скорост (km/h)
- \\(slope\\) – среден наклон (%)

Коефициенти:
- **a** = {a:.4f}
- **b_slope** = {b:.4f}
"""
                )

                # 4) 2D визуализация: скорост срещу наклон
                st.subheader("Зависимост скорост – наклон (използвани сегменти)")
                fig = px.scatter(
                    segments_trimmed,
                    x="mean_slope_percent",
                    y="mean_speed_kmh",
                    color="file_name",
                    labels={
                        "mean_slope_percent": "Среден наклон (%)",
                        "mean_speed_kmh": "Средна скорост (km/h)",
                        "file_name": "Активност",
                    },
                    trendline="ols",  # ще добави регресионна права (само за визуализация)
                )
                st.plotly_chart(fig, use_container_width=True)

                # 5) Оценъчен модел за всяка активност
                st.subheader("Оценка на активностите – индекс за плъзгаемост")

                # За всяка активност взимаме trimmed downhill сегментите й
                eval_rows = []
                for fname, g in segments_trimmed.groupby("file_name"):
                    if g.empty:
                        continue

                    # Предсказана скорост за всеки сегмент според глобалния модел
                    g = g.copy()
                    g["speed_pred_kmh"] = a + b * g["mean_slope_percent"]

                    mean_real_speed = g["mean_speed_kmh"].mean()
                    mean_pred_speed = g["speed_pred_kmh"].mean()
                    mean_slope_file = g["mean_slope_percent"].mean()

                    diff_index = mean_real_speed - mean_pred_speed
                    ratio_index = mean_real_speed / mean_pred_speed if mean_pred_speed != 0 else np.nan

                    eval_rows.append(
                        {
                            "file_name": fname,
                            "n_segments_used": g.shape[0],
                            "mean_slope_percent": mean_slope_file,
                            "mean_real_speed_kmh": mean_real_speed,
                            "mean_pred_speed_kmh": mean_pred_speed,
                            "difference_index_kmh": diff_index,
                            "ratio_index": ratio_index,
                        }
                    )

                if not eval_rows:
                    st.error("Не успях да изчисля оценъчен индекс за активностите.")
                else:
                    eval_df = pd.DataFrame(eval_rows).sort_values("difference_index_kmh", ascending=False)
                    st.dataframe(eval_df)

                    st.markdown(
                        """
Интерпретация:

- **difference_index_kmh**: реална средна скорост – прогнозирана скорост  
  - > 0 → активността е *по-бърза* от очакваното за нейните наклони (по-добра плъзгаемост)  
  - < 0 → активността е *по-бавна* от очакваното

- **ratio_index**: реална / прогнозирана скорост  
  - > 1 → по-добра плъзгаемост  
  - < 1 → по-лоша плъзгаемост
"""
                    )

                    # бутон за сваляне на сегментите
                    csv_segments = segments_trimmed.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Свали използваните сегменти (downhill, trimmed) като CSV",
                        data=csv_segments,
                        file_name="ski_glide_segments_simple_model.csv",
                        mime="text/csv",
                    )

                    # бутон за сваляне на оценката по активности
                    csv_eval = eval_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Свали оценката по активности като CSV",
                        data=csv_eval,
                        file_name="ski_glide_activity_scores.csv",
                        mime="text/csv",
                    )
else:
    st.info("Качи поне един TCX файл, за да започнем анализа.")
