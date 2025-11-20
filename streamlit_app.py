import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import plotly.express as px

# ------------------------
# НАСТРОЙКИ НА МОДЕЛА
# ------------------------

SEGMENT_LENGTH_SEC = 10.0        # дължина на сегмента (в сек)
MIN_SEGMENT_DURATION_SEC = 8.0   # минимална реална продължителност
MIN_SEGMENT_DISTANCE_M = 5.0     # минимална хоризонтална дистанция (за да няма деление на 0)
MIN_SLOPE_PERCENT = -5.0         # търсим сегменти със среден наклон < -5%
MAX_ALT_JUMP_PER_SEC = 1.0       # макс. позволен скок по височина (м/сек)
MAX_SPEED_JUMP_PER_SEC = 2.0     # макс. позволен скок по скорост (км/ч за 1 сек)


# ------------------------
# ПОМОЩНИ ФУНКЦИИ
# ------------------------

def parse_tcx(file) -> pd.DataFrame:
    """
    Парсва TCX файл и връща DataFrame с:
    time, altitude_m, distance_m
    и изчислени dt, speed_kmh, d_alt, d_speed, valid.
    """
    content = file.read()
    tree = ET.parse(BytesIO(content))
    root = tree.getroot()

    # Взимаме всички Trackpoint, без да се занимаваме с namespace-ите
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

    # Правим разлика в дистанцията
    df["distance_m"] = df["distance_m"].ffill()
    df["d_dist"] = df["distance_m"].diff()

    # Нереалистично обратно движение -> маскираме
    df.loc[df["d_dist"] < 0, "d_dist"] = np.nan

    # Скорост (km/h) на база d_dist и dt
    df["speed_kmh"] = (df["d_dist"] / df["dt"]) * 3.6

    # Разлики по височина и скорост
    df["d_alt"] = df["altitude_m"].diff()
    df["d_speed"] = df["speed_kmh"].diff()

    # Флаг за валидност
    df["valid"] = True

    # Невалидно време (dt <= 0 или много голямо) -> invalid
    df.loc[(df["dt"] <= 0) | (df["dt"].isna()), "valid"] = False

    # Филтър за артефакти по денивелация:
    # ако |Δalt| > MAX_ALT_JUMP_PER_SEC * dt  -> invalid
    mask_bad_alt = (
        (df["dt"] > 0)
        & (df["d_alt"].abs() > MAX_ALT_JUMP_PER_SEC * df["dt"])
    )

    # Филтър за артефакти по скорост:
    # ако |ΔV| > MAX_SPEED_JUMP_PER_SEC * dt -> invalid
    mask_bad_speed = (
        (df["dt"] > 0)
        & (df["d_speed"].abs() > MAX_SPEED_JUMP_PER_SEC * df["dt"])
    )

    df.loc[mask_bad_alt | mask_bad_speed, "valid"] = False

    return df


def extract_segments(df: pd.DataFrame, seg_length_sec: float = SEGMENT_LENGTH_SEC) -> pd.DataFrame:
    """
    Разделя активността на 10-секундни сегменти и връща
    само тези, които:
    - имат среден наклон < MIN_SLOPE_PERCENT;
    - предходният сегмент също има среден наклон < MIN_SLOPE_PERCENT;
    - височината вътре в сегмента е само намаляваща (мон. не-нарастваща);
    - всички точки са "valid".
    """

    # Взимаме само валидните точки
    df = df[df["valid"]].copy()
    df = df.dropna(subset=["time", "altitude_m", "distance_m", "speed_kmh"])
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame()

    t0 = df["time"].iloc[0]
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

        # Проверка за достатъчна продължителност
        if duration < MIN_SEGMENT_DURATION_SEC:
            continue

        dist_start = g["distance_m"].iloc[0]
        dist_end = g["distance_m"].iloc[-1]
        if pd.isna(dist_start) or pd.isna(dist_end):
            continue

        seg_dist = dist_end - dist_start
        if seg_dist <= MIN_SEGMENT_DISTANCE_M:
            continue

        # Проверка за строго спускане: височината не трябва да нараства никъде
        alt_values = g["altitude_m"].values
        if np.any(np.diff(alt_values) > 0):
            # има поне една стъпка, където височината се е качила
            continue

        # Изчисляваме средна скорост (по сегмент)
        mean_speed = (seg_dist / duration) * 3.6

        alt_start = alt_values[0]
        alt_end = alt_values[-1]
        delta_h = alt_end - alt_start  # отрицателна при спускане

        mean_slope_percent = (delta_h / seg_dist) * 100.0

        # Начална и крайна скорост в сегмента
        v_start = g["speed_kmh"].iloc[0]
        v_end = g["speed_kmh"].iloc[-1]
        if pd.isna(v_start) or pd.isna(v_end):
            continue

        delta_v = v_end - v_start

        seg_rows.append(
            {
                "seg_idx": seg_idx,
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration,
                "distance_m": seg_dist,
                "mean_speed_kmh": mean_speed,
                "mean_slope_percent": mean_slope_percent,
                "v_start_kmh": v_start,
                "v_end_kmh": v_end,
                "delta_v_kmh": delta_v,
            }
        )

    if not seg_rows:
        return pd.DataFrame()

    seg_df = pd.DataFrame(seg_rows).sort_values("seg_idx").reset_index(drop=True)

    # Маркираме "надолу" сегментите
    seg_df["downhill"] = seg_df["mean_slope_percent"] < MIN_SLOPE_PERCENT

    # Предходен сегмент с наклон < -5%
    seg_df["prev_downhill"] = seg_df["downhill"].shift(1).fillna(False)

    # Тук вече прилагаме критерия:
    # - сегментът да е downhill
    # - предходният също да е downhill
    seg_df = seg_df[seg_df["downhill"] & seg_df["prev_downhill"]].copy()

    if seg_df.empty:
        return seg_df

    # ΔV / наклон% (използваме абсолютната стойност на наклона,
    # за да имаме по-интуитивен положителен коефициент)
    seg_df["dv_per_slope_abs"] = seg_df["delta_v_kmh"] / seg_df["mean_slope_percent"].abs()
    seg_df["dv_per_slope_signed"] = seg_df["delta_v_kmh"] / seg_df["mean_slope_percent"]

    return seg_df


def fit_criterion_model(segments: pd.DataFrame):
    """
    Строи проста линейна регресия:
    dv_per_slope_abs ≈ b0 + b1 * mean_speed_kmh + b2 * mean_slope_percent

    Връща:
    - segments с колона 'criterion_factor' (остатък)
    - речник с коефициенти.
    """
    if segments.shape[0] < 3:
        return segments, None

    # Подготвяме матрицата за линейна регресия
    X = np.column_stack(
        [
            np.ones(len(segments)),
            segments["mean_speed_kmh"].values,
            segments["mean_slope_percent"].values,
        ]
    )
    y = segments["dv_per_slope_abs"].values

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    residual = y - y_pred

    segments = segments.copy()
    segments["criterion_factor"] = residual

    coeffs = {
        "b0": beta[0],
        "b_speed": beta[1],
        "b_slope": beta[2],
    }

    return segments, coeffs


# ------------------------
# STREAMLIT UI
# ------------------------

st.title("Ski Glide Dynamics – нов модел")

st.markdown(
    """
Моделът:
1. **Чисти артефакти** от суровите данни (денивелация и скорост).
2. **Реже** цялата активност на 10-секундни сегменти.
3. Избира само сегменти, които:
   - са **спускане** със среден наклон под -5%;
   - са **предхождани** от друг спускащ сегмент (< -5%);
   - имат **само намаляващи стойности на височината** (чисто спускане).
4. От всички валидни сегменти (от всички активности) изчислява:
   - средна скорост,
   - среден наклон,
   - ΔV (крайна - начална скорост),
   - ΔV / наклон%,
   - и прост **критериен фактор** за „плъзгаемост“.
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

        seg_df = extract_segments(df, SEGMENT_LENGTH_SEC)

        if seg_df.empty:
            st.warning("Няма сегменти, които да покриват критериите за този файл.")
            continue

        seg_df["file_name"] = f.name
        all_segments.append(seg_df)

    if not all_segments:
        st.error("Няма нито един сегмент, преминал критериите, за всички файлове.")
    else:
        segments = pd.concat(all_segments, ignore_index=True)

        st.subheader("Таблица с всички валидни сегменти (от всички активности)")
        st.dataframe(
            segments[
                [
                    "file_name",
                    "seg_idx",
                    "t_start",
                    "t_end",
                    "duration_s",
                    "distance_m",
                    "mean_speed_kmh",
                    "mean_slope_percent",
                    "v_start_kmh",
                    "v_end_kmh",
                    "delta_v_kmh",
                    "dv_per_slope_abs",
                ]
            ]
        )

        # Обобщени средни стойности
        st.subheader("Средни стойности на преминалите сегменти (всички файлове)")
        mean_speed = segments["mean_speed_kmh"].mean()
        mean_slope = segments["mean_slope_percent"].mean()
        mean_delta_v = segments["delta_v_kmh"].mean()
        mean_dv_per_slope = segments["dv_per_slope_abs"].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Средна скорост на сегментите (km/h)", f"{mean_speed:.2f}")
            st.metric("Средна ΔV (крайна - начална скорост, km/h)", f"{mean_delta_v:.2f}")
        with col2:
            st.metric("Среден наклон (%)", f"{mean_slope:.2f}")
            st.metric("Средно ΔV / |наклон|", f"{mean_dv_per_slope:.4f}")

        # Фитваме критериен модел
        segments, coeffs = fit_criterion_model(segments)

        if coeffs is not None:
            st.subheader("Критериен модел за ΔV/наклон%")
            st.markdown(
                f"""
Модел (линейна регресия):

\\[
\\frac{{\\Delta V}}{{|slope|}} \\approx
b_0 + b_{{speed}} \\cdot V_{{mean}} + b_{{slope}} \\cdot slope
\\]

където:
- \\(V_{{mean}}\\) – средна скорост (km/h),
- \\(slope\\) – среден наклон (%).

Коефициенти:
- **b0** = {coeffs['b0']:.4f}
- **b_speed** = {coeffs['b_speed']:.4f}
- **b_slope** = {coeffs['b_slope']:.4f}

Остатъкът (реално - предсказано) е запазен като **criterion_factor** –
той може да се ползва като индикатор за *плъзгаемост* (положителен → по-добро плъзгане
от очакваното за дадена скорост и наклон).
"""
            )

            st.subheader("Разпределение на критериения фактор")
            st.write(
                segments[["file_name", "seg_idx", "mean_speed_kmh", "mean_slope_percent", "dv_per_slope_abs", "criterion_factor"]]
            )

        # 3D визуализация
        st.subheader("3D зависимост: скорост – наклон – ΔV/наклон%")

        fig = px.scatter_3d(
            segments,
            x="mean_speed_kmh",
            y="mean_slope_percent",
            z="dv_per_slope_abs",
            color="criterion_factor" if "criterion_factor" in segments.columns else None,
            hover_data=["file_name", "seg_idx", "delta_v_kmh"],
            labels={
                "mean_speed_kmh": "Средна скорост (km/h)",
                "mean_slope_percent": "Среден наклон (%)",
                "dv_per_slope_abs": "ΔV / |наклон|",
                "criterion_factor": "Критериен фактор",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        # Възможност за сваляне на резултата
        csv_bytes = segments.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Свали всички сегменти като CSV",
            data=csv_bytes,
            file_name="ski_glide_segments.csv",
            mime="text/csv",
        )
else:
    st.info("Качи поне един TCX файл, за да започнем анализа.")
