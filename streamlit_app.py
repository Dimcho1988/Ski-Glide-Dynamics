import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ---- НАСТРОЙКИ ----
SEGMENT_LENGTH_SEC = 10          # дължина на сегмента в секунди
MIN_DOWNHILL_SLOPE = -5.0        # % наклон, под който броим сегмента (напр. -5% = поне 5% спускане)
MIN_SEGMENT_DURATION = 8.0       # минимални реални секунди в сегмента, за да го считаме за "стабилен"
MAX_SPEED_M_S = 30.0             # груб филтър за артефакти (~108 km/h)
MAX_ALT_RATE_M_S = 50.0          # филтър за ненормални скокове във височината


# ---- ПАРСВАНЕ НА TCX ----
def parse_tcx(file) -> pd.DataFrame:
    """
    Чете TCX файл и връща DataFrame с колони:
    time (datetime64), distance_m (float), altitude_m (float)
    """
    # за да може ET.parse да работи повече от веднъж, правим копие
    file_bytes = file.read()
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    # Събираме всички Trackpoint елементи, без да се занимаваме с namespace
    trackpoints = []
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        time_el = None
        dist_el = None
        alt_el = None

        for child in tp:
            if child.tag.endswith("Time"):
                time_el = child
            elif child.tag.endswith("DistanceMeters"):
                dist_el = child
            elif child.tag.endswith("AltitudeMeters"):
                alt_el = child

        if time_el is None:
            continue  # без време няма как да използваме точката

        time = pd.to_datetime(time_el.text)

        distance_m = float(dist_el.text) if dist_el is not None else np.nan
        altitude_m = float(alt_el.text) if alt_el is not None else np.nan

        trackpoints.append(
            {
                "time": time,
                "distance_m": distance_m,
                "altitude_m": altitude_m,
            }
        )

    if not trackpoints:
        raise ValueError("Не бяха намерени Trackpoint данни в TCX файла.")

    df = pd.DataFrame(trackpoints).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Нормализираме, ако има липсващи разстояния или височини
    # (тук просто оставяме NaN – по-нататък ще ги игнорираме при slope)
    return df


# ---- ЧИСТЕНЕ НА АРТЕФАКТИ ----
def clean_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Премахва очевидно невалидни точки:
    - отрицателен или нулев времеви интервал
    - отрицателна промяна в дистанцията
    - нереалистично висока скорост
    - нереалистичен вертикален темп (скок във височината)
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["dt"] = df["time"].diff().dt.total_seconds()
    df["ddist"] = df["distance_m"].diff()
    df["dalt"] = df["altitude_m"].diff()

    # Базови филтри
    # първият ред ще е с NaN, ще го запазим отделно
    mask_valid = True

    # dt трябва да е положително
    mask_valid = mask_valid & ((df["dt"].isna()) | (df["dt"] > 0))

    # не допускаме назад във времето или нулево dt (освен първия ред)
    # ddist >= 0 (не допускаме да "намалява" дистанцията)
    mask_valid = mask_valid & ((df["ddist"].isna()) | (df["ddist"] >= 0))

    # скорост = ddist / dt
    speed_m_s = df["ddist"] / df["dt"]
    # нереалистично високи скорости и отрицателни скорости
    mask_valid = mask_valid & ((speed_m_s.isna()) | ((speed_m_s >= 0) & (speed_m_s <= MAX_SPEED_M_S)))

    # скорост на промяна на височината
    alt_rate_m_s = df["dalt"] / df["dt"]
    mask_valid = mask_valid & (
        (alt_rate_m_s.isna()) | (alt_rate_m_s.abs() <= MAX_ALT_RATE_M_S)
    )

    # Пазим само валидните точки
    df_clean = df[mask_valid].copy().reset_index(drop=True)

    # Премахваме помощните колони
    df_clean = df_clean[["time", "distance_m", "altitude_m"]]

    return df_clean


# ---- СЕГМЕНТИРАНЕ В 10-СЕК ИНТЕРВАЛИ ----
def segment_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Сегментира цялата активност на 10-секундни интервали.
    За всеки сегмент изчислява:
      - duration_s
      - segment_distance_m
      - delta_elev_m
      - avg_speed_kmh
      - slope_percent
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    start_time = df["time"].min()
    df["elapsed_s"] = (df["time"] - start_time).dt.total_seconds()

    # Индекс на сегмента: 0,1,2... на всеки 10 секунди
    df["segment_idx"] = (df["elapsed_s"] // SEGMENT_LENGTH_SEC).astype(int)

    # Групираме по сегмент
    grouped = df.groupby("segment_idx")

    rows = []
    for seg_idx, g in grouped:
        if len(g) < 2:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration_s = (t_end - t_start).total_seconds()

        dist_start = g["distance_m"].iloc[0]
        dist_end = g["distance_m"].iloc[-1]
        segment_distance_m = dist_end - dist_start

        alt_start = g["altitude_m"].iloc[0]
        alt_end = g["altitude_m"].iloc[-1]
        delta_elev_m = alt_end - alt_start

        # филтри за стабилност
        if duration_s < MIN_SEGMENT_DURATION:
            continue
        if segment_distance_m <= 0:
            continue
        if pd.isna(alt_start) or pd.isna(alt_end):
            continue

        avg_speed_m_s = segment_distance_m / duration_s
        avg_speed_kmh = avg_speed_m_s * 3.6

        slope_percent = (delta_elev_m / segment_distance_m) * 100.0

        rows.append(
            {
                "segment_idx": seg_idx,
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration_s,
                "segment_distance_m": segment_distance_m,
                "delta_elev_m": delta_elev_m,
                "avg_speed_kmh": avg_speed_kmh,
                "slope_percent": slope_percent,
            }
        )

    seg_df = pd.DataFrame(rows)
    if seg_df.empty:
        return seg_df

    seg_df = seg_df.sort_values("segment_idx").reset_index(drop=True)
    return seg_df


# ---- ФИЛТЪР ЗА СПУСКАНИЯ И АГРЕГАЦИЯ ----
def compute_downhill_sums(seg_df: pd.DataFrame):
    """
    Взима само сегментите със наклон < MIN_DOWNHILL_SLOPE
    и връща:
      - sum_speed_kmh
      - sum_slope_percent
      - avg_speed_kmh
      - avg_slope_percent
      - count_segments
    """
    downhill = seg_df[seg_df["slope_percent"] <= MIN_DOWNHILL_SLOPE].copy()

    if downhill.empty:
        return {
            "count_segments": 0,
            "sum_speed_kmh": 0.0,
            "sum_slope_percent": 0.0,
            "avg_speed_kmh": 0.0,
            "avg_slope_percent": 0.0,
            "downhill_df": downhill,
        }

    sum_speed_kmh = downhill["avg_speed_kmh"].sum()
    sum_slope_percent = downhill["slope_percent"].sum()

    avg_speed_kmh = downhill["avg_speed_kmh"].mean()
    avg_slope_percent = downhill["slope_percent"].mean()

    return {
        "count_segments": len(downhill),
        "sum_speed_kmh": sum_speed_kmh,
        "sum_slope_percent": sum_slope_percent,
        "avg_speed_kmh": avg_speed_kmh,
        "avg_slope_percent": avg_slope_percent,
        "downhill_df": downhill,
    }


# ---- STREAMLIT UI ----
def main():
    st.title("Ski-Glide-Dynamics — 10 сек сегменти по наклон")

    st.write(
        """
        **Описание на модела:**

        1. Зареждаме TCX файл и премахваме артефакти (обратна дистанция, нереалистични скорости и скокове във височината).
        2. Сегментираме цялата активност на **10-секундни интервали**.
        3. За всеки сегмент изчисляваме:
           - средна скорост (km/h),
           - промяна във височината (м) от началото до края на сегмента,
           - изминат път в сегмента (м),
           - реален наклон в %.
        4. Вземаме **само сегментите**, където наклонът е под **-5%**.
        5. Накрая извеждаме:
           - сумата от скоростите в тези сегменти,
           - сумата от наклоните в % в тези сегменти.
        """
    )

    uploaded_file = st.file_uploader("Качи TCX файл", type=["tcx"])

    if uploaded_file is not None:
        try:
            df_raw = parse_tcx(uploaded_file)
        except Exception as e:
            st.error(f"Грешка при парсване на TCX файла: {e}")
            return

        st.success(f"Успешно зареден TCX с {len(df_raw)} точки.")
        if st.checkbox("Покажи суровите данни (първите 10 реда)"):
            st.dataframe(df_raw.head(10))

        # Чистене на артефакти
        df_clean = clean_artifacts(df_raw)
        st.write(f"След почистване останаха **{len(df_clean)}** точки.")

        # Сегментиране
        seg_df = segment_activity(df_clean)
        if seg_df.empty:
            st.error("Не бяха създадени валидни сегменти. Проверете данните.")
            return

        st.write(f"Общ брой сегменти (след филтри за стабилност): **{len(seg_df)}**")

        if st.checkbox("Покажи всички сегменти (първите 20):"):
            st.dataframe(seg_df.head(20))

        # Изчисление на сумите за спускане
        results = compute_downhill_sums(seg_df)

        st.subheader("Резултати за сегментите с наклон < -5%")

        st.write(f"Брой спускащи сегменти: **{results['count_segments']}**")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Сума от скоростите (km/h)", f"{results['sum_speed_kmh']:.2f}")
            st.metric("Средна скорост (km/h)", f"{results['avg_speed_kmh']:.2f}")
        with col2:
            st.metric("Сума от наклоните (%)", f"{results['sum_slope_percent']:.2f}")
            st.metric("Среден наклон (%)", f"{results['avg_slope_percent']:.2f}")

        if results["count_segments"] == 0:
            st.warning("Няма сегменти с наклон под -5%.")
        else:
            if st.checkbox("Покажи детайлите за downhill сегментите (първите 20):"):
                st.dataframe(results["downhill_df"].head(20))

    else:
        st.info("Моля, качи TCX файл, за да започнем анализа.")


if __name__ == "__main__":
    main()
