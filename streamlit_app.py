import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO

# ------------------------
# ПАРСВАНЕ НА TCX
# ------------------------

def parse_tcx(file) -> pd.DataFrame:
    file_bytes = file.read()
    tree = ET.parse(BytesIO(file_bytes))
    root = tree.getroot()

    trackpoints = []
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        time_el = dist_el = alt_el = None

        for child in tp:
            if child.tag.endswith("Time"):
                time_el = child
            elif child.tag.endswith("DistanceMeters"):
                dist_el = child
            elif child.tag.endswith("AltitudeMeters"):
                alt_el = child

        if time_el is None:
            continue

        trackpoints.append({
            "time": pd.to_datetime(time_el.text),
            "distance_m": float(dist_el.text) if dist_el is not None else np.nan,
            "altitude_m": float(alt_el.text) if alt_el is not None else np.nan
        })

    df = (
        pd.DataFrame(trackpoints)
        .dropna(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return df


# ------------------------
# СГЛАЖДАНЕ НА ВИСОЧИНАТА
# ------------------------

def smooth_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["altitude_m"] = df["altitude_m"].rolling(window=3, center=True).median()
    return df


# ------------------------
# ЧИСТЕНЕ НА АРТЕФАКТИ
# ------------------------

def clean_artifacts(
    df: pd.DataFrame,
    max_speed_m_s: float,
    max_alt_rate_m_s: float
) -> pd.DataFrame:
    df = df.copy()
    df["dt"] = df["time"].diff().dt.total_seconds()
    df["ddist"] = df["distance_m"].diff()
    df["dalt"] = df["altitude_m"].diff()

    speed_m_s = df["ddist"] / df["dt"]
    alt_rate = df["dalt"] / df["dt"]

    mask_valid = True
    mask_valid &= (df["dt"].isna() | (df["dt"] > 0))
    mask_valid &= (df["ddist"].isna() | (df["ddist"] >= 0))
    mask_valid &= (speed_m_s.isna() | ((speed_m_s >= 0) & (speed_m_s <= max_speed_m_s)))
    mask_valid &= (alt_rate.isna() | (abs(alt_rate) <= max_alt_rate_m_s))

    df = df[mask_valid].copy().reset_index(drop=True)
    df = df[["time", "distance_m", "altitude_m"]]
    return df


# ------------------------
# СЕГМЕНТИРАНЕ В 15s БЛОКОВЕ
# ------------------------
def segment_activity(
    df: pd.DataFrame,
    segment_length_sec: int,
    min_segment_duration: float,
    min_segment_distance_m: float,
    min_abs_delta_elev: float,
    min_segment_speed_kmh: float,
    max_abs_slope_percent: float
) -> pd.DataFrame:
    """
    Връща DataFrame със сегменти и базови метрики.
    Ако няма нито един валиден сегмент, връща празен DF без да хвърля KeyError.
    """
    df = df.copy()
    df["elapsed_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["segment_idx"] = (df["elapsed_s"] // segment_length_sec).astype(int)

    rows = []
    for seg_idx, g in df.groupby("segment_idx"):
        if len(g) < 2:
            continue

        t_start, t_end = g["time"].iloc[0], g["time"].iloc[-1]
        duration_s = (t_end - t_start).total_seconds()
        if duration_s < min_segment_duration:
            continue

        dist_start, dist_end = g["distance_m"].iloc[0], g["distance_m"].iloc[-1]
        alt_start, alt_end = g["altitude_m"].iloc[0], g["altitude_m"].iloc[-1]

        segment_distance_m = dist_end - dist_start
        delta_elev_m = alt_end - alt_start

        if segment_distance_m <= 0:
            continue
        if segment_distance_m < min_segment_distance_m:
            continue
        if abs(delta_elev_m) < min_abs_delta_elev:
            continue  # твърде малък Δh → шум

        avg_speed_m_s = segment_distance_m / duration_s
        avg_speed_kmh = avg_speed_m_s * 3.6

        if avg_speed_kmh < min_segment_speed_kmh:
            continue

        slope_percent = (delta_elev_m / segment_distance_m) * 100
        if abs(slope_percent) > max_abs_slope_percent:
            continue

        rows.append({
            "segment_idx": seg_idx,
            "t_start": t_start,
            "t_end": t_end,
            "duration_s": duration_s,
            "segment_distance_m": segment_distance_m,
            "delta_elev_m": delta_elev_m,
            "avg_speed_kmh": avg_speed_kmh,
            "slope_percent": slope_percent,
            "idx_start": g.index[0],
            "idx_end": g.index[-1],
        })

    seg_df = pd.DataFrame(rows)

    # 🔧 FIX: ако няма нито един валиден сегмент, не сортираме по column, която я няма
    if seg_df.empty:
        return seg_df

    return seg_df.sort_values("segment_idx").reset_index(drop=True)

def filter_glide_segments(
    seg_df: pd.DataFrame,
    df_clean: pd.DataFrame,
    min_downhill_slope: float,
    eps_h: float
) -> pd.DataFrame:
    """
    Взима сегментите от segment_activity и:
      - изисква slope_percent <= min_downhill_slope;
      - за всеки такъв сегмент гледа предходния (segment_idx - 1),
        който също трябва да има slope_percent <= min_downhill_slope;
      - проверява в рамките на сегмента височината да намалява
        (alt[i+1] <= alt[i] + eps_h).
    Връща само сегментите, които изпълняват всички условия.
    """
    if seg_df.empty:
        return seg_df

    seg_df = seg_df.copy()
    seg_df.set_index("segment_idx", inplace=True)

    valid_rows = []

    for idx, row in seg_df.iterrows():
        slope = row["slope_percent"]
        if slope > min_downhill_slope:
            continue  # не е достатъчно стръмно спускане

        # предходният сегмент
        prev_idx = idx - 1
        if prev_idx not in seg_df.index:
            continue

        prev_slope = seg_df.loc[prev_idx, "slope_percent"]
        if prev_slope > min_downhill_slope:
            continue  # предходният не е достатъчно стръмен

        # монотонна (почти) низходяща височина в рамките на сегмента
        i_start = int(row["idx_start"])
        i_end = int(row["idx_end"])
        alts = df_clean.loc[i_start:i_end, "altitude_m"].values

        # позволяваме малко покачване eps_h
        if np.any(np.diff(alts) > eps_h):
            continue

        valid_rows.append(row)

    if not valid_rows:
        return pd.DataFrame(columns=seg_df.reset_index().columns)

    glide_df = pd.DataFrame(valid_rows).reset_index().rename(columns={"segment_idx": "segment_idx"})
    return glide_df


# ------------------------
# STREAMLIT APP
# ------------------------

def main():
    st.title("Ski-Glide-Dynamics — сегменти + допълнителни условия")

    st.write(
        "Качи една или няколко **TCX активности**. "
        "Първо прилагаме стария модел за филтриране и 15 s сегментиране, "
        "след това върху тези сегменти прилагаме допълнителните условия "
        "за спускане (предхождащ сегмент и низходяща височина)."
    )

    # ---- Sidebar настройки ----
    st.sidebar.header("Основни настройки")

    segment_length_sec = st.sidebar.number_input(
        "Дължина на сегмента (s)", min_value=5, max_value=60, value=15, step=1
    )
    min_segment_duration = st.sidebar.number_input(
        "Мин. реална продължителност на сегмента (s)",
        min_value=1.0, max_value=60.0, value=10.0, step=1.0
    )
    min_segment_distance_m = st.sidebar.number_input(
        "Мин. хоризонтална дистанция (m)",
        min_value=0.0, max_value=500.0, value=20.0, step=5.0
    )
    min_segment_speed_kmh = st.sidebar.number_input(
        "Мин. средна скорост на сегмента (km/h)",
        min_value=0.0, max_value=80.0, value=10.0, step=1.0
    )
    max_abs_slope_percent = st.sidebar.number_input(
        "Макс. абсолютен наклон на сегмента (%)",
        min_value=0.0, max_value=100.0, value=30.0, step=1.0
    )
    min_abs_delta_elev = st.sidebar.number_input(
        "Мин. |Δh| в сегмента (m) за да не е шум",
        min_value=0.0, max_value=10.0, value=0.3, step=0.1
    )

    st.sidebar.header("Филтри на сурови данни")
    max_speed_m_s = st.sidebar.number_input(
        "Макс. скорост (m/s)", min_value=1.0, max_value=50.0, value=30.0, step=1.0
    )
    max_alt_rate_m_s = st.sidebar.number_input(
        "Макс. вертикален градиент |dalt/dt| (m/s)",
        min_value=0.5, max_value=20.0, value=5.0, step=0.5
    )

    st.sidebar.header("Допълнителни glide-условия")
    min_downhill_slope = st.sidebar.number_input(
        "Мин. наклон за спускане (%) (напр. -5)",
        min_value=-50.0, max_value=-0.1, value=-5.0, step=0.5
    )
    eps_h = st.sidebar.number_input(
        "Допустимо локално покачване на височината в сегмента (ε, m)",
        min_value=0.0, max_value=1.0, value=0.1, step=0.01
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Можеш да пипаш параметрите според това колко сегменти излизат.")

    # ---- Качване на файлове ----
    uploaded_files = st.file_uploader(
        "Качи TCX файлове (може няколко)", type=["tcx"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Моля, качи поне един TCX файл.")
        return

    summary_rows = []

    for file in uploaded_files:
        st.subheader(f"Активност: {file.name}")

        df_raw = parse_tcx(file)
        if df_raw.empty:
            st.warning("Не успях да прочета валидни данни от този файл.")
            continue

        df_smooth = smooth_altitude(df_raw)
        df_clean = clean_artifacts(df_smooth, max_speed_m_s, max_alt_rate_m_s)

        seg_df = segment_activity(
            df_clean,
            segment_length_sec=segment_length_sec,
            min_segment_duration=min_segment_duration,
            min_segment_distance_m=min_segment_distance_m,
            min_abs_delta_elev=min_abs_delta_elev,
            min_segment_speed_kmh=min_segment_speed_kmh,
            max_abs_slope_percent=max_abs_slope_percent,
        )

        if seg_df.empty:
            st.error("Няма валидни сегменти при тези базови филтри.")
            continue

        glide_df = filter_glide_segments(
            seg_df, df_clean,
            min_downhill_slope=min_downhill_slope,
            eps_h=eps_h
        )

        st.write(f"Общ брой базови сегменти: **{len(seg_df)}**")
        if glide_df.empty:
            st.warning("❌ Няма сегменти, които да изпълняват допълнителните glide-условия.")
            if st.checkbox(f"Покажи базовите сегменти ({file.name})", key=f"base_{file.name}"):
                st.dataframe(seg_df.head(50))
            continue

        n_glide = len(glide_df)
        mean_speed = glide_df["avg_speed_kmh"].mean()
        mean_slope = glide_df["slope_percent"].mean()

        st.success(
            f"✅ Glide-сегменти: **{n_glide}**  \n"
            f"Средна скорост на glide-сегментите: **{mean_speed:.2f} km/h**  \n"
            f"Среден наклон на glide-сегментите: **{mean_slope:.2f} %**"
        )

        if st.checkbox(f"Покажи glide-сегментите ({file.name})", key=f"glide_{file.name}"):
            st.dataframe(glide_df[[
                "segment_idx", "t_start", "t_end",
                "duration_s", "segment_distance_m",
                "delta_elev_m", "avg_speed_kmh", "slope_percent"
            ]].head(100))

        summary_rows.append({
            "activity": file.name,
            "n_glide_segments": n_glide,
            "mean_speed_kmh": mean_speed,
            "mean_slope_percent": mean_slope,
        })

    if summary_rows:
        st.markdown("---")
        st.subheader("Обобщение по активности (само glide-сегментите)")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
