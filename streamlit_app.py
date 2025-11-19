import io
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ---------- 1) ПАРСВАНЕ НА TCX В DATAFRAME ----------

def parse_tcx_to_df(file) -> pd.DataFrame:
    """
    Чете TCX файл и връща DataFrame с:
    time (datetime), latitude, longitude, altitude (m), distance (m).
    Ресемплира на 1 Hz.
    """
    content = file.read()
    if isinstance(content, bytes):
        tree = ET.parse(io.BytesIO(content))
    else:
        tree = ET.parse(file)

    root = tree.getroot()
    trackpoints = root.findall('.//{*}Trackpoint')

    records = []
    for tp in trackpoints:
        time_el = tp.find('.//{*}Time')
        alt_el = tp.find('.//{*}AltitudeMeters')
        dist_el = tp.find('.//{*}DistanceMeters')
        lat_el = tp.find('.//{*}Position/{*}LatitudeDegrees')
        lon_el = tp.find('.//{*}Position/{*}LongitudeDegrees')

        if time_el is None or dist_el is None or alt_el is None:
            continue

        try:
            t = pd.to_datetime(time_el.text)
        except Exception:
            continue

        try:
            altitude = float(alt_el.text)
        except Exception:
            altitude = np.nan

        try:
            distance = float(dist_el.text)
        except Exception:
            distance = np.nan

        lat = float(lat_el.text) if lat_el is not None else np.nan
        lon = float(lon_el.text) if lon_el is not None else np.nan

        records.append(
            {
                "time": t,
                "latitude": lat,
                "longitude": lon,
                "altitude": altitude,
                "distance": distance,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values("time").drop_duplicates(subset=["time"])
    df = df.set_index("time")

    # Ресемплираме на 1 Hz
    df = df.resample("1S").nearest(limit=1)

    # Интерполация на altitude и distance по време
    df["altitude"] = df["altitude"].interpolate(method="time")
    df["distance"] = df["distance"].interpolate(method="time")

    df = df.reset_index()
    return df


# ---------- 2) ФУНКЦИЯ ЗА ФИЛТРИРАНЕ И ИЗВЛИЧАНЕ НА GLIDE-СЕГМЕНТИ ----------

def extract_glide_segments(
    df: pd.DataFrame,
    segment_length_s: int = 10,
    min_slope_pct: float = -5.0,
    prev_min_slope_pct: float = -5.0,
    min_dh_P: float = 0.8,
    min_dh_S: float = 1.0,
    min_dist_P: float = 20.0,
    min_dist_S: float = 30.0,
    min_speed_S_kmh: float = 10.0,
    max_cv_v_S: float = 0.25,
    eps_h: float = 0.25,
    max_slope_diff_pct_points: float = 7.0,
    v_max_kmh: float = 70.0,
    alt_grad_thresh: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Приема DataFrame с колони: time, altitude, distance.
    Връща:
      - DataFrame с намерените валидни glide-сегменти:
        колони: start_time, end_time, mean_speed_kmh, mean_slope_pct, dh_S, dist_S
      - stats dict с броячи за debug.
    """

    stats = {
        "total_windows": 0,
        "pass_P": 0,
        "pass_S_basic": 0,
        "pass_monotonic": 0,
        "pass_slope_diff": 0,
        "final_segments": 0,
    }

    if df.empty:
        return pd.DataFrame(), stats

    # Нужни са поне 2*L + 1 точки
    if len(df) < 2 * segment_length_s + 1:
        return pd.DataFrame(), stats

    df = df.copy()

    # --- 2.1 Почистване на очевидни артефакти и изглаждане на височината ---

    df["dt"] = df["time"].diff().dt.total_seconds().fillna(1.0)
    df.loc[df["dt"] <= 0, "dt"] = 1.0  # след ресемплиране би трябвало да е 1

    # Вертикален градиент (преди изглаждане)
    df["dalt"] = df["altitude"].diff()
    df["alt_grad"] = df["dalt"].abs() / df["dt"]

    # Премахваме точки с огромен вертикален градиент
    df.loc[df["alt_grad"] > alt_grad_thresh, "altitude"] = np.nan
    df["altitude"] = df["altitude"].interpolate(limit_direction="both")

    # Първо леко изглаждане (moving median, 3 точки)
    df["alt_smooth"] = (
        df["altitude"].rolling(window=3, center=True).median().fillna(df["altitude"])
    )

    # По-силен тренд филтър (rolling mean, 15 точки)
    df["alt_trend"] = (
        df["alt_smooth"]
        .rolling(window=15, center=True, min_periods=1)
        .mean()
    )

    # Скорост от дистанция
    df["distance"] = df["distance"].interpolate(limit_direction="both")
    df["ddist"] = df["distance"].diff()
    df["speed_m_s"] = (df["ddist"] / df["dt"]).fillna(0.0)

    # Премахваме нереалистично високи скорости
    v_max = v_max_kmh / 3.6
    df.loc[df["speed_m_s"].abs() > v_max, "speed_m_s"] = np.nan
    df["speed_m_s"] = df["speed_m_s"].interpolate(limit_direction="both").fillna(0.0)

    df = df.reset_index(drop=True)

    # Нека използваме alt_trend за наклоните и монотонността
    alt = df["alt_trend"].values
    dist = df["distance"].values
    v = df["speed_m_s"].values
    times = df["time"].values

    N = len(df)
    L = int(segment_length_s)

    min_slope = min_slope_pct / 100.0  # от % към десетична форма
    prev_min_slope = prev_min_slope_pct / 100.0
    max_slope_diff = max_slope_diff_pct_points / 100.0

    glide_segments: List[Dict] = []

    # --- 2.2 Плъзгащ се прозорец: P = [k-L, k), S = [k, k+L) ---
    for k in range(L, N - L):
        stats["total_windows"] += 1

        P_start = k - L
        P_end = k  # exclusive
        S_start = k
        S_end = k + L

        # Проверка за NaN в ключовите полета
        if (
            np.isnan(alt[P_start])
            or np.isnan(alt[S_start])
            or np.isnan(dist[P_start])
            or np.isnan(dist[S_start])
        ):
            continue

        # --- Метрики за P ---
        dh_P = alt[P_start] - alt[P_end - 1]
        dist_P = dist[P_end - 1] - dist[P_start]
        if dist_P <= 0:
            continue

        mean_slope_P = dh_P / dist_P
        mean_speed_P = np.nanmean(v[P_start:P_end])

        if dh_P < min_dh_P:
            continue
        if dist_P < min_dist_P:
            continue
        if mean_slope_P > prev_min_slope:  # трябва да е <= min_slope_P (отрицателно)
            continue
        if np.isnan(mean_speed_P):
            continue

        stats["pass_P"] += 1

        # --- Метрики за S ---
        dh_S = alt[S_start] - alt[S_end - 1]
        dist_S = dist[S_end - 1] - dist[S_start]
        if dist_S <= 0:
            continue

        mean_slope_S = dh_S / dist_S
        mean_speed_S = np.nanmean(v[S_start:S_end])
        if np.isnan(mean_speed_S) or mean_speed_S <= 0:
            continue

        mean_speed_S_kmh = mean_speed_S * 3.6

        std_v_S = np.nanstd(v[S_start:S_end])
        if mean_speed_S > 0:
            cv_v_S = std_v_S / mean_speed_S
        else:
            cv_v_S = np.inf

        # Базови условия за S
        if dh_S < min_dh_S:
            continue
        if dist_S < min_dist_S:
            continue
        if mean_slope_S > min_slope:  # трябва да е <= -5%
            continue
        if mean_speed_S_kmh < min_speed_S_kmh:
            continue
        if cv_v_S > max_cv_v_S:
            continue

        stats["pass_S_basic"] += 1

        # Монотонно (почти) спускане в S: alt[i+1] <= alt[i] + eps_h
        alts_S = alt[S_start:S_end]
        if np.any(np.diff(alts_S) > eps_h):
            continue

        stats["pass_monotonic"] += 1

        # Сходство на наклона между P и S
        if abs(mean_slope_S - mean_slope_P) > max_slope_diff:
            continue

        stats["pass_slope_diff"] += 1

        # Ако стигнем дотук, сегментът е валиден glide-сегмент
        segment_info = {
            "start_time": times[S_start],
            "end_time": times[S_end - 1],
            "mean_speed_kmh": mean_speed_S_kmh,
            "mean_slope_pct": mean_slope_S * 100.0,
            "dh_S": dh_S,
            "dist_S": dist_S,
        }
        glide_segments.append(segment_info)
        stats["final_segments"] += 1

    if not glide_segments:
        return pd.DataFrame(), stats

    seg_df = pd.DataFrame(glide_segments)
    return seg_df, stats


# ---------- 3) STREAMLIT UI ----------

def main():
    st.title("Ski-Glide-Dynamics — Glide Segment Analyzer (10 s / -5%)")

    st.write(
        "Зареди една или повече **TCX активности**. "
        "Моделът ще намери 10-секундни спускания, които отговарят на "
        "критериите за чисто плъзгане, и ще изчисли средна скорост и наклон."
    )

    # --- Sidebar: настройки на модела ---
    st.sidebar.header("Настройки на модела")

    segment_length_s = st.sidebar.number_input(
        "Дължина на сегмента (s)", min_value=5, max_value=30, value=10, step=1
    )

    min_slope_pct = st.sidebar.number_input(
        "Мин. среден наклон S (в %; напр. -5)",
        min_value=-30.0,
        max_value=-1.0,
        value=-5.0,
        step=0.5,
    )

    prev_min_slope_pct = st.sidebar.number_input(
        "Мин. среден наклон P (в %; напр. -5)",
        min_value=-30.0,
        max_value=-1.0,
        value=-5.0,
        step=0.5,
    )

    min_dh_P = st.sidebar.number_input(
        "Мин. денивелация в P (m)", min_value=0.0, max_value=20.0, value=0.8, step=0.1
    )
    min_dh_S = st.sidebar.number_input(
        "Мин. денивелация в S (m)", min_value=0.0, max_value=20.0, value=1.0, step=0.1
    )

    min_dist_P = st.sidebar.number_input(
        "Мин. хор. дистанция в P (m)", min_value=0.0, max_value=200.0, value=20.0, step=5.0
    )
    min_dist_S = st.sidebar.number_input(
        "Мин. хор. дистанция в S (m)", min_value=0.0, max_value=200.0, value=30.0, step=5.0
    )

    min_speed_S_kmh = st.sidebar.number_input(
        "Мин. средна скорост S (km/h)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
    )

    max_cv_v_S = st.sidebar.number_input(
        "Макс. CV на скоростта в S",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
    )

    eps_h = st.sidebar.number_input(
        "Допустимо покачване на alt в S (ε_h, m)",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.01,
    )

    max_slope_diff_pct_points = st.sidebar.number_input(
        "Макс. разлика между slope_P и slope_S (в процентни пункта)",
        min_value=0.0,
        max_value=20.0,
        value=7.0,
        step=0.5,
    )

    v_max_kmh = st.sidebar.number_input(
        "Макс. допустима скорост (km/h)",
        min_value=10.0,
        max_value=120.0,
        value=70.0,
        step=5.0,
    )

    alt_grad_thresh = st.sidebar.number_input(
        "Макс. вертикален градиент (m/s)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

    debug_mode = st.sidebar.checkbox("Показвай debug информация", value=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Може да започнем с тези стойности и после да ги прецизираме според резултатите."
    )

    # --- Качване на файлове ---
    uploaded_files = st.file_uploader(
        "Качи TCX файлове (може няколко наведнъж)", type=["tcx"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Моля, качи поне един TCX файл.")
        return

    summary_rows = []
    all_segments = []

    first_alt_plot_done = False

    for file in uploaded_files:
        st.subheader(f"Активност: {file.name}")

        df = parse_tcx_to_df(file)
        if df.empty:
            st.warning("Не успях да прочета валидни данни от този файл.")
            continue

        # Показваме altitude vs alt_trend за първата активност (за усещане)
        if not first_alt_plot_done:
            tmp_df = df.copy()
            # Тук повтаряме само филтрацията за alt_trend за визуализация
            tmp_df["dt"] = tmp_df["time"].diff().dt.total_seconds().fillna(1.0)
            tmp_df.loc[tmp_df["dt"] <= 0, "dt"] = 1.0
            tmp_df["dalt"] = tmp_df["altitude"].diff()
            tmp_df["alt_grad"] = tmp_df["dalt"].abs() / tmp_df["dt"]
            tmp_df.loc[tmp_df["alt_grad"] > alt_grad_thresh, "altitude"] = np.nan
            tmp_df["altitude"] = tmp_df["altitude"].interpolate(limit_direction="both")
            tmp_df["alt_smooth"] = (
                tmp_df["altitude"].rolling(window=3, center=True).median().fillna(tmp_df["altitude"])
            )
            tmp_df["alt_trend"] = (
                tmp_df["alt_smooth"]
                .rolling(window=15, center=True, min_periods=1)
                .mean()
            )
            st.markdown("**Примерна графика на altitude (raw) и alt_trend за първата активност:**")
            plot_df = tmp_df[["time", "altitude", "alt_trend"]].set_index("time")
            st.line_chart(plot_df)
            first_alt_plot_done = True

        seg_df, stats = extract_glide_segments(
            df,
            segment_length_s=segment_length_s,
            min_slope_pct=min_slope_pct,
            prev_min_slope_pct=prev_min_slope_pct,
            min_dh_P=min_dh_P,
            min_dh_S=min_dh_S,
            min_dist_P=min_dist_P,
            min_dist_S=min_dist_S,
            min_speed_S_kmh=min_speed_S_kmh,
            max_cv_v_S=max_cv_v_S,
            eps_h=eps_h,
            max_slope_diff_pct_points=max_slope_diff_pct_points,
            v_max_kmh=v_max_kmh,
            alt_grad_thresh=alt_grad_thresh,
        )

        if seg_df.empty:
            st.write("❌ Няма намерени сегменти, които да отговарят на условията.")
            if debug_mode:
                with st.expander("Debug за тази активност"):
                    st.json(stats)
            continue

        seg_df["activity"] = file.name
        all_segments.append(seg_df)

        mean_speed = seg_df["mean_speed_kmh"].mean()
        mean_slope = seg_df["mean_slope_pct"].mean()
        n_segments = len(seg_df)

        st.write(
            f"✅ Намерени сегменти: **{n_segments}**  \n"
            f"Средна скорост на валидните сегменти: **{mean_speed:.2f} km/h**  \n"
            f"Среден наклон на валидните сегменти: **{mean_slope:.2f} %**"
        )

        if debug_mode:
            with st.expander("Debug за тази активност"):
                st.json(stats)

        summary_rows.append(
            {
                "activity": file.name,
                "n_segments": n_segments,
                "mean_speed_kmh": mean_speed,
                "mean_slope_pct": mean_slope,
            }
        )

    if summary_rows:
        st.markdown("---")
        st.subheader("Обобщение по активности")

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    if all_segments:
        st.markdown("---")
        st.subheader("Всички валидни сегменти (подробно)")

        all_seg_df = pd.concat(all_segments, ignore_index=True)
        all_seg_df["mean_speed_kmh"] = all_seg_df["mean_speed_kmh"].round(2)
        all_seg_df["mean_slope_pct"] = all_seg_df["mean_slope_pct"].round(2)
        all_seg_df["dh_S"] = all_seg_df["dh_S"].round(2)
        all_seg_df["dist_S"] = all_seg_df["dist_S"].round(2)

        st.dataframe(all_seg_df, use_container_width=True)


if __name__ == "__main__":
    main()
