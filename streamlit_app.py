import io
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st

G = 9.81  # m/s^2


# ---------- TCX ПАРСЕР ----------

def parse_tcx(file_obj) -> pd.DataFrame:
    """
    Парсва TCX и връща DataFrame с:
    time, time_s, dt, altitude_m, distance_m, speed_m_s
    """
    data = {
        "time": [],
        "altitude_m": [],
        "distance_m": [],
        "speed_m_s": [],
    }

    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    file_obj = io.StringIO(content)

    tree = ET.parse(file_obj)
    root = tree.getroot()

    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
    else:
        ns_uri = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    ns = {"tcx": ns_uri}

    trackpoints = root.findall(".//tcx:Trackpoint", ns)
    if not trackpoints:
        trackpoints = root.findall(".//Trackpoint")

    for tp in trackpoints:
        t_el = tp.find(".//{*}Time")
        if t_el is None or t_el.text is None:
            continue
        try:
            time = pd.to_datetime(t_el.text)
        except Exception:
            continue

        alt_el = tp.find(".//{*}AltitudeMeters")
        altitude = float(alt_el.text) if alt_el is not None and alt_el.text is not None else np.nan

        dist_el = tp.find(".//{*}DistanceMeters")
        distance = float(dist_el.text) if dist_el is not None and dist_el.text is not None else np.nan

        speed = np.nan
        for tag in ["Speed", "ns3:Speed", "ns2:Speed"]:
            s_el = tp.find(f".//{{*}}{tag}")
            if s_el is not None and s_el.text is not None:
                try:
                    speed = float(s_el.text)
                    break
                except Exception:
                    pass

        data["time"].append(time)
        data["altitude_m"].append(altitude)
        data["distance_m"].append(distance)
        data["speed_m_s"].append(speed)

    df = pd.DataFrame(data)
    if df.empty:
        raise ValueError("Не бяха намерени валидни Trackpoint данни в TCX файла.")

    df = df.sort_values("time").reset_index(drop=True)

    df["time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["dt"] = df["time_s"].diff().fillna(1.0).clip(lower=0.2, upper=10.0)

    # Дистанция
    if df["distance_m"].isna().all():
        if df["speed_m_s"].notna().any():
            df["distance_m"] = (df["speed_m_s"].fillna(0) * df["dt"]).cumsum()
        else:
            df["distance_m"] = np.arange(len(df))
    else:
        df["distance_m"] = df["distance_m"].fillna(method="ffill")
        df["distance_m"] = df["distance_m"].fillna(0)
        df["distance_m"] = np.maximum.accumulate(df["distance_m"].values)

    # Скорост
    nan_ratio = df["speed_m_s"].isna().mean()
    ddist = df["distance_m"].diff().fillna(0.0)
    speed_from_dist = (ddist / df["dt"]).clip(lower=0.0)

    if nan_ratio > 0.2:
        df["speed_m_s"] = speed_from_dist
    else:
        df["speed_m_s"] = df["speed_m_s"].fillna(speed_from_dist)

    return df


# ---------- ПОМОЩНИ ФУНКЦИИ ----------

def smooth_series(series: pd.Series, window_sec: int, dt_series: pd.Series) -> pd.Series:
    """Просто изглаждане с плъзгащ прозорец по време."""
    median_dt = float(dt_series.median()) if not dt_series.isna().all() else 1.0
    if median_dt <= 0:
        median_dt = 1.0
    window = max(int(round(window_sec / median_dt)), 1)
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Изчислява наклон (slope_dec, slope_pct) от изгладени височина и дистанция.
    slope_dec ~ Δh/Δs (отрицателно при спускане)
    """
    df = df.copy()

    df["alt_smooth"] = smooth_series(df["altitude_m"], window_sec=30, dt_series=df["dt"])
    df["dist_smooth"] = smooth_series(df["distance_m"], window_sec=30, dt_series=df["dt"])

    dalt = df["alt_smooth"].diff().fillna(0.0)
    ddist = df["dist_smooth"].diff().fillna(1.0)

    ddist = ddist.where(ddist.abs() > 0.5, np.nan)
    slope_dec = (dalt / ddist).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slope_dec = slope_dec.clip(lower=-0.5, upper=0.5)

    df["slope_dec"] = slope_dec
    df["slope_pct"] = slope_dec * 100.0
    return df


# ---------- НОВ МОДЕЛ: GLIDE RATIO ----------

def estimate_glide_ratio(
    df: pd.DataFrame,
    slope_threshold_pct: float = -5.0,
    v_min: float = 1.5,
):
    """
    Оценява "glide_ratio" = a_real / a_theor за спускания.

    - взимаме само точки със slope_pct <= slope_threshold_pct (спускане)
    - гледаме само ускорения a_real > 0 (ускорява надолу)
    - игнорираме много малки наклони
    - правим медиана на r = a_real / a_theor

    Връща:
        glide_ratio (медиана), loss_pct (100*(1-r)), n_points
    """
    df = df.copy()

    # изглаждаме скорост за по-стабилно ускорение
    df["speed_smooth"] = smooth_series(df["speed_m_s"], window_sec=10, dt_series=df["dt"])
    df["a_real"] = df["speed_smooth"].diff() / df["dt"]
    df["a_real"] = df["a_real"].replace([np.inf, -np.inf], np.nan)

    # маска за спускане
    mask = (
        (df["slope_pct"] <= slope_threshold_pct) &
        (df["speed_smooth"] >= v_min)
    )

    if not mask.any():
        return np.nan, np.nan, 0

    sub = df.loc[mask].copy()

    # големина на наклона (положителна при спускане)
    sub["slope_mag"] = -sub["slope_dec"]
    sub = sub[sub["slope_mag"] > 0]  # само реални спускания

    # теоретично ускорение без триене
    sub["a_theor"] = G * sub["slope_mag"]

    # интересуват ни точки, където реално ускорява
    sub = sub[sub["a_real"] > 0]

    if sub.empty:
        return np.nan, np.nan, 0

    r = sub["a_real"] / sub["a_theor"]
    r = r.replace([np.inf, -np.inf], np.nan)
    # режем абсурди
    r = r[(r > 0) & (r < 2)].dropna()

    if r.empty:
        return np.nan, np.nan, 0

    glide_ratio = float(r.median())
    loss_pct = float((1.0 - glide_ratio) * 100.0)
    n_points = int(r.shape[0])
    return glide_ratio, loss_pct, n_points


def modulate_speed_by_glide(df: pd.DataFrame, glide_ratio: float, glide_ref: float) -> pd.DataFrame:
    """
    Модулира скоростта според glide_ratio и референтен glide_ref:
        v_mod = v_real * (glide_ref / glide_ratio)
    """
    df = df.copy()
    if not np.isfinite(glide_ratio) or glide_ratio <= 0 or glide_ref <= 0:
        df["speed_mod_m_s"] = df["speed_m_s"]
        return df

    factor = glide_ref / glide_ratio
    df["speed_mod_m_s"] = df["speed_m_s"] * factor
    return df


# ---------- STREAMLIT UI ----------

def main():
    st.title("onFlows — Модулиране на скоростта (glide ratio модел)")

    st.markdown(
        """
        **Идея на модела**

        - При спускане теоретичното ускорение без триене е: `a_theor = g * |slope|`.  
        - От данните смятаме реалното ускорение `a_real = dv/dt`.  
        - Отношението `r = a_real / a_theor` показва каква част от гравитацията се "реализира".  
          - `r ≈ 1` → почти без съпротивление.  
          - `r = 0.6` → 40% загуба от съпротивления (триене, въздух и др.).  
        - За всяка активност взимаме медиана на r → **glide_ratio**.  
        - Модулираме скоростта към референтна активност с glide_ratio_ref.
        """
    )

    uploaded_files = st.file_uploader(
        "Качи един или повече TCX файла (ски / ролки)",
        type=["tcx", "xml"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("⬆️ Качи поне един файл, за да започнеш.")
        return

    st.sidebar.header("Настройки")

    smooth_window_sec = st.sidebar.slider(
        "Изглаждане на скоростта (секунди)",
        min_value=5,
        max_value=60,
        value=20,
        step=5,
    )

    slope_threshold_pct = st.sidebar.slider(
        "Минимален наклон за анализ на спускане (%)",
        min_value=-15,
        max_value=-1,
        value=-5,
        step=1,
    )

    activities = []
    for f in uploaded_files:
        try:
            df = parse_tcx(f)
            df = compute_slopes(df)
            # изглаждаме скоростта за цялостна употреба (отделно от този в estimate_glide_ratio)
            df["speed_m_s"] = smooth_series(df["speed_m_s"], window_sec=smooth_window_sec, dt_series=df["dt"])

            glide_ratio, loss_pct, n_pts = estimate_glide_ratio(
                df,
                slope_threshold_pct=slope_threshold_pct,
                v_min=1.5,
            )
            avg_speed = df["speed_m_s"].mean() * 3.6

            activities.append(
                {
                    "name": f.name,
                    "df": df,
                    "glide_ratio": glide_ratio,
                    "loss_pct": loss_pct,
                    "n_pts": n_pts,
                    "avg_speed": avg_speed,
                }
            )
        except Exception as e:
            st.error(f"Грешка при обработка на {f.name}: {e}")

    if not activities:
        st.error("Няма успешно обработени активности.")
        return

    # таблица с обобщение
    summary_rows = []
    for a in activities:
        summary_rows.append(
            {
                "Файл": a["name"],
                "Средна скорост (реална) km/h": a["avg_speed"],
                "Glide ratio (a_real/a_theor)": a["glide_ratio"],
                "Загуба от съпротивление %": a["loss_pct"],
                "Брой точки (спускане)": a["n_pts"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.subheader("Обобщение по активности")
    st.dataframe(summary_df.style.format(precision=3))

    # избор на референтна активност
    valid_for_ref = [a for a in activities if np.isfinite(a["glide_ratio"]) and a["glide_ratio"] > 0]
    if not valid_for_ref:
        st.warning("Няма активност с валиден glide_ratio → модулацията ще използва v_real без промяна.")
        glide_ref = 1.0
        ref_name = None
    else:
        ref_name = st.sidebar.selectbox(
            "Референтна активност (нейният glide_ratio става референция):",
            options=[a["name"] for a in valid_for_ref],
        )
        glide_ref = [a["glide_ratio"] for a in valid_for_ref if a["name"] == ref_name][0]
        st.sidebar.write(f"Избран glide_ref = {glide_ref:.3f}")

    # модулация на скоростите за всички активности
    for a in activities:
        a["df_mod"] = modulate_speed_by_glide(a["df"], a["glide_ratio"], glide_ref)
        a["avg_speed_mod"] = a["df_mod"]["speed_mod_m_s"].mean() * 3.6

    # дописваме в таблицата средните модулирани скорости
    summary_df["Средна скорост (модулирана) km/h"] = [
        a["avg_speed_mod"] for a in activities
    ]
    st.subheader("Обобщение с модулирана скорост")
    st.dataframe(summary_df.style.format(precision=3))

    # подробна визуализация
    sel_name = st.selectbox(
        "Избери активност за подробен преглед:",
        options=[a["name"] for a in activities],
    )
    sel = [a for a in activities if a["name"] == sel_name][0]

    st.markdown(
        f"""
        **{sel_name}**  
        - Glide ratio: `{sel['glide_ratio']:.3f}`  
        - Загуба от съпротивление: `{sel['loss_pct']:.1f}%`  
        - Брой точки за оценка: `{sel['n_pts']}`  
        - Средна скорост (реална): `{sel['avg_speed']:.2f} km/h`  
        - Средна скорост (модулирана): `{sel['avg_speed_mod']:.2f} km/h`  
        """
    )

    df_plot = pd.DataFrame(
        {
            "Време [мин]": sel["df_mod"]["time_s"] / 60.0,
            "Реална скорост [km/h]": sel["df_mod"]["speed_m_s"] * 3.6,
            "Модулирана скорост [km/h]": sel["df_mod"]["speed_mod_m_s"] * 3.6,
        }
    ).set_index("Време [мин]")

    st.write("Първите 10 реда (проверка на данните):")
    st.write(df_plot.head(10))

    st.line_chart(df_plot)

    st.caption(
        "Glide ratio се базира само на спускания: сравнява реалното ускорение с теоретичното без триене. "
        "Модулираната скорост показва как би изглеждала активността при същото усилие, "
        "ако съпротивленията бяха като в референтната активност."
    )


if __name__ == "__main__":
    main()
