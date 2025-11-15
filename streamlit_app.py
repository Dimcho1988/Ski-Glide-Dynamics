import io
import math
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt  # НОВО

G = 9.81  # gravitational acceleration m/s^2


def parse_tcx(file_obj) -> pd.DataFrame:
    """
    Parse a TCX file-like object and return a DataFrame with:
    time (datetime), time_s, altitude_m, distance_m, speed_m_s
    """
    data = {
        "time": [],
        "altitude_m": [],
        "distance_m": [],
        "speed_m_s": [],
    }

    # Read content for ElementTree
    content = file_obj.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="ignore")
    file_obj = io.StringIO(content)

    tree = ET.parse(file_obj)
    root = tree.getroot()

    # detect namespace
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0].strip("{")
    else:
        ns_uri = "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"
    ns = {"tcx": ns_uri}

    # try to find all Trackpoint elements
    trackpoints = root.findall(".//tcx:Trackpoint", ns)
    if not trackpoints:
        # fallback: no namespace prefix
        trackpoints = root.findall(".//Trackpoint")

    for tp in trackpoints:
        # time
        t_el = tp.find(".//{*}Time")
        if t_el is None or t_el.text is None:
            continue
        try:
            time = pd.to_datetime(t_el.text)
        except Exception:
            continue

        # altitude
        alt_el = tp.find(".//{*}AltitudeMeters")
        altitude = float(alt_el.text) if alt_el is not None and alt_el.text is not None else np.nan

        # distance
        dist_el = tp.find(".//{*}DistanceMeters")
        distance = float(dist_el.text) if dist_el is not None and dist_el.text is not None else np.nan

        # speed – може да е в Extensions; ако не, ще я смятаме от дистанцията
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

    # relative time in seconds
    df["time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["dt"] = df["time_s"].diff().fillna(1.0).clip(lower=0.2, upper=10.0)

    # fill distance gaps if needed (monotonic increasing)
    if df["distance_m"].isna().all():
        if df["speed_m_s"].notna().any():
            df["distance_m"] = (df["speed_m_s"].fillna(0) * df["dt"]).cumsum()
        else:
            df["distance_m"] = np.arange(len(df))
    else:
        df["distance_m"] = df["distance_m"].fillna(method="ffill")
        df["distance_m"] = df["distance_m"].fillna(0)
        df["distance_m"] = np.maximum.accumulate(df["distance_m"].values)

    # compute speed from distance if missing or too many NaNs
    nan_ratio = df["speed_m_s"].isna().mean()
    ddist = df["distance_m"].diff().fillna(0.0)
    speed_from_dist = (ddist / df["dt"]).clip(lower=0.0)

    if nan_ratio > 0.2:
        df["speed_m_s"] = speed_from_dist
    else:
        df["speed_m_s"] = df["speed_m_s"].fillna(speed_from_dist)

    return df


def smooth_series(series: pd.Series, window_sec: int, dt_series: pd.Series) -> pd.Series:
    """
    Просто изглаждане с плъзгащ прозорец в секунди.
    Приемаме приблизително постоянен sampling.
    """
    median_dt = float(dt_series.median()) if not dt_series.isna().all() else 1.0
    if median_dt <= 0:
        median_dt = 1.0
    window = max(int(round(window_sec / median_dt)), 1)
    return series.rolling(window=window, center=True, min_periods=1).mean()


def compute_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Изчислява наклон (slope) в десетична и в % от изгладени височина и дистанция.
    """
    df = df.copy()

    df["alt_smooth"] = smooth_series(df["altitude_m"], window_sec=30, dt_series=df["dt"])
    df["dist_smooth"] = smooth_series(df["distance_m"], window_sec=30, dt_series=df["dt"])

    dalt = df["alt_smooth"].diff().fillna(0.0)
    ddist = df["dist_smooth"].diff().fillna(1.0)

    ddist = ddist.where(ddist.abs() > 0.5, np.nan)  # <0.5 m – шум
    slope_dec = (dalt / ddist).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slope_dec = slope_dec.clip(lower=-0.5, upper=0.5)  # -50% до +50%

    df["slope_dec"] = slope_dec
    df["slope_pct"] = slope_dec * 100.0
    return df


def estimate_mu_eff(df: pd.DataFrame) -> float:
    """
    Оценка на ефективния коефициент на триене µ_eff за цялата активност.
    - гледаме само спускания (slope <= -5%)
    - игнорираме първите 5 s на всяко спускане
    Формула: mu ≈ -(a + g sinθ) / (g cosθ)
    """
    df = df.copy()

    # изглаждаме скоростта за ускорението
    df["speed_smooth"] = smooth_series(df["speed_m_s"], window_sec=10, dt_series=df["dt"])

    # ускорение
    df["a"] = df["speed_smooth"].diff() / df["dt"]
    df["a"] = df["a"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # маска за спускане
    df["is_downhill"] = df["slope_pct"] <= -5.0

    # идентифицираме сегменти
    seg_id = 0
    seg_ids = []
    prev_flag = False
    for flag in df["is_downhill"]:
        if flag and not prev_flag:
            seg_id += 1
        seg_ids.append(seg_id if flag else 0)
        prev_flag = flag
    df["seg_id"] = seg_ids

    use_mask = np.zeros(len(df), dtype=bool)
    for sid in sorted(df["seg_id"].unique()):
        if sid == 0:
            continue
        seg_idx = np.where(df["seg_id"].values == sid)[0]
        if len(seg_idx) < 10:
            continue  # минимум 10 s
        seg_idx = seg_idx[5:]  # махаме първите 5 s
        use_mask[seg_idx] = True

    # допълнителни филтри
    use_mask &= df["speed_smooth"].values > 1.0   # > 1 m/s
    use_mask &= df["speed_smooth"].values < 15.0  # < 54 km/h

    if not use_mask.any():
        return np.nan

    subset = df.loc[use_mask].copy()

    subset["theta"] = np.arctan(subset["slope_dec"].values)
    sin_theta = np.sin(subset["theta"])
    cos_theta = np.cos(subset["theta"])

    mu = -(subset["a"].values + G * sin_theta) / (G * cos_theta)

    mu = np.where(np.isfinite(mu), mu, np.nan)
    mu = np.where((mu >= 0.001) & (mu <= 0.2), mu, np.nan)

    mu_clean = pd.Series(mu).dropna()
    if mu_clean.empty:
        return np.nan

    return float(mu_clean.median())


def modulate_speed(df: pd.DataFrame, mu_eff: float, mu_ref: float) -> pd.DataFrame:
    """
    Модулиране на скоростта към референтно триене:
    v_mod = v * sqrt(mu_eff / mu_ref)
    """
    df = df.copy()
    if not np.isfinite(mu_eff) or mu_eff <= 0 or mu_ref <= 0:
        df["speed_mod_m_s"] = df["speed_m_s"]
        return df

    factor = math.sqrt(mu_eff / mu_ref)
    df["speed_mod_m_s"] = df["speed_m_s"] * factor
    return df


def main():
    st.title("onFlows — Модулиране на скоростта при ски / ролки спрямо плъзгаемост")

    st.markdown(
        """
        Това приложение:

        1. Зарежда **TCX** файлове от ски бягане / ролки.  
        2. Оценява ефективен коефициент на триене **µ_eff** от спусканията.  
        3. Модулира скоростта към избрана референтна плъзгаемост **µ_ref**.  
        4. Показва сравнение между **реална** и **модулирана** скорост.
        """
    )

    uploaded_files = st.file_uploader(
        "Качи един или повече TCX файла",
        type=["tcx", "xml"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("⬆️ Качи поне един TCX файл, за да започнеш.")
        return

    # Sidebar: настройки
    st.sidebar.header("Настройки")

    mu_ref_mode = st.sidebar.radio(
        "Референтни условия (µ_ref):",
        options=["Фиксирана стойност", "Избери референтна активност"],
    )

    mu_ref_fixed = st.sidebar.slider(
        "Фиксирана µ_ref (по-ниска = по-добра плъзгаемост)",
        min_value=0.01,
        max_value=0.08,
        value=0.025,
        step=0.001,
    )

    smooth_window_sec = st.sidebar.slider(
        "Изглаждане на скоростта (секунди)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
    )

    activities = {}
    summary_rows = []

    # Обработка на всеки файл
    for f in uploaded_files:
        try:
            df = parse_tcx(f)
            df = compute_slopes(df)
            # изглаждане на скоростта за визуализация и за ускорение
            df["speed_m_s"] = smooth_series(
                df["speed_m_s"],
                window_sec=smooth_window_sec,
                dt_series=df["dt"],
            )
            mu_eff = estimate_mu_eff(df)
            activities[f.name] = {"df": df, "mu_eff": mu_eff}

            avg_speed = df["speed_m_s"].mean() * 3.6  # km/h
            summary_rows.append(
                {
                    "Файл": f.name,
                    "Средна скорост (реална) km/h": avg_speed,
                    "µ_eff (оценена)": mu_eff,
                }
            )
        except Exception as e:
            st.error(f"Грешка при обработка на файла {f.name}: {e}")

    if not activities:
        st.error("Няма успешно обработени активности.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # избор на µ_ref
    if mu_ref_mode == "Избери референтна активност":
        valid_mu = summary_df.dropna(subset=["µ_eff (оценена)"])
        if valid_mu.empty:
            st.sidebar.warning("Няма активност с валидна µ_eff. Използвам фиксирана µ_ref.")
            mu_ref = mu_ref_fixed
        else:
            ref_name = st.sidebar.selectbox(
                "Избери референтна активност (нейната µ_eff става µ_ref):",
                options=list(valid_mu["Файл"].values),
            )
            mu_ref = float(
                valid_mu.loc[valid_mu["Файл"] == ref_name, "µ_eff (оценена)"].iloc[0]
            )
            st.sidebar.write(f"Избрана µ_ref = {mu_ref:.4f}")
    else:
        mu_ref = mu_ref_fixed

    # прилагаме модулацията
    for name, obj in activities.items():
        df = obj["df"]
        mu_eff = obj["mu_eff"]
        df_mod = modulate_speed(df, mu_eff=mu_eff, mu_ref=mu_ref)
        activities[name]["df"] = df_mod

        avg_speed_mod = df_mod["speed_mod_m_s"].mean() * 3.6
        summary_df.loc[
            summary_df["Файл"] == name,
            "Средна скорост (модулирана) km/h",
        ] = avg_speed_mod

    st.subheader("Обобщение по активности")
    st.dataframe(summary_df.style.format(precision=3))

    # избор на активност за детайлен plot
    selected_name = st.selectbox(
        "Избери активност за подробен преглед:",
        options=list(activities.keys()),
    )

    df_sel = activities[selected_name]["df"]
    mu_eff_sel = activities[selected_name]["mu_eff"]

    st.markdown(
        f"""
        **{selected_name}**  
        - Оценена µ_eff: `{mu_eff_sel:.4f}`  
        - Използвана µ_ref: `{mu_ref:.4f}`  
        """
    )

    # данни за графика (чистим NaN)
    plot_df = pd.DataFrame(
        {
            "Време [мин]": df_sel["time_s"] / 60.0,
            "Скорост реална [km/h]": df_sel["speed_m_s"] * 3.6,
            "Скорост модулирана [km/h]": df_sel["speed_mod_m_s"] * 3.6,
        }
    ).dropna()

    if plot_df.empty:
        st.warning("Няма достатъчно валидни данни за визуализация.")
    else:
        # Altair: две линии + легенда
        chart = (
            alt.Chart(plot_df)
            .transform_fold(
                ["Скорост реална [km/h]", "Скорост модулирана [km/h]"],
                as_=["Вид", "Скорост"],
            )
            .mark_line()
            .encode(
                x=alt.X("Време [мин]:Q", title="Време [мин]"),
                y=alt.Y("Скорост:Q", title="Скорост [km/h]"),
                color=alt.Color("Вид:N", title="Тип скорост"),
                tooltip=["Време [мин]:Q", "Вид:N", "Скорост:Q"],
            )
            .properties(height=350)
        )

        st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Модулираната скорост показва каква би била скоростта при същото усилие, "
        "ако плъзгаемостта/триенето беше равна на избраната µ_ref."
    )


if __name__ == "__main__":
    main()
