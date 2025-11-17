import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

st.set_page_config(page_title="onFlows — Mechanical Power (from TCX)", layout="wide")


# ---------- TCX parser ----------

def parse_tcx_bytes(data: bytes) -> pd.DataFrame:
    """
    Парсва TCX от bytes → DataFrame с time, distance_m, altitude_m, speed (m/s).
    """
    tree = ET.parse(io.BytesIO(data))
    root = tree.getroot()

    ns = {"tcx": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}

    times, dist, alt = [], [], []

    for tp in root.findall(".//tcx:Trackpoint", ns):
        t_el = tp.find("tcx:Time", ns)
        d_el = tp.find("tcx:DistanceMeters", ns)
        a_el = tp.find("tcx:AltitudeMeters", ns)

        if t_el is None or d_el is None or a_el is None:
            continue

        times.append(t_el.text)
        dist.append(float(d_el.text))
        alt.append(float(a_el.text))

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "distance_m": dist,
            "altitude_m": alt,
        }
    )

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)  # m/s

    return df


# ---------- helpers ----------

def central_diff(y, x):
    dy = np.zeros_like(y)
    dx = np.zeros_like(y)

    dy[1:-1] = y[2:] - y[:-2]
    dx[1:-1] = x[2:] - x[:-2]

    dy[0] = y[1] - y[0]
    dx[0] = x[1] - x[0]
    dy[-1] = y[-1] - y[-2]
    dx[-1] = x[-1] - x[-2]

    dx[dx == 0] = np.nan
    return dy / dx


def smooth_and_kin(df: pd.DataFrame, frac: float = 0.02) -> pd.DataFrame:
    """
    LOWESS изглаждане на височина, дистанция, скорост + наклон (%) и ускорение.
    """
    t = df["t_sec"].values

    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=frac, return_sorted=False)
    df["dist_smooth"] = lowess(df["distance_m"], t, frac=frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=frac, return_sorted=False)

    dalt_dt = central_diff(df["alt_smooth"].values, t)
    ddist_dt = central_diff(df["dist_smooth"].values, t)
    dspeed_dt = central_diff(df["speed_smooth"].values, t)

    # наклон (%)
    raw_grade = (dalt_dt / ddist_dt) * 100.0
    df["grade"] = lowess(raw_grade, t, frac=0.05, return_sorted=False)

    # ускорение (m/s^2)
    df["acc"] = lowess(dspeed_dt, t, frac=0.05, return_sorted=False)

    return df


def compute_mechanical_power(df: pd.DataFrame, mass_kg: float) -> pd.DataFrame:
    """
    P_grav = m g v grade/100 (само ако grade>0)
    P_kin  = m a v (само ако a>0)
    P_mech = P_grav_pos + P_kin_pos
    """
    g = 9.80665

    v = df["speed_smooth"].values      # m/s
    grade_frac = df["grade"].values / 100.0
    a = df["acc"].values

    # мощност срещу гравитацията
    P_grav = mass_kg * g * v * grade_frac
    P_grav_pos = np.maximum(P_grav, 0.0)

    # мощност за ускоряване
    a_pos = np.maximum(a, 0.0)
    P_kin = mass_kg * a_pos * v

    P_mech = P_grav_pos + P_kin

    df["P_grav_W"] = P_grav_pos
    df["P_kin_W"] = P_kin
    df["P_mech_W"] = P_mech
    df["P_mech_Wkg"] = df["P_mech_W"] / mass_kg

    return df


def time_weighted_mean(series: pd.Series, t_sec: pd.Series) -> float:
    """
    Времево претеглена средна мощност.
    """
    if len(series) < 2:
        return np.nan

    t = t_sec.values
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    x = series.values
    w = dt

    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan

    return float(np.sum(x[mask] * w[mask]) / np.sum(w[mask]))


# ---------- Streamlit UI ----------

def main():
    st.title("onFlows — Механична мощност от TCX (гравитация + ускорение)")

    st.markdown(
        """
Това приложение изчислява **минималната положителна механична мощност** на спортиста
според профила на скоростта и наклона:

- **P_grav** – мощност за преодоляване на изкачванията  
- **P_kin** – мощност за ускорения  
- **P_mech = P_grav + P_kin** (само положителната част)  

Не включва триене и въздушно съпротивление, така че стойностите са по-скоро
*минимална необходима механична мощност*, но са полезни за сравнение между активности.
"""
    )

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл.")
        return

    st.sidebar.header("Параметри на модела")

    mass_kg = st.sidebar.number_input(
        "Тегло на спортиста (kg)",
        min_value=40.0,
        max_value=120.0,
        value=70.0,
        step=1.0,
    )

    smooth_frac = st.sidebar.slider(
        "Степен на изглаждане (LOWESS frac)",
        min_value=0.01,
        max_value=0.20,
        value=0.02,
        step=0.01,
    )

    results = {}

    for file in uploaded_files:
        try:
            data = file.read()
            df_raw = parse_tcx_bytes(data)
            df = smooth_and_kin(df_raw, frac=smooth_frac)
            df = compute_mechanical_power(df, mass_kg)

            # времево претеглени средни стойности
            mean_P = time_weighted_mean(df["P_mech_W"], df["t_sec"])
            mean_Pkg = time_weighted_mean(df["P_mech_Wkg"], df["t_sec"])
            mean_Pgrav = time_weighted_mean(df["P_grav_W"], df["t_sec"])
            mean_Pkin = time_weighted_mean(df["P_kin_W"], df["t_sec"])

            results[file.name] = {
                "df": df,
                "mean_P": mean_P,
                "mean_Pkg": mean_Pkg,
                "mean_Pgrav": mean_Pgrav,
                "mean_Pkin": mean_Pkin,
            }

        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    # Резюме таблица
    rows = []
    for name, obj in results.items():
        rows.append(
            {
                "activity": name,
                "mean_P_W": obj["mean_P"],
                "mean_P_Wkg": obj["mean_Pkg"],
                "mean_P_grav_W": obj["mean_Pgrav"],
                "mean_P_kin_W": obj["mean_Pkin"],
            }
        )

    res_df = pd.DataFrame(rows)

    st.subheader("Резюме по активности (механична мощност)")
    st.dataframe(
        res_df.style.format(
            {
                "mean_P_W": "{:.1f}",
                "mean_P_Wkg": "{:.2f}",
                "mean_P_grav_W": "{:.1f}",
                "mean_P_kin_W": "{:.1f}",
            }
        )
    )

    # Детайлна визуализация
    selected = st.selectbox(
        "Избери активност за детайлна графика",
        res_df["activity"].tolist(),
    )

    df_sel = results[selected]["df"]

    st.subheader(f"Графики — {selected}")

    with st.expander("Скорост, наклон и ускорение"):
        st.line_chart(
            df_sel.set_index("t_sec")[["speed_smooth"]].rename(
                columns={"speed_smooth": "speed (m/s)"}
            )
        )
        st.line_chart(
            df_sel.set_index("t_sec")[["grade"]].rename(columns={"grade": "grade (%)"})
        )
        st.line_chart(
            df_sel.set_index("t_sec")[["acc"]].rename(columns={"acc": "acc (m/s²)"})
        )

    st.markdown("**Механична мощност във времето (W и W/kg):**")
    power_plot = df_sel.set_index("t_sec")[["P_mech_W", "P_mech_Wkg"]]
    st.line_chart(power_plot)

    st.caption(
        "Важно: това е минимална външна механична мощност (само гравитация + ускорения). "
        "Реалната метаболитна мощност е по-висока заради триене, въздух и енергийни загуби."
    )


if __name__ == "__main__":
    main()
