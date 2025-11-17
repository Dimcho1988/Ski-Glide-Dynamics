import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

G = 9.80665   # gravitational acceleration

st.set_page_config(page_title="onFlows — Glide Coefficient μ_eff", layout="wide")


# ---------------------------------------------------------
# TCX PARSER
# ---------------------------------------------------------

def parse_tcx(file_obj) -> pd.DataFrame:
    """Reads TCX file → returns DataFrame with time, distance, altitude, speed."""
    data = file_obj.read()
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

    df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "distance_m": dist,
        "altitude_m": alt
    })

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)

    return df


# ---------------------------------------------------------
# SMOOTHING + GRADIENT + ACCELERATION
# ---------------------------------------------------------

def central_diff(y, x):
    """Central difference derivative with edge handling."""
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


def process_df(df, frac=0.02):
    """Smooth altitude, distance, speed; compute grade and acceleration."""
    t = df["t_sec"].values

    # smoothing (LOWESS)
    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=frac, return_sorted=False)
    df["dist_smooth"] = lowess(df["distance_m"], t, frac=frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=frac, return_sorted=False)

    # derivatives
    dalt_dt = central_diff(df["alt_smooth"].values, t)
    ddist_dt = central_diff(df["dist_smooth"].values, t)
    dspeed_dt = central_diff(df["speed_smooth"].values, t)

    # grade % = (dh/ds) * 100
    raw_grade = (dalt_dt / ddist_dt) * 100
    df["grade"] = lowess(raw_grade, t, frac=0.05, return_sorted=False)

    # acceleration
    raw_acc = dspeed_dt
    df["acc"] = lowess(raw_acc, t, frac=0.05, return_sorted=False)

    return df


# ---------------------------------------------------------
# STEADY-STATE FILTER
# ---------------------------------------------------------

def filter_steady_state(df, acc_threshold=0.05):
    """Return only points with |acc| < threshold."""
    mask = np.abs(df["acc"]) < acc_threshold
    return df[mask]


# ---------------------------------------------------------
# μ_eff COMPUTATION VIA SPEED–GRADE REGRESSION
# ---------------------------------------------------------

def compute_mu_eff(df_ss):
    """
    Compute μ_eff using linear regression:
        speed = c0 + c1 * grade
    μ_eff = |grade_zero| / 100
    where grade_zero = -c0 / c1
    """
    if len(df_ss) < 50:
        return np.nan, np.nan, np.nan, None

    X = df_ss["grade"].values.reshape(-1, 1)
    y = df_ss["speed_smooth"].values

    model = LinearRegression()
    model.fit(X, y)

    c1 = model.coef_[0]
    c0 = model.intercept_

    if c1 == 0:
        return np.nan, c0, c1, model

    grade_zero = -c0 / c1     # % grade where speed = 0
    mu_eff = abs(grade_zero) / 100

    return mu_eff, c0, c1, model


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

def main():

    st.title("onFlows — Steady-State Glide Coefficient (μ_eff)")
    st.markdown("""
    **Стандартен научен модел за измерване на плазгаемост / съпротивление.**  
    Прилага се в SkiLab тестовете и норвежката федерация.  
    Работи чрез steady-state регресия между скорост и наклон.
    """)

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла",
        type=["tcx"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Качи поне един файл.")
        return

    # Sidebar controls
    st.sidebar.header("Параметри")
    acc_threshold = st.sidebar.slider(
        "Steady-State прах за ускорение |a| < (m/s²)",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01
    )

    smooth_frac = st.sidebar.slider(
        "Степен на изглаждане (LOWESS frac)",
        min_value=0.01, max_value=0.20, value=0.02, step=0.01
    )

    # Results store
    results = {}

    for file in uploaded_files:
        try:
            df_raw = parse_tcx(file)
            df = process_df(df_raw, frac=smooth_frac)
            df_ss = filter_steady_state(df, acc_threshold)

            mu_eff, c0, c1, model = compute_mu_eff(df_ss)

            results[file.name] = {
                "df": df,
                "df_ss": df_ss,
                "mu": mu_eff,
                "c0": c0,
                "c1": c1,
                "model": model,
            }

        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    # Show results
    st.subheader("Резултати:")

    summary_rows = []
    for name, obj in results.items():
        summary_rows.append(
            {
                "активност": name,
                "μ_eff": obj["mu"],
                "c0 (intercept)": obj["c0"],
                "c1 (slope)": obj["c1"],
                "steady-state точки": len(obj["df_ss"])
            }
        )

    st.dataframe(pd.DataFrame(summary_rows).style.format({
        "μ_eff": "{:.4f}",
        "c0 (intercept)": "{:.3f}",
        "c1 (slope)": "{:.3f}",
    }))

    # Graph viewer
    selected_name = st.selectbox("Избери активност за визуализация", list(results.keys()))
    obj = results[selected_name]

    df = obj["df"]
    df_ss = obj["df_ss"]
    mu_eff = obj["mu"]
    c0, c1 = obj["c0"], obj["c1"]

    st.subheader(f"Графика — {selected_name}")

    st.markdown("### Speed vs Grade (Steady-State Points + Regression Line)")
    if mu_eff is not np.nan and obj["model"] is not None:
        X_line = np.linspace(df["grade"].min(), df["grade"].max(), 200).reshape(-1, 1)
        y_line = obj["model"].predict(X_line)

        chart_df = pd.DataFrame({
            "grade": df_ss["grade"],
            "speed": df_ss["speed_smooth"]
        })

        st.scatter_chart(chart_df, x="grade", y="speed")

        reg_df = pd.DataFrame({
            "grade": X_line.flatten(),
            "speed": y_line
        })
        st.line_chart(reg_df.set_index("grade"))

    else:
        st.warning("Недостатъчно steady-state точки за регресия.")

    st.markdown(f"### μ_eff = **{mu_eff:.4f}**")

    st.markdown("""---""")


if __name__ == "__main__":
    main()
