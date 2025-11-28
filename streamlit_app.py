import streamlit as st
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from io import BytesIO
from datetime import datetime
import altair as alt

# -----------------------------
# Настройки по подразбиране
# -----------------------------
T_SEG = 5.0              # дължина на сегмента [s]
MIN_POINTS_SEG = 2       # минимум точки в сегмента
MIN_D_SEG = 5.0          # минимум хоризонтална дистанция [m]
MIN_T_SEG = 3.0          # минимум продължителност [s]
MAX_ABS_SLOPE = 30.0     # макс. наклон [%]

DOWN_MIN = -15.0         # долна граница за downhill [%] за Glide модела
DOWN_MAX = -5.0          # горна граница за downhill [%]

SLOPE_MODEL_MIN = -3.0   # диапазон за ΔV% модела
SLOPE_MODEL_MAX = 10.0
FLAT_BAND = 1.0          # |slope| <= 1% се счита за "равно"


# --------------------------------------------------------------------------------
# Помощни функции за TCX и сегменти
# --------------------------------------------------------------------------------
def parse_tcx(file: BytesIO, activity_id: str) -> pd.DataFrame:
    """Парсира TCX файл и връща DataFrame с ['activity_id','time','dist','alt','hr']"""
    tree = ET.parse(file)
    root = tree.getroot()

    ns = {}
    if root.tag[0] == "{":
        uri = root.tag[1:].split("}")[0]
        ns["tcx"] = uri
    else:
        ns["tcx"] = ""

    trackpoints = root.findall(".//tcx:Trackpoint", ns)
    rows = []

    for tp in trackpoints:
        time_el = tp.find("tcx:Time", ns)
        dist_el = tp.find("tcx:DistanceMeters", ns)
        alt_el = tp.find("tcx:AltitudeMeters", ns)
        hr_el = tp.find("tcx:HeartRateBpm/tcx:Value", ns)

        if time_el is None or dist_el is None or alt_el is None:
            continue

        t_str = time_el.text
        try:
            t = pd.to_datetime(t_str)
        except Exception:
            try:
                t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%S.%fZ")
            except Exception:
                try:
                    t = datetime.strptime(t_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    continue

        try:
            dist = float(dist_el.text)
            alt = float(alt_el.text)
        except (TypeError, ValueError):
            continue

        if hr_el is not None and hr_el.text is not None:
            try:
                hr = float(hr_el.text)
            except (TypeError, ValueError):
                hr = np.nan
        else:
            hr = np.nan

        rows.append(
            {
                "activity_id": activity_id,
                "time": t,
                "dist": dist,
                "alt": alt,
                "hr": hr,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["activity_id", "time", "dist", "alt", "hr"])

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def preprocess_points(df: pd.DataFrame) -> pd.DataFrame:
    """Сортира по време, прави time_sec и alt_smooth (без филтри)."""
    if df.empty:
        return df
    df = df.sort_values("time").reset_index(drop=True)
    df["alt_smooth"] = df["alt"]
    df["time_sec"] = df["time"].astype("int64") / 1e9
    return df


def build_segments(df: pd.DataFrame, activity_id: str, t_seg: float = T_SEG) -> pd.DataFrame:
    """
    5-секундни сегменти (без припокриване), използваме първа/последна точка.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "activity_id",
                "seg_id",
                "t_start",
                "t_end",
                "duration_s",
                "D_m",
                "dh_m",
                "slope_pct",
                "V_kmh",
                "hr_mean",
            ]
        )

    t0 = df["time_sec"].iloc[0]
    df["t_rel"] = df["time_sec"] - t0
    df["seg_id"] = (df["t_rel"] / t_seg).astype(int)

    seg_rows = []
    for seg_id, g in df.groupby("seg_id"):
        g = g.sort_values("time")
        if len(g) < MIN_POINTS_SEG:
            continue

        t_start = g["time"].iloc[0]
        t_end = g["time"].iloc[-1]
        duration = (t_end - t_start).total_seconds()
        if duration < MIN_T_SEG:
            continue

        dist_start = g["dist"].iloc[0]
        dist_end = g["dist"].iloc[-1]
        D = dist_end - dist_start
        if D < MIN_D_SEG:
            continue

        alt_start = g["alt_smooth"].iloc[0]
        alt_end = g["alt_smooth"].iloc[-1]
        dh = alt_end - alt_start
        slope = (dh / D) * 100.0 if D > 0 else np.nan
        if np.isnan(slope) or abs(slope) > MAX_ABS_SLOPE:
            continue

        V_kmh = (D / duration) * 3.6
        hr_mean = g["hr"].mean()

        seg_rows.append(
            {
                "activity_id": activity_id,
                "seg_id": int(seg_id),
                "t_start": t_start,
                "t_end": t_end,
                "duration_s": duration,
                "D_m": D,
                "dh_m": dh,
                "slope_pct": slope,
                "V_kmh": V_kmh,
                "hr_mean": hr_mean,
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    return seg_df


# --------------------------------------------------------------------------------
# Модел 1 – Референтна зависимост за Glide: V = f_glide(slope)
# --------------------------------------------------------------------------------
def compute_glide_reference(segments: pd.DataFrame, deg_glide: int = 2):
    """
    Глобален Glide модел:
    - взимаме downhill сегменти (DOWN_MIN..DOWN_MAX) + предварителен downhill;
    - тренираме полином V = f(slope) върху всички downhill сегменти;
    - за всяка активност смятаме K_glide_raw + K_glide_soft и V_glide.

    Връща:
    - segments_glide: сегменти с V_glide и флагове
    - glide_summary: обобщение по активност
    - glide_poly: np.poly1d или None
    - down_df: downhill сегментите, използвани за модела
    """
    seg = segments.copy()
    if seg.empty:
        return seg, pd.DataFrame(), None, pd.DataFrame()

    seg = seg.sort_values(["activity_id", "seg_id"]).reset_index(drop=True)

    seg["is_downhill"] = (seg["slope_pct"] >= DOWN_MIN) & (seg["slope_pct"] <= DOWN_MAX)
    seg["is_prev_downhill"] = False
    for aid, g in seg.groupby("activity_id"):
        idx = g.index
        prev_down = g["is_downhill"].shift(1).fillna(False).values
        seg.loc[idx, "is_prev_downhill"] = prev_down

    downhill_mask = seg["is_downhill"] & seg["is_prev_downhill"]
    down_df = seg.loc[downhill_mask].copy()

    if len(down_df) < 20:
        seg["K_glide_raw"] = 1.0
        seg["K_glide_soft"] = 1.0
        seg["V_glide"] = seg["V_kmh"]
        summary = (
            seg.groupby("activity_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "activity_id": g["activity_id"].iloc[0],
                        "n_downhill": 0,
                        "mean_down_slope": np.nan,
                        "mean_down_V_real": np.nan,
                        "V_down_model": np.nan,
                        "K_glide_raw": 1.0,
                        "K_glide_soft": 1.0,
                    }
                )
            )
            .reset_index(drop=True)
        )
        return seg, summary, None, down_df

    # глобален Glide полином
    x = down_df["slope_pct"].values
    y = down_df["V_kmh"].values
    try:
        coeffs = np.polyfit(x, y, deg=deg_glide)
        glide_poly = np.poly1d(coeffs)
    except Exception:
        glide_poly = None

    activity_rows = []
    seg["K_glide_raw"] = 1.0
    seg["K_glide_soft"] = 1.0

    for aid, g in seg.groupby("activity_id"):
        g_down = down_df[down_df["activity_id"] == aid]
        if glide_poly is None or len(g_down) < 5:
            K_raw = 1.0
            K_soft = 1.0
            mean_down_slope = np.nan
            mean_down_V_real = np.nan
            V_down_model = np.nan
            n_down = len(g_down)
        else:
            w = g_down["duration_s"].values
            mean_down_slope = np.average(g_down["slope_pct"].values, weights=w)
            mean_down_V_real = np.average(g_down["V_kmh"].values, weights=w)
            V_down_model = float(glide_poly(mean_down_slope))
            if V_down_model <= 0:
                K_raw = 1.0
            else:
                K_raw = mean_down_V_real / V_down_model
            if not np.isfinite(K_raw) or K_raw <= 0.3 or K_raw >= 1.7:
                K_raw = 1.0
            # тук можеш да сложиш alpha_glide ако искаш омекотяване; за момента =1.0
            K_soft = K_raw
            n_down = len(g_down)

        seg.loc[seg["activity_id"] == aid, "K_glide_raw"] = K_raw
        seg.loc[seg["activity_id"] == aid, "K_glide_soft"] = K_soft

        activity_rows.append(
            {
                "activity_id": aid,
                "n_downhill": n_down,
                "mean_down_slope": mean_down_slope,
                "mean_down_V_real": mean_down_V_real,
                "V_down_model": V_down_model,
                "K_glide_raw": K_raw,
                "K_glide_soft": K_soft,
            }
        )

    seg["V_glide"] = seg["V_kmh"] / seg["K_glide_soft"]
    summary_df = pd.DataFrame(activity_rows)

    return seg, summary_df, glide_poly, down_df


# --------------------------------------------------------------------------------
# Модел 2 – Референтна зависимост за наклон: ΔV_real% = f_slope(slope)
# --------------------------------------------------------------------------------
def compute_slope_reference(segments_glide: pd.DataFrame):
    """
    Изчислява:
    - V_flat,A (по активност, от V_glide при |slope|<=FLAT_BAND)
    - ΔV_real% за всеки сегмент
    - глобален полином ΔV_model(slope)

    Връща:
    - segments_slope: сегменти с V_flat_A и DeltaV_real
    - slope_poly: np.poly1d или None
    - train_df: наборът, използван за тренировка (slope, DeltaV_real)
    - summary_df: обобщение по активност (средни стойности)
    """
    seg = segments_glide.copy()
    if seg.empty:
        return seg, None, pd.DataFrame(), pd.DataFrame()

    # V_flat по активност
    vflat_map = {}
    for aid, g in seg.groupby("activity_id"):
        flat = g[abs(g["slope_pct"]) <= FLAT_BAND]
        if len(flat) < 5:
            flat = g
        V_flat = (flat["V_glide"] * flat["duration_s"]).sum() / flat["duration_s"].sum()
        vflat_map[aid] = V_flat

    seg["V_flat_A"] = seg["activity_id"].map(vflat_map)
    seg["DeltaV_real"] = 100.0 * (seg["V_glide"] - seg["V_flat_A"]) / seg["V_flat_A"]

    train = seg[
        (seg["slope_pct"] > SLOPE_MODEL_MIN)
        & (seg["slope_pct"] < SLOPE_MODEL_MAX)
        & seg["V_flat_A"].notna()
    ].copy()

    if len(train) < 20:
        summary = (
            seg.groupby("activity_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "activity_id": g["activity_id"].iloc[0],
                        "n_slope_segments": 0,
                        "mean_slope_model": np.nan,
                        "mean_DeltaV_real": np.nan,
                    }
                )
            )
            .reset_index(drop=True)
        )
        return seg, None, train, summary

    x = train["slope_pct"].values
    y = train["DeltaV_real"].values
    try:
        coeffs = np.polyfit(x, y, deg=2)
        slope_poly = np.poly1d(coeffs)
    except Exception:
        slope_poly = None

    activity_rows = []
    for aid, g in seg.groupby("activity_id"):
        g_train = g[
            (g["slope_pct"] > SLOPE_MODEL_MIN)
            & (g["slope_pct"] < SLOPE_MODEL_MAX)
        ]
        if len(g_train) > 0:
            w = g_train["duration_s"].values
            mean_slope_model = np.average(g_train["slope_pct"].values, weights=w)
            mean_DeltaV_real = np.average(g_train["DeltaV_real"].values, weights=w)
            n_slope_segments = len(g_train)
        else:
            mean_slope_model = np.nan
            mean_DeltaV_real = np.nan
            n_slope_segments = 0

        activity_rows.append(
            {
                "activity_id": aid,
                "n_slope_segments": n_slope_segments,
                "mean_slope_model": mean_slope_model,
                "mean_DeltaV_real": mean_DeltaV_real,
            }
        )

    summary_df = pd.DataFrame(activity_rows)
    return seg, slope_poly, train, summary_df


# --------------------------------------------------------------------------------
# Streamlit UI – диагностика + Excel експорт
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Glide & Slope – диагностичен модел", layout="wide")
st.title("Glide & Slope – референтни зависимости + Excel експорт")

uploaded_files = st.file_uploader(
    "Качи един или няколко TCX файла", type=["tcx"], accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

all_segments = []
info_rows = []

for i, f in enumerate(uploaded_files, start=1):
    activity_id = f"{i}: {f.name}"

    df_points_raw = parse_tcx(f, activity_id)
    if df_points_raw.empty:
        continue

    df_raw = df_points_raw.sort_values("time").reset_index(drop=True)
    t_start = df_raw["time"].iloc[0]
    t_end = df_raw["time"].iloc[-1]
    total_time_s = (t_end - t_start).total_seconds()
    total_dist_m = df_raw["dist"].iloc[-1] - df_raw["dist"].iloc[0]
    V_overall_real = (total_dist_m / total_time_s) * 3.6 if total_time_s > 0 else np.nan

    df_clean = preprocess_points(df_points_raw)
    seg_df = build_segments(df_clean, activity_id, T_SEG)
    if seg_df.empty:
        continue

    all_segments.append(seg_df)
    info_rows.append(
        {
            "activity_id": activity_id,
            "n_segments": len(seg_df),
            "time_min": total_time_s / 60.0,
            "dist_km": total_dist_m / 1000.0,
            "V_overall_real": V_overall_real,
        }
    )

if not all_segments:
    st.error("Няма валидни сегменти.")
    st.stop()

segments = pd.concat(all_segments, ignore_index=True)
info_df = pd.DataFrame(info_rows)

st.subheader("Базова информация по активност")
st.dataframe(
    info_df.style.format(
        {"time_min": "{:.1f}", "dist_km": "{:.2f}", "V_overall_real": "{:.2f}"}
    )
)

# --------- Glide референтна зависимост ----------
st.markdown("---")
st.subheader("Референтна зависимост Glide: V = f_glide(slope)")

segments_glide, glide_summary, glide_poly, down_df = compute_glide_reference(
    segments, deg_glide=2
)

if glide_poly is None or down_df.empty:
    st.warning("Няма достатъчно downhill сегменти за Glide модел.")
else:
    coeffs = glide_poly.c
    st.markdown(
        f"""
Полином (квадратичен):

\\[
V_{{glide,model}}(s) = {coeffs[0]:.4f} \\cdot s^2 + {coeffs[1]:.4f} \\cdot s + {coeffs[2]:.4f}
\\]

където \\(s\\) е наклонът в %.
"""
    )

    st.dataframe(
        glide_summary.style.format(
            {
                "mean_down_slope": "{:.2f}",
                "mean_down_V_real": "{:.2f}",
                "V_down_model": "{:.2f}",
                "K_glide_raw": "{:.3f}",
                "K_glide_soft": "{:.3f}",
            }
        )
    )

    scatter = (
        alt.Chart(down_df)
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X("slope_pct", title="Наклон [%]"),
            y=alt.Y("V_kmh", title="Скорост [km/h]"),
            color=alt.Color("activity_id", title="Активност"),
            tooltip=["activity_id", "slope_pct", "V_kmh"],
        )
    )

    x_min = float(down_df["slope_pct"].min())
    x_max = float(down_df["slope_pct"].max())
    x_line = np.linspace(x_min, x_max, 200)
    y_line = glide_poly(x_line)
    line_df = pd.DataFrame({"slope_pct": x_line, "V_model": y_line})

    line = (
        alt.Chart(line_df)
        .mark_line()
        .encode(
            x="slope_pct",
            y=alt.Y("V_model", title="V_model [km/h]"),
            color=alt.value("black"),
        )
    )

    st.altair_chart(scatter + line, use_container_width=True)

# --------- Slope референтна зависимост ----------
st.markdown("---")
st.subheader("Референтна зависимост наклон: ΔV_real% = f_slope(slope)")

segments_slope, slope_poly, train_df, slope_summary = compute_slope_reference(
    segments_glide
)

if slope_poly is None or train_df.empty:
    st.warning("Няма достатъчно сегменти за наклонов модел.")
else:
    coeffs_s = slope_poly.c
    st.markdown(
        f"""
Полином (квадратичен):

\\[
\\Delta V_{{real,model}}(s) = {coeffs_s[0]:.4f} \\cdot s^2 + {coeffs_s[1]:.4f} \\cdot s + {coeffs_s[2]:.4f}
\\]

където \\(\\Delta V\\) е в % спрямо V_flat за съответната активност.
"""
    )

    st.dataframe(
        slope_summary.style.format(
            {
                "mean_slope_model": "{:.2f}",
                "mean_DeltaV_real": "{:.2f}",
            }
        )
    )

    scatter2 = (
        alt.Chart(train_df)
        .mark_circle(size=30, opacity=0.4)
        .encode(
            x=alt.X("slope_pct", title="Наклон [%]"),
            y=alt.Y("DeltaV_real", title="ΔV_real [%]"),
            color=alt.Color("activity_id", title="Активност"),
            tooltip=["activity_id", "slope_pct", "DeltaV_real"],
        )
    )

    x_min2 = float(train_df["slope_pct"].min())
    x_max2 = float(train_df["slope_pct"].max())
    x_line2 = np.linspace(x_min2, x_max2, 200)
    y_line2 = slope_poly(x_line2)
    line2_df = pd.DataFrame({"slope_pct": x_line2, "DeltaV_model": y_line2})

    line2 = (
        alt.Chart(line2_df)
        .mark_line()
        .encode(
            x="slope_pct",
            y=alt.Y("DeltaV_model", title="ΔV_model [%]"),
            color=alt.value("black"),
        )
    )

    st.altair_chart(scatter2 + line2, use_container_width=True)

# --------- Excel експорт по сегменти ----------
st.markdown("---")
st.subheader("Excel експорт по сегменти за избрана активност")

activity_options = sorted(segments_slope["activity_id"].unique())
selected_activity = st.selectbox("Избери активност за експорт", activity_options)

seg_act = segments_slope[segments_slope["activity_id"] == selected_activity].copy()

# Добавяме референтните стойности от двата полинома
if glide_poly is not None:
    seg_act["V_glide_ref"] = glide_poly(seg_act["slope_pct"])
else:
    seg_act["V_glide_ref"] = np.nan

if slope_poly is not None:
    seg_act["DeltaV_model"] = slope_poly(seg_act["slope_pct"])
else:
    seg_act["DeltaV_model"] = np.nan

export_cols = [
    "activity_id",
    "seg_id",
    "duration_s",
    "D_m",
    "slope_pct",
    "V_kmh",          # реална скорост
    "V_glide",        # модулирана само по Glide
    "V_glide_ref",    # референтна V от Glide полинома
    "V_flat_A",       # референтна скорост на равно за активността
    "DeltaV_real",    # реално отклонение спрямо V_flat_A
    "DeltaV_model",   # моделно отклонение спрямо наклона
]

export_df = seg_act[export_cols].copy()

st.markdown("Преглед на сегментите за избраната активност:")
st.dataframe(
    export_df.head(50).style.format(
        {
            "duration_s": "{:.1f}",
            "D_m": "{:.1f}",
            "slope_pct": "{:.2f}",
            "V_kmh": "{:.2f}",
            "V_glide": "{:.2f}",
            "V_glide_ref": "{:.2f}",
            "V_flat_A": "{:.2f}",
            "DeltaV_real": "{:.2f}",
            "DeltaV_model": "{:.2f}",
        }
    )
)

# Excel download
# Експорт като CSV (Excel го отваря директно)
csv_bytes = export_df.to_csv(index=False).encode("utf-8")

safe_name = (
    selected_activity.replace(":", "_")
    .replace(" ", "_")
    .replace(".", "_")
)

st.download_button(
    label="Свали CSV (сегменти + референтни зависимости)",
    data=csv_bytes,
    file_name=f"segments_{safe_name}.csv",
    mime="text/csv",
)

