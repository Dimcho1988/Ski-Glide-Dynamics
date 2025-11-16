import io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.nonparametric.smoothers_lowess import lowess

G = 9.80665  # гравитационно ускорение (m/s^2)


# ---------- ПАРСВАНЕ НА TCX ----------

def parse_tcx(file_obj) -> pd.DataFrame:
    """Чете TCX и връща DataFrame с време, дистанция, височина и сурова скорост."""
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

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(times),
            "distance_m": dist,
            "altitude_m": alt,
        }
    )

    df["t_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()

    dt = df["t_sec"].diff().replace(0, np.nan)
    df["speed"] = df["distance_m"].diff().divide(dt).fillna(0.0)

    return df


# ---------- МАТЕМАТИЧЕСКИ ПОМОЩНИ ФУНКЦИИ ----------

def central_diff(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Централна разлика за по-гладка производна."""
    dy = np.zeros_like(y)
    dx = np.zeros_like(y)

    # вътрешни точки
    dy[1:-1] = y[2:] - y[:-2]
    dx[1:-1] = x[2:] - x[:-2]

    # краища
    dy[0] = y[1] - y[0]
    dx[0] = x[1] - x[0]
    dy[-1] = y[-1] - y[-2]
    dx[-1] = x[-1] - x[-2]

    dx[dx == 0] = np.nan
    return dy / dx


# ---------- ИЗГЛАЖДАНЕ + НАКЛОН + УСКОРЕНИЕ ----------

def process_df(df: pd.DataFrame, lowess_frac: float = 0.02) -> pd.DataFrame:
    """Изглаждане на данните и смятане на наклон и ускорение върху трендовете."""
    t = df["t_sec"].values

    # тренд линии
    df["alt_smooth"] = lowess(df["altitude_m"], t, frac=lowess_frac, return_sorted=False)
    df["distance_smooth"] = lowess(df["distance_m"], t, frac=lowess_frac, return_sorted=False)
    df["speed_smooth"] = lowess(df["speed"], t, frac=lowess_frac, return_sorted=False)

    alt = df["alt_smooth"].values
    dist = df["distance_smooth"].values
    speed = df["speed_smooth"].values

    dalt_dt = central_diff(alt, t)
    ddist_dt = central_diff(dist, t)
    dspeed_dt = central_diff(speed, t)

    # наклон по тренд
    grade_raw = (dalt_dt / ddist_dt) * 100.0
    grade_smooth = lowess(grade_raw, t, frac=0.03, return_sorted=False)

    # ускорение по тренд
    acc_raw = dspeed_dt
    acc_smooth = lowess(acc_raw, t, frac=0.05, return_sorted=False)

    df["grade_percent"] = grade_smooth
    df["acc_m_s2"] = acc_smooth

    return df


# ---------- СПУСКАНИЯ И ИНДЕКС НА ТРИЕНЕ ----------

def find_downhill_segments(
    df: pd.DataFrame,
    min_total_duration_s: float = 10.0,
    cut_first_s: float = 5.0,
) -> pd.DataFrame:
    """
    Намира сегменти с наклон < 0% (спускания), реже първите cut_first_s секунди
    и връща таблица със среден наклон, ускорение и индекс на триене.
    friction_index = a_theor / a_real
    """
    t = df["t_sec"].values
    grade = df["grade_percent"].values
    acc = df["acc_m_s2"].values

    downhill_mask = grade < 0.0

    segments = []
    if not downhill_mask.any():
        return pd.DataFrame(
            columns=[
                "seg_id",
                "t_start",
                "t_end",
                "duration_s",
                "mean_grade_percent",
                "mean_acc_m_s2",
                "theoretical_acc_m_s2",
                "friction_index",
            ]
        )

    idx = np.where(downhill_mask)[0]
    start = idx[0]
    prev = idx[0]
    seg_id = 1

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segments.append((start, prev))
        start = i
        prev = i
    segments.append((start, prev))

    rows = []
    for (s_idx, e_idx) in segments:
        t_start = t[s_idx]
        t_end = t[e_idx]
        duration = t_end - t_start
        if duration < min_total_duration_s:
            continue

        # режем първите cut_first_s сек
        t_cut_start = t_start + cut_first_s
        full_idx = np.arange(len(t))
        seg_mask = (full_idx >= s_idx) & (full_idx <= e_idx) & (t >= t_cut_start)

        if not seg_mask.any():
            continue

        seg_t = t[seg_mask]
        seg_grade = grade[seg_mask]
        seg_acc = acc[seg_mask]

        mean_grade = np.nanmean(seg_grade)
        mean_acc = np.nanmean(seg_acc)

        if np.isnan(mean_grade) or np.isnan(mean_acc):
            continue

        # теоретично ускорение по наклона (без триене)
        grade_frac = -mean_grade / 100.0  # downhill → положително число
        if grade_frac <= 0:
            continue

        a_theor = G * grade_frac

        if abs(mean_acc) < 1e-3:
            friction_index = np.nan
        else:
            friction_index = a_theor / mean_acc

        rows.append(
            {
                "seg_id": seg_id,
                "t_start": float(seg_t[0]),
                "t_end": float(seg_t[-1]),
                "duration_s": float(seg_t[-1] - seg_t[0]),
                "mean_grade_percent": mean_grade,
                "mean_acc_m_s2": mean_acc,
                "theoretical_acc_m_s2": a_theor,
                "friction_index": friction_index,
            }
        )
        seg_id += 1

    return pd.DataFrame(rows)


def aggregate_friction(seg_df: pd.DataFrame) -> float:
    """Един агрегатен индекс на триене за цялата активност (медиана на сегментите)."""
    if seg_df is None or seg_df.empty:
        return np.nan
    return float(np.nanmedian(seg_df["friction_index"].values))


# ---------- STREAMLIT UI ----------

def main():
    st.title("onFlows — Модулиране на скоростта при ски / ролки")

    st.markdown(
        """
Приложението:
- Изглажда скорост, височина и дистанция от TCX файлове  
- Изчислява наклон и ускорение по тренда  
- Намира спускания и оценява **индекс на триене** за всяка активност  
- Позволява избор на **референтна активност**  
- Модулира скоростта на останалите активности, сякаш са карани
  при референтна пълзаемост (μ_ref)
"""
    )

    uploaded_files = st.file_uploader(
        "Качи един или няколко TCX файла",
        type=["tcx"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Качи поне един TCX файл, за да започнем.")
        return

    # обработка на всички файлове
    results = {}  # name -> dict(df, segments, friction)

    for file in uploaded_files:
        try:
            df_raw = parse_tcx(file)
            df_proc = process_df(df_raw)
            seg_df = find_downhill_segments(df_proc)
            friction = aggregate_friction(seg_df)

            results[file.name] = {
                "df": df_proc,
                "segments": seg_df,
                "friction": friction,
            }

        except Exception as e:
            st.error(f"Грешка при обработка на {file.name}: {e}")

    file_names = list(results.keys())
    if not file_names:
        st.error("Не успяхме да обработим нито един файл.")
        return

    # избор на референтна активност
    st.subheader("Стъпка 1 – Избор на референтна активност")
    ref_name = st.selectbox("Референтна активност", file_names)

    friction_ref = results[ref_name]["friction"]
    if np.isnan(friction_ref):
        st.warning(
            "Референтната активност няма валиден индекс на триене "
            "(липса на достатъчно спускания). Модулацията може да е некоректна."
        )

    # таблица с индексите за всички активности
    summary_rows = []
    for name, obj in results.items():
        fr = obj["friction"]
        if np.isnan(friction_ref) or np.isnan(fr):
            mod_coef = np.nan
        else:
            # коефициент на модулация спрямо референтната
            # ако триенето е по-голямо (по-лош glide),
            # модул. коефициентът > 1 → скоростта се повишава
            mod_coef = fr / friction_ref

        summary_rows.append(
            {
                "activity": name,
                "friction_index": fr,
                "modulation_coef_vs_ref": mod_coef,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    st.subheader("Индекс на триене и коефициент на модулация")
    st.dataframe(
        summary_df.style.format(
            {
                "friction_index": "{:.3f}",
                "modulation_coef_vs_ref": "{:.3f}",
            }
        )
    )

    # избор на активност за визуализация (реална vs модулирана скорост)
    st.subheader("Стъпка 2 – Визуализация на модулираната скорост")

    vis_name = st.selectbox(
        "Активност за визуализация (real vs modulated)",
        file_names,
        index=0,
    )

    df_vis = results[vis_name]["df"]
    fr_vis = results[vis_name]["friction"]

    if np.isnan(friction_ref) or np.isnan(fr_vis):
        st.warning(
            "Няма валиден коефициент на модулация (липсва индекс на триене "
            "за референтната или избраната активност). Показваме само реалната скорост."
        )
        vis_df_plot = df_vis[["t_sec", "speed_smooth"]].set_index("t_sec")
        vis_df_plot.rename(columns={"speed_smooth": "speed_real"}, inplace=True)
        st.line_chart(vis_df_plot)
    else:
        mod_coef = fr_vis / friction_ref
        df_vis = df_vis.copy()
        df_vis["speed_modulated"] = df_vis["speed_smooth"] * mod_coef

        # графика real vs modulated
        plot_df = df_vis[["t_sec", "speed_smooth", "speed_modulated"]].set_index(
            "t_sec"
        )
        plot_df.rename(
            columns={
                "speed_smooth": "speed_real",
                "speed_modulated": "speed_modulated_vs_ref",
            },
            inplace=True,
        )

        st.markdown(
            f"**Коефициент на модулация за тази активност спрямо референтната**: "
            f"`k = {mod_coef:.3f}`"
        )
        st.line_chart(plot_df)

        # таблица с няколко резюмета по сегменти (по желание – засега оставяме скорост по време)
        st.markdown("Пример от данните (първите 200 реда):")
        st.dataframe(
            df_vis[["t_sec", "speed_smooth", "speed_modulated"]]
            .head(200)
            .rename(
                columns={
                    "speed_smooth": "speed_real",
                    "speed_modulated": "speed_modulated_vs_ref",
                }
            )
        )

        # бутон за сваляне на модифицирания CSV
        csv_bytes = df_vis.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Свали CSV (реална + модулирана скорост)",
            data=csv_bytes,
            file_name=vis_name.replace(".tcx", "_modulated_vs_ref.csv"),
            mime="text/csv",
        )

    st.caption(
        "Важно: моделът за модулация в момента е линейна апроксимация "
        "на база индекс на триене (a_theor / a_real). В бъдеще може да "
        "го калибрираме емпирично спрямо реални тестове."
    )


if __name__ == "__main__":
    main()
