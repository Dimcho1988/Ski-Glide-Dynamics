# cs_modulator.py
import numpy as np


def tau_of_delta(delta_v, tmin, k, q):
    """
    τ(Δv) = τ_min + k * (Δv)^q
    delta_v в km/h
    """
    delta_v = np.maximum(np.asarray(delta_v, dtype=float), 0.0)
    return tmin + k * (delta_v ** q)


def predict_t90_for_reference(CS, ref_percent, tmin, k, q):
    """
    При постоянна скорост v = ref_percent * CS:
    t90 ≈ 2.303 * τ(Δv_ref)
    """
    dv_ref = (ref_percent / 100.0 - 1.0) * CS
    tau_ref = tau_of_delta(dv_ref, tmin, k, q)
    t90_est = 2.303 * tau_ref
    return dv_ref, tau_ref, t90_est


def calibrate_k_for_target_t90(CS, ref_percent, tmin, q, target_t90):
    """
    Намира k така, че t90 при v = ref_percent * CS да е target_t90.
    Формула от t90 = 2.303 * (tmin + k * (Δv_ref)^q).
    """
    dv_ref = (ref_percent / 100.0 - 1.0) * CS
    dvq = max(dv_ref, 0.0) ** q if dv_ref > 0 else 0.0
    tau_ref_needed = target_t90 / 2.303

    if dvq <= 0:
        return 0.0

    k_needed = max(0.0, (tau_ref_needed - tmin) / dvq)
    return k_needed


def apply_cs_modulation(v, dt, CS, tau_min, k_par, q_par, gamma):
    """
    Основен CS модел:

    вход:
      v  – скорост [km/h] (np.array)
      dt – стъпка по време за всеки елемент [s] (np.array)
      CS, tau_min, k_par, q_par, gamma – параметри на модела

    връща dict с:
      v_mod         – модулирана скорост [km/h]
      delta_v_plus  – Δv⁺ = max(v - CS, 0)
      r             – „повдигане“ на скоростта [km/h]
      tau_s         – ефективна τ(t) [s]
    """
    v = np.asarray(v, dtype=float)
    dt = np.asarray(dt, dtype=float)

    if v.shape != dt.shape:
        raise ValueError("v и dt трябва да са с еднаква дължина.")

    # гаранция за положителни dt
    dt = np.maximum(dt, 1e-6)

    delta_v_plus = np.maximum(v - CS, 0.0)

    A = np.zeros_like(v)          # натрупан „дълг“
    r = np.zeros_like(v)          # повдигане
    tau_series = np.zeros_like(v)

    tau_last = tau_min

    for i in range(len(v)):
        dvp = delta_v_plus[i]
        if dvp > 0:
            tau_i = tau_of_delta(dvp, tau_min, k_par, q_par)
            tau_last = tau_i
        else:
            tau_i = tau_last

        tau_series[i] = tau_i

        decay = np.exp(-dt[i] / tau_i)
        if i == 0:
            A_prev = 0.0
        else:
            A_prev = A[i - 1]

        A[i] = A_prev * decay + dvp * dt[i]

        if dt[i] > 0:
            r[i] = (1.0 - decay) * A[i] / dt[i]
        else:
            r[i] = 0.0

    v_mod = v + gamma * r

    return {
        "v_mod": v_mod,
        "delta_v_plus": delta_v_plus,
        "r": r,
        "tau_s": tau_series,
    }
