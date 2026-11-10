"""Shared state-dependent rules. Scenario labels are deliberately absent."""
from datetime import datetime
import numpy as np
from controller_config import CONFIG as C


def is_occupied(dt_now: datetime):
    h = dt_now.hour + dt_now.minute / 60
    return dt_now.weekday() >= 5 or h < 7 or h >= 20


def comfort_bounds(dt_now):
    return (21.0, 24.0) if is_occupied(dt_now) else (15.0, 30.0)


def rule_teacher_target_temperature(occ, T_heat_sp_K, Tz_C, Tout_C, Qsol,
                                    price=None, use_weather_correction=True):
    target = T_heat_sp_K - 273.15 + C.occupied_offset if occ else C.relaxed_target
    if use_weather_correction:
        target -= 0.10 * (Qsol > 300) + 0.20 * (Qsol > 600)
        target -= 0.10 * (Tout_C > 12) + 0.20 * (Tout_C > 16)
        target -= 0.20 * (Tz_C > 23) + 0.30 * (Tz_C > 23.5)
        target += 0.15 * (Tz_C < 21) + 0.20 * (Tz_C < 20.6)
        target += 0.15 * (Tout_C < 2)
        if price is not None:
            target -= 0.10 * (price > 0.30 and Tz_C > 21.5)
            target += 0.05 * (price < 0.22 and Tz_C < 22 and Tout_C < 8)
    return float(np.clip(target, C.rule_min, C.rule_max))


def combine_rule_and_nn_target(T_rule_C, delta_T_nn_C, beta):
    return float(np.clip(T_rule_C + np.clip(beta, 0, 1) *
                         np.clip(delta_T_nn_C, -C.target_bound, C.target_bound),
                         C.target_min, C.target_max))


def high_level_teacher_delta_T(Tz_C, T_rule_C, Tout_C, Qsol, price, occ,
                              future_features=None):
    f = future_features or {}
    delta = 0.0
    for threshold, amount in [(22.5,.15),(22.8,.25),(23.2,.35),(23.6,.45)]:
        delta -= amount * (Tz_C > threshold)
    for threshold, amount in [(250,.15),(500,.35),(700,.45)]:
        delta -= amount * (Qsol > threshold)
    for threshold, amount in [(10,.10),(14,.20),(17,.25)]:
        delta -= amount * (Tout_C > threshold)
    for threshold, amount in [(21.1,.15),(20.8,.25),(20.5,.35)]:
        delta += amount * (Tz_C < threshold)
    delta += .15 * (Tout_C < 5) + .20 * (Tout_C < 0)
    for p, margin, amount in [(.30,.5,.15),(.35,.7,.25),(.40,1.,.35)]:
        delta -= amount * (price > p and Tz_C - 21 > margin)
    delta += .08 * (price < .22 and Tz_C < 21.6 and Tout_C < 8)
    delta += .12 * (price < .20 and Tz_C < 21.4 and Tout_C < 8)
    if f:
        solar = f['Qsol_max_6h']; outdoor = f['Tout_max_6h']
        if f['Thigh_min_6h'] <= 24.5:
            delta -= .20 * (solar > 400 and Tz_C > 21.2)
            delta -= .30 * (solar > 600 and Tz_C > 21)
            delta -= .25 * (f['Qsol_sum_6h'] > 1800 and Tz_C > 21)
        delta -= .15 * (outdoor > 15 and Tz_C > 21.2)
        delta -= .25 * (outdoor > 17 and Tz_C > 21)
        delta -= .12 * (f['price_mean_6h'] > .30 and Tz_C > 21.5)
        delta -= .18 * (f['price_max_6h'] > .35 and Tz_C > 21.7)
        delta -= .25 * (f['price_max_6h'] > .40 and Tz_C > 22)
        spread = f['price_max_6h'] - price
        strict_ahead = f['Tlow_max_6h'] >= 21
        delta += .08 * (spread > .08 and price < .24 and Tz_C < 21.6
                         and strict_ahead and solar < 500)
        delta += .12 * (spread > .12 and price < .22 and Tz_C < 21.4
                         and strict_ahead and solar < 400)
        if strict_ahead and solar < 400 and outdoor < 12:
            delta += .20 * (Tz_C < 21.2) + .30 * (Tz_C < 21)
        # Protect the lower boundary only when occupied or a strict interval is ahead.
        if occ or strict_ahead:
            if Tz_C < 21.3:
                delta = max(delta, 0.0)
            elif Tz_C < 21.6:
                delta = max(delta, -.10)
    return float(np.clip(delta, -C.target_bound, C.target_bound))


def protect_target(Tsp_C, Tz_C, dt_now, future_features):
    """Apply the same physical-risk protections to every evaluation window."""
    f = future_features
    if f['Qsol_max_6h'] > 600 and Tz_C > 21:
        Tsp_C = min(Tsp_C, 20.0)
    if f['Qsol_sum_6h'] > 2500 and Tz_C > 21:
        Tsp_C = min(Tsp_C, 19.9)
    if f['Tout_max_6h'] > 16 and Tz_C > 21.2:
        Tsp_C = min(Tsp_C, 20.1)
    h = dt_now.hour + dt_now.minute / 60
    if dt_now.weekday() < 5 and 19 <= h < 20:
        # State-based preheating; no Peak/Typical label or parameter switching.
        if Tz_C < 21.2:
            Tsp_C = max(Tsp_C, 21.7)
        elif Tz_C < 21.4:
            Tsp_C = max(Tsp_C, 21.5)
    low = future_features["current_low_C"]
    if low >= 21:
        Tsp_C = max(Tsp_C, 21.1 if Tz_C < 21.3 else 21.0)
    return float(np.clip(Tsp_C, C.target_min, C.target_max))


def action_teacher_delta_u_by_grid(Tz_C, Tsp_C, T_low_C, T_high_C, Tout_C,
                                  Qsol, price, d_hat, theta, u_base, u_prev,
                                  delta_u_max=0.08, n_grid=21, predictor=None):
    candidates = np.clip(u_base + np.linspace(-delta_u_max, delta_u_max, n_grid), 0, 1)
    pred = (theta[0]*Tz_C + theta[1]*Tout_C + theta[2]*Qsol/1000
            + theta[3]*candidates + theta[4] + d_hat)
    if predictor is not None:
        pred = predictor.predict_actions(Tz_C, Tout_C, Qsol, candidates) + d_hat
    # Official bounds determine violations; target tracking already expresses a preference.
    cost = (C.teacher_low_weight * np.maximum(0, T_low_C-pred)**2
            + C.teacher_high_weight * np.maximum(0, pred-T_high_C)**2
            + C.teacher_track_weight * (pred-Tsp_C)**2
            + C.teacher_energy_weight * max(0, price) * candidates**2
            + C.teacher_smooth_weight * (candidates-u_prev)**2)
    return float(np.clip(candidates[np.argmin(cost)]-u_base, -delta_u_max, delta_u_max))


def gate_action(delta, margin_low, margin_high, u_base, Tz_C, future_features):
    f = future_features
    overheat = (f['Qsol_max_6h'] > 500 or f['Qsol_sum_6h'] > 1800
                or f['Tout_max_6h'] > 16) and Tz_C > 20.8
    if margin_low < C.directional_margin and delta < 0:
        if not (overheat and margin_low > C.hard_margin):
            delta = 0.0
    if margin_high < C.directional_margin and delta > 0:
        delta = 0.0
    if margin_low < C.hard_margin and delta < 0:
        delta = 0.0
    if margin_high < C.hard_margin and delta > 0:
        delta = 0.0
    if u_base <= .02 and delta < 0:
        delta = 0.0
    return float(np.clip(delta, -C.action_bound, C.action_bound))
