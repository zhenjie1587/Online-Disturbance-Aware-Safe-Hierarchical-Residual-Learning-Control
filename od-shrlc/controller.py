# PID_disturbance_online_id.py
# -*- coding: utf-8 -*-

import os
import time
from experiment_support import MeasurementStream, forecast_features, OnlineARXPredictor
from controller_config import CONFIG, participation
from unified_rules import protect_target, gate_action
import json
from datetime import datetime, timedelta
from collections import deque
from unified_rules import (
    is_occupied,
    comfort_bounds,
    rule_teacher_target_temperature,
    combine_rule_and_nn_target,
    high_level_teacher_delta_T,
    action_teacher_delta_u_by_grid,
)

from nn_compensator_tcn_gru_v2 import (
    TargetTempResidualNN,
    ActionResidualNN,
    SimpleReplayBuffer,
    build_fixed_length_sequence,
    make_target_nn_features,
    make_action_nn_features,
    train_target_nn,
    train_action_nn,
    beta_schedule,
    alpha_u_schedule,
    predict_delta_T_nn,
    predict_delta_u_nn,
)
import torch
import torch.optim as optim
import random
import argparse
import numpy as np
import pandas as pd
import requests

# ====== 画图用（无界面环境也能保存图片）======
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. BOPTEST 基本配置
# ============================================================

BASE_URL = os.environ.get("BOPTEST_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
TESTCASE_NAME = "bestest_hydronic_heat_pump"

STEP = 900.0               # 15 min
WARMUP_DAYS = 7
SIM_DAYS = 14

SCENARIO_START_DAY = {
    "peak": 16,
    "typical": 108,
}

PRICE_SCENARIO = "highly_dynamic"


# ============================================================
# 1-扩展. 消融实验配置与随机种子
# ============================================================

DEFAULT_SEED = 2026


def set_random_seed(seed: int = DEFAULT_SEED):
    """Set random seeds for more reproducible online NN training."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kpi_get(kpi: dict, candidates, default=np.nan):
    """Read one KPI value from possible BOPTEST key names."""
    for key in candidates:
        if key in kpi:
            try:
                return float(kpi[key])
            except Exception:
                return default
    return default


# ============================================================
# 2. REST API 工具函数
# ============================================================

def api_get(path, params=None, timeout=60):
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
    if not r.ok:
        print("GET failed:", r.status_code, r.text)
    r.raise_for_status()
    return r.json()


def api_put(path, json=None, timeout=60):
    r = requests.put(f"{BASE_URL}{path}", json=json, timeout=timeout)
    if not r.ok:
        print("PUT failed:", r.status_code, r.text)
    r.raise_for_status()

    # /stop 有时可能返回空内容，这里兼容一下
    try:
        return r.json()
    except Exception:
        return {"status": r.status_code, "payload": None}


def api_post(path, json=None, timeout=60):
    r = requests.post(f"{BASE_URL}{path}", json=json, timeout=timeout)
    if not r.ok:
        print("POST failed:", r.status_code, r.text)
    r.raise_for_status()
    return r.json()


def select_testcase(out_dir):
    available=api_get("/testcases")
    if TESTCASE_NAME not in [v["testcaseid"] for v in available]:
        raise RuntimeError("Requested testcase unavailable")
    # Async selection returns ownership immediately, allowing safe queue cleanup.
    res=api_post(f"/testcases/{TESTCASE_NAME}/select-async")
    testid=res.get("testid") or res.get("payload",{}).get("testid")
    if not testid:raise RuntimeError("No test ID returned")
    with open(os.path.join(out_dir,"session.json"),"w") as f:
        json.dump({"testid":testid},f)
    try:
        deadline=time.monotonic()+300
        while time.monotonic()<deadline:
            status=api_get(f"/status/{testid}")
            value=status.get("payload",status.get("status")) if isinstance(status,dict) else status
            if value=="Running":return testid
            time.sleep(.5)
        raise TimeoutError("BOPTEST worker not available within 300 seconds")
    except BaseException:
        stop(testid)
        raise


def set_step(testid, step):
    api_put(f"/step/{testid}", json={"step": float(step)})


def set_scenario(testid, electricity_price):
    try:
        api_put(f"/scenario/{testid}", json={"electricity_price": electricity_price})
    except Exception:
        raise  # A tariff-setting failure invalidates the experiment.


def initialize(testid, start_time, warmup_period):
    res = api_put(
        f"/initialize/{testid}",
        json={
            "start_time": float(start_time),
            "warmup_period": float(warmup_period),
        },
    )
    return res["payload"]


def stop(testid):
    try:
        api_put(f"/stop/{testid}")
    except Exception:
        pass


def advance(testid, inputs):
    res = api_post(f"/advance/{testid}", json=inputs)
    return res["payload"]


def get_kpi(testid):
    for path in (f"/kpi/{testid}", "/kpi"):
        try:
            res = api_get(path)
            return res.get("payload", res)
        except Exception:
            continue
    raise RuntimeError("KPI unavailable")


def put_forecast(testid, point_names, horizon, interval):
    res = api_put(
        f"/forecast/{testid}",
        json={
            "point_names": point_names,
            "horizon": float(horizon),
            "interval": float(interval),
        },
    )
    return res["payload"]


# ============================================================
# 3. 占用与舒适区间
# ============================================================



# ============================================================
# 4. PID 控制器：带 anti-windup 和 D 项滤波
# ============================================================

class PIDController:
    """
    PID 控制器。

    u = Kp*e + Ki*I + Kd*de/dt

    注意：
    - I 项做 anti-windup；
    - D 项做一阶滤波，避免噪声放大；
    - 如果你想退化成 PI，把 kd 设置为 0 即可。
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        i_min: float = -2e5,
        i_max: float = 2e5,
        d_filter_alpha: float = 0.2,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.i_min = float(i_min)
        self.i_max = float(i_max)

        self.d_filter_alpha = float(d_filter_alpha)

        self.I = 0.0
        self.e_prev = None
        self.d_filt = 0.0

    def reset(self):
        self.I = 0.0
        self.e_prev = None
        self.d_filt = 0.0

    def step(self, e, dt):
        e = float(e)
        dt = float(dt)

        # ---------- D 项 ----------
        if self.e_prev is None:
            d_raw = 0.0
        else:
            d_raw = (e - self.e_prev) / dt

        self.d_filt = (
            (1.0 - self.d_filter_alpha) * self.d_filt
            + self.d_filter_alpha * d_raw
        )

        self.e_prev = e

        # ---------- 先用当前积分计算未饱和输出 ----------
        u_unsat = self.kp * e + self.ki * self.I + self.kd * self.d_filt
        u_sat = float(np.clip(u_unsat, 0.0, 1.0))

        # ---------- anti-windup ----------
        pushing_up = (u_sat >= 1.0 - 1e-9) and (e > 0)
        pushing_down = (u_sat <= 0.0 + 1e-9) and (e < 0)

        if not (pushing_up or pushing_down):
            self.I += e * dt
            self.I = float(np.clip(self.I, self.i_min, self.i_max))

        # ---------- 更新积分后重新计算 ----------
        u_raw = self.kp * e + self.ki * self.I + self.kd * self.d_filt
        u = float(np.clip(u_raw, 0.0, 1.0))

        return u, u_raw, self.I, self.d_filt


# ============================================================
# 5. 在线 RC 等效模型辨识
# ============================================================

class OnlineLinearRCIdentifier:
    """
    在线辨识简化离散热模型：

        Tz[k+1] = a*Tz[k] + b*Tout[k] + c*Qsol[k] + g*u[k] + bias

    其中：
    - Tz, Tout 单位：°C
    - Qsol 使用 kW/m2，即 W/m2 / 1000
    - u 为热泵动作 [0,1]

    这里不假设你一开始知道 RC 参数，也不假设知道热泵 u 到 Q 的换算。
    g 是“u 对下一步室温影响”的等效增益。
    """

    def __init__(self, forgetting=0.995):
        # theta = [a, b, c, g, bias]
        self.theta = np.array([0.95, 0.03, 0.05, 0.30, 0.0], dtype=float)

        # 初始协方差大一点，表示初始模型不确定
        self.P = np.eye(5) * 100.0

        # 遗忘因子
        self.forgetting = float(forgetting)

    def make_phi(self, Tz_C, Tout_C, Qsol_Wm2, u_hp):
        qsol_kw = float(Qsol_Wm2) / 1000.0
        return np.array(
            [
                float(Tz_C),
                float(Tout_C),
                qsol_kw,
                float(u_hp),
                1.0,
            ],
            dtype=float,
        )

    def predict(self, phi):
        return float(np.dot(self.theta, phi))

    def update(self, phi_prev, Tz_next_C):
        """
        用上一时刻的 phi[k] 和当前真实温度 Tz[k+1] 更新参数。
        """
        phi = phi_prev.reshape(-1, 1)
        y = float(Tz_next_C)

        theta_col = self.theta.reshape(-1, 1)

        denom = self.forgetting + (phi.T @ self.P @ phi).item()
        K = (self.P @ phi) / denom

        pred = (theta_col.T @ phi).item()
        err = y - pred

        theta_col = theta_col + K * err
        self.P = (self.P - K @ phi.T @ self.P) / self.forgetting

        self.theta = theta_col.flatten()

        # 参数投影，避免冷启动早期发散
        self.theta[0] = np.clip(self.theta[0], 0.70, 1.05)    # a
        self.theta[1] = np.clip(self.theta[1], -0.20, 0.20)   # b
        self.theta[2] = np.clip(self.theta[2], -2.00, 2.00)   # c
        self.theta[3] = np.clip(self.theta[3], 0.00, 3.00)    # g
        self.theta[4] = np.clip(self.theta[4], -10.0, 10.0)   # bias

        return pred, err, self.theta.copy()


# ============================================================
# 6. 扰动观测器
# ============================================================

class DisturbanceObserver:
    """
    扰动观测器。

    d_hat 单位是 °C，表示“当前模型没有解释掉的等效温度扰动”。

    d_hat > 0:
        真实温度高于模型预测，可能有太阳、人员、设备等额外热增益；
        供热时应该减小热泵动作。

    d_hat < 0:
        真实温度低于模型预测，可能有开窗、渗透风、额外热损失；
        供热时应该增大热泵动作。

    关键修改：
        原代码使用固定比例 u_d = k_ud * d_hat。
        这里改为与在线模型一致的逆增益补偿：

            u_d = sat(kappa_d * d_hat / max(L_k, L_min))

        其中 L_k 是在线估计的 u 前面的等效控制增益，即 theta[3]。
        这样对应：

            T(k+1) = ... + L_k u(k) + d(k)
            u(k)   = u_b(k) - d(k) / L_k

        为避免冷启动阶段 L_k 估计不准导致过补偿，保留 kappa_d、L_min 和 u_d_max。
    """

    def __init__(
        self,
        lam=0.10,
        d_min=-2.0,
        d_max=2.0,
        k_ud=0.08,          # 仅作为没有 gain_L 时的 fallback
        u_d_max=0.20,
        kappa_d=0.10,       # 逆增益补偿保守系数；想完全抵消可设为 1.0
        L_min=0.30,         # 防止 L_k 太小导致 d/L 过大
    ):
        self.lam = float(lam)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.k_ud = float(k_ud)
        self.u_d_max = float(u_d_max)
        self.kappa_d = float(kappa_d)
        self.L_min = float(L_min)

        self.d_hat = 0.0

        # 日志用：记录最近一次补偿计算的中间量
        self.last_u_d_raw = 0.0
        self.last_L_eff = np.nan
        self.last_comp_mode = "init"

    def reset(self):
        self.d_hat = 0.0
        self.last_u_d_raw = 0.0
        self.last_L_eff = np.nan
        self.last_comp_mode = "reset"

    def update(self, pred_error):
        """
        pred_error = Tz_measured - Tz_predicted_base
        """
        self.d_hat = (1.0 - self.lam) * self.d_hat + self.lam * float(pred_error)
        self.d_hat = float(np.clip(self.d_hat, self.d_min, self.d_max))
        return self.d_hat

    def compensation_u(self, gain_L=None):
        """
        将温度等效扰动换算成动作补偿。

        推荐用法：
            gain_L = theta_now[3]

        若在线增益可用，则采用：
            u_d = kappa_d * d_hat / max(gain_L, L_min)

        若 gain_L 不可用，则回退到原来的固定比例：
            u_d = k_ud * d_hat

        最终基础动作仍按：
            u_base = clip(u_pid - alpha_d * u_d, 0, 1)
        """
        if gain_L is None or (not np.isfinite(float(gain_L))):
            u_d_raw = self.k_ud * self.d_hat
            self.last_L_eff = np.nan
            self.last_comp_mode = "fixed_k_ud"
        else:
            # 你的在线辨识里 L_k = theta[3] 已投影到 [0, 3]，这里再加下界保护。
            L_eff = max(float(gain_L), self.L_min)
            u_d_raw = self.kappa_d * self.d_hat / L_eff
            self.last_L_eff = L_eff
            self.last_comp_mode = "inverse_gain"

        self.last_u_d_raw = float(u_d_raw)
        u_d = float(np.clip(u_d_raw, -self.u_d_max, self.u_d_max))
        return u_d


# ============================================================
# 7. 动作联动逻辑
# ============================================================

def action_link_baseline(u_hp: float):
    """
    严格对齐你原 baseline：
    u_hp > 0 则 fan/pump 开；
    u_hp = 0 则全部关。
    """
    u = float(np.clip(u_hp, 0.0, 1.0))

    if u > 0.0:
        return u, 1.0, 1.0

    return 0.0, 0.0, 0.0


# ============================================================
# 8. forecast 工具
# ============================================================

def _pick_series(payload: dict, keys_try):
    for k in keys_try:
        if k in payload:
            return np.array(payload[k], dtype=float)

    raise KeyError(f"Cannot find any of these keys in forecast payload: {keys_try}")


def _fit_len(arr, n):
    arr = np.asarray(arr, dtype=float)

    if len(arr) >= n:
        return arr[:n]

    return np.pad(arr, (0, n - len(arr)), mode="edge")


# ============================================================
# 8-扩展. 预测窗口特征工具
# ============================================================

def _to_celsius_series(arr):
    """
    Convert forecast temperature series to Celsius if it looks like Kelvin.
    """
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    if np.nanmean(arr) > 100.0:
        return arr - 273.15
    return arr


def get_forecast_points(testid):
    """
    Query available BOPTEST forecast point names.
    """
    try:
        res = api_get(f"/forecast_points/{testid}")
        payload = res.get("payload", res)
        if isinstance(payload, dict):
            return list(payload.keys())
        if isinstance(payload, list):
            return payload
    except Exception as e:
        print("[WARN] Failed to query forecast_points:", e)
    return []


def choose_name(available_names, candidates):
    """
    Choose the first forecast point name that exists in available_names.
    """
    available = set(available_names)
    for name in candidates:
        if name in available:
            return name
    return None


def get_future_window(arr, i, horizon_steps):
    """
    Return arr[i:i+horizon_steps], padded by edge value if needed.
    """
    arr = np.asarray(arr, dtype=float)

    if len(arr) == 0:
        return np.zeros(horizon_steps, dtype=float)

    i = int(np.clip(i, 0, len(arr) - 1))
    end = min(len(arr), i + horizon_steps)
    win = arr[i:end]

    if len(win) == 0:
        win = np.array([arr[i]], dtype=float)

    if len(win) < horizon_steps:
        win = np.pad(win, (0, horizon_steps - len(win)), mode="edge")

    return win


def future_comfort_bounds(dt_now, horizon_steps, step_seconds):
    """
    Compute future comfort lower/upper bounds using the same schedule as comfort_bounds().
    """
    lows = []
    highs = []

    for j in range(horizon_steps):
        dt_j = dt_now + timedelta(seconds=float(j) * float(step_seconds))
        low_j, high_j = comfort_bounds(dt_j)
        lows.append(low_j)
        highs.append(high_j)

    return np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)


def make_future_summary_features(
    i,
    dt_now,
    horizon_steps,
    tout_series_C,
    qsol_series,
    price_series,
):
    """
    Build future 6-hour summary features for target-temperature NN.

    Includes:
        - future outdoor temperature statistics
        - future solar radiation statistics
        - future electricity price statistics
        - future comfort-bound statistics
    """
    tout_win = get_future_window(tout_series_C, i, horizon_steps)
    qsol_win = get_future_window(qsol_series, i, horizon_steps)
    price_win = get_future_window(price_series, i, horizon_steps)

    tlow_win, thigh_win = future_comfort_bounds(
        dt_now=dt_now,
        horizon_steps=horizon_steps,
        step_seconds=STEP,
    )

    return {
        "Tout_mean_6h": float(np.mean(tout_win)),
        "Tout_max_6h": float(np.max(tout_win)),

        "Qsol_mean_6h": float(np.mean(qsol_win)),
        "Qsol_max_6h": float(np.max(qsol_win)),
        "Qsol_sum_6h": float(np.sum(qsol_win) * STEP / 3600.0),

        "price_mean_6h": float(np.mean(price_win)),
        "price_max_6h": float(np.max(price_win)),
        "price_min_6h": float(np.min(price_win)),

        "Tlow_mean_6h": float(np.mean(tlow_win)),
        "Tlow_max_6h": float(np.max(tlow_win)),

        "Thigh_mean_6h": float(np.mean(thigh_win)),
        "Thigh_min_6h": float(np.min(thigh_win)),
    }


# ============================================================
# 9. 画图函数
# ============================================================

def run_one(
    scenario: str,
    out_dir: str,
    variant_name: str = "full",
    use_do: bool = True,
    use_tcn: bool = True,
    use_gru: bool = True,
    use_gate: bool = True,
    seed: int = DEFAULT_SEED,
    high_mode: str = "learned",
    low_mode: str = "learned",
    use_compensation: bool = True,
    predictor_order: int = 1,
    noise_sigma: float = 0.0,
    noise_bias: float = 0.0,
    environment_seed: int = 2026,
    weather_uncertainty=None,
    kp: float = 0.40,
    ki: float = 2.0e-4,
    kd: float = 10.0,
    id_warmup_steps: int = 96,
    dist_ramp_steps: int = 96,
    forgetting: float = 0.995,
    d_lam: float = 0.10,
    k_ud: float = 0.08,       # fallback 参数：只有 gain_L 不可用时才用
    u_d_max: float = 0.20,
    kappa_d: float = 0.10,    # 新增：u_d = kappa_d*d_hat/max(L, L_min)
    L_min: float = 0.30,      # 新增：在线控制增益下界
):
    """
    PID + 在线 RC 辨识 + 扰动观测补偿。

    控制律：
        T[k+1] = ... + L_k*u[k] + d_hat[k]
        u_d    = sat(kappa_d * d_hat[k] / max(L_k, L_min))
        u_base = clip(u_PID - alpha_d * u_d, 0, 1)

    其中：
        L_k = theta_now[3]，由 RLS 在线估计，不是已知常数；
        alpha_d 在冷启动初期为 0，随后逐渐增加到 1。
    """

    if high_mode not in ("learned", "direct", "off") or low_mode not in ("learned", "direct", "off"):
        raise ValueError("Unknown residual mode")
    use_tcn = high_mode != "off"
    use_gru = low_mode != "off"
    set_random_seed(seed)
    variant_name = str(variant_name)
    os.makedirs(out_dir, exist_ok=True)

    start_day = SCENARIO_START_DAY[scenario]
    start_time = start_day * 86400.0
    warmup_period = WARMUP_DAYS * 86400.0
    steps = int(SIM_DAYS * 86400 / STEP)

    sensor = MeasurementStream(steps, environment_seed, noise_sigma, noise_bias)
    testid = select_testcase(out_dir)

    try:
        print("\n" + "=" * 90)
        print(
            f"Running ablation: variant={variant_name}, scenario={scenario}, "
            f"DO={int(use_do)}, High={high_mode}, Low={low_mode}, Gate={int(use_gate)}"
        )
        print(f"testid={testid}")
        print("=" * 90)

        set_step(testid, STEP)
        api_put(f"/scenario/{testid}", json={"electricity_price": PRICE_SCENARIO,
                  "temperature_uncertainty": weather_uncertainty,
                  "solar_uncertainty": weather_uncertainty, "seed": int(environment_seed)})
        meas = initialize(testid, start_time, warmup_period)

        # ====== forecast：电价 + 室外温度 + 太阳辐射 ======
        available_fc = get_forecast_points(testid)

        price_name = choose_name(
            available_fc,
            [
                "PriceElectricPowerHighlyDynamic",
                "PriceElectricPowerDynamic",
                "PriceElectricPowerConstant",
                "PriceElectricPower",
            ],
        )

        tout_name = choose_name(
            available_fc,
            [
                "TDryBul",
                "weaSta_reaWeaTDryBul_y",
                "TDryBul_y",
            ],
        )

        qsol_name = choose_name(
            available_fc,
            [
                "HGloHor",
                "weaSta_reaWeaHGloHor_y",
                "HGloHor_y",
            ],
        )

        if not all([price_name,tout_name,qsol_name]):
            raise RuntimeError("Required tariff/weather forecast points unavailable")
        point_names=[price_name,tout_name,qsol_name,"LowerSetp[1]","UpperSetp[1]"]
        print("[Rolling forecast points]",point_names,flush=True)

        # ====== PID 控制器 ======
        pid = PIDController(
            kp=kp,
            ki=ki,
            kd=kd,
            d_filter_alpha=0.2,
        )
        pid.reset()
        # ====== 高层目标温度纯 TCN 残差网络 ======
        # target_seq_len = 24 表示 24 * 900 s = 6 h 的历史动态。
        target_seq_len = 24
        target_nn = target_optimizer = None
        if high_mode == "learned":
            target_nn = TargetTempResidualNN(
                input_dim=23,
                seq_input_dim=23,
                hidden_dim=64,
                delta_t_max=0.8,
                tcn_channels=(32, 64),
                kernel_size=3,
                dropout=0.05,
            )
    
            target_optimizer = optim.Adam(
                target_nn.parameters(),
                lr=1.0e-3,
                weight_decay=1.0e-5,
            )
    
        target_buffer = SimpleReplayBuffer(maxlen=50000)
        target_feature_history = deque(maxlen=target_seq_len)

        # ====== 低层动作补偿纯 GRU 残差网络 ======
        # action_seq_len = 12 表示 12 * 900 s = 3 h 的短时动作响应历史。
        action_seq_len = 12
        action_nn = action_optimizer = None
        if low_mode == "learned":
            action_nn = ActionResidualNN(
                input_dim=16,
                seq_input_dim=16,
                hidden_dim=64,
                gru_hidden_dim=64,
                delta_u_max=0.08,
            )
    
            action_optimizer = optim.Adam(
                action_nn.parameters(),
                lr=3.0e-4,
                weight_decay=1.0e-5,
            )
    
        action_buffer = SimpleReplayBuffer(maxlen=50000)
        action_feature_history = deque(maxlen=action_seq_len)

        # 上一时刻最终动作，用于动作平滑和动作 teacher
        u_prev = 0.0


        # ====== 在线 RC 辨识器 + 扰动观测器 ======
        rc_id = OnlineARXPredictor(order=predictor_order, forgetting=forgetting)

        dist_obs = DisturbanceObserver(
            lam=d_lam,
            d_min=-2.0,
            d_max=2.0,
            k_ud=k_ud,
            u_d_max=u_d_max,
            kappa_d=kappa_d,
            L_min=L_min,
        )

        base_date = datetime(2019, 1, 1)
        logs = []

        # 上一步缓存
        phi_prev = None
        pred_base_prev = None
        pred_with_d_prev = None

        target_ready = action_ready = None
        valid_action_count = 0
        controller_started = time.perf_counter()
        for i in range(steps):
            step_started = time.perf_counter()
            t_now = float(meas["time"])
            dt_now = base_date + timedelta(seconds=t_now)

            # =====================================================
            # 1. 当前测量
            # =====================================================
            true_tz_c=float(meas["reaTZon_y"])-273.15
            observed=sensor.observe(meas,i)
            tz_c=float(observed["reaTZon_y"])-273.15
            tout_c=float(observed["weaSta_reaWeaTDryBul_y"])-273.15
            qsol=float(observed.get("weaSta_reaWeaHGloHor_y",0.))
            fc=put_forecast(testid,point_names=point_names,horizon=24*STEP,interval=STEP)
            future_features,price,t_low_now,t_high_now=forecast_features(
                fc,price_name,tout_name,qsol_name,t_now,STEP)

            p_hp = float(meas.get("reaPHeaPum_y", 0.0))
            p_fan = float(meas.get("reaPFan_y", 0.0))
            p_pum = float(meas.get("reaPPumEmi_y", 0.0))
            cop = float(meas.get("reaCOP_y", 1.0))

            # =====================================================
            # 2. 在线 RC 模型更新 + 扰动估计
            # =====================================================
            rls_pred = np.nan
            rls_err = np.nan
            pred_error_base = np.nan
            pred_error_with_d = np.nan

            if phi_prev is not None:
                pred_error_base = tz_c - pred_base_prev
                pred_error_with_d = tz_c - pred_with_d_prev

                rls_pred, rls_err, theta_now = rc_id.update(phi_prev, tz_c)

                if use_do:
                    d_hat = dist_obs.update(pred_error_base)
                else:
                    # w/o DO：保留在线模型辨识，但关闭扰动估计和动作补偿。
                    dist_obs.d_hat = 0.0
                    d_hat = 0.0
            else:
                theta_now = rc_id.theta.copy()
                if use_do:
                    d_hat = dist_obs.d_hat
                else:
                    dist_obs.d_hat = 0.0
                    d_hat = 0.0

            # =====================================================
            # 3. 生成设定温度：沿用你的 baseline 逻辑
            # =====================================================
            T_heat_sp_K = t_low_now + 273.15
            occ = t_low_now >= 21.0
            hour_now = dt_now.hour + dt_now.minute / 60.0

            # 1. 规则老师基础目标
            T_rule_C = rule_teacher_target_temperature(
                occ=occ,
                T_heat_sp_K=T_heat_sp_K,
                Tz_C=tz_c,
                Tout_C=tout_c,
                Qsol=qsol,
                price=price,
            )

            # 2.2 高层老师生成训练标签：当前状态 + 未来预测状态
            delta_T_teacher = high_level_teacher_delta_T(
                Tz_C=tz_c,
                T_rule_C=T_rule_C,
                Tout_C=tout_c,
                Qsol=qsol,
                price=price,
                occ=occ,
                future_features=future_features,
            )

            # 3. 构造 NN 输入
            x_target = make_target_nn_features(
                Tz_C=tz_c,
                T_rule_C=T_rule_C,
                Tout_C=tout_c,
                Qsol=qsol,
                price=price,
                occ=occ,
                d_hat=dist_obs.d_hat,
                theta_g=theta_now[3],
                hour=hour_now,
                future_features=future_features,
            )

            target_feature_history.append(x_target)
            target_train_loss = np.nan
            delta_T_nn = 0.0
            if high_mode == "learned":
                seq = build_fixed_length_sequence(target_feature_history, target_seq_len, 23)
                target_buffer.add(x_target, delta_T_teacher, seq)
                target_train_loss = train_target_nn(target_nn, target_optimizer, target_buffer,
                                                   batch_size=CONFIG.batch_size,
                                                   train_steps=CONFIG.target_train_steps)
                if target_train_loss is not None and np.isfinite(target_train_loss):
                    if target_ready is None:
                        target_ready = i
                    delta_T_nn = predict_delta_T_nn(target_nn, x_target, seq)
            elif high_mode == "direct":
                delta_T_nn = delta_T_teacher
                # Match the learned branch's earliest possible readiness, without training.
                if i + 1 >= CONFIG.batch_size and target_ready is None:
                    target_ready = i
            beta_t = participation(i, CONFIG.target_warmup, CONFIG.target_ramp, target_ready)

            Tsp_C = combine_rule_and_nn_target(
                T_rule_C=T_rule_C,
                delta_T_nn_C=delta_T_nn,
                beta=beta_t,
            )

            # =====================================================
            # Shared state-dependent protections; no evaluation-window label.
            Tsp_C = protect_target(Tsp_C, tz_c, dt_now, future_features)

            T_set_oper_K = Tsp_C + 273.15

            # =====================================================
            # 4. PID 输出
            # =====================================================
            e = Tsp_C - tz_c
            u_pid, u_pid_raw, I_term, D_term = pid.step(e, STEP)

            # =====================================================
            # 5. 扰动补偿 + 动作 NN 补偿
            # =====================================================

            # ---------- 5.1 扰动补偿 ----------

            L_gain = float(theta_now[3])
            if use_do:
                u_d = dist_obs.compensation_u(gain_L=L_gain)

                if i < id_warmup_steps:
                    alpha_d = 0.0
                else:
                    alpha_d = min(
                        1.0,
                        (i - id_warmup_steps) / max(1, dist_ramp_steps)
                    )
            else:
                # w/o DO：关闭扰动补偿通道；在线 RC 模型仍保留用于 teacher 预测。
                u_d = 0.0
                alpha_d = 0.0
                dist_obs.last_u_d_raw = 0.0
                dist_obs.last_L_eff = np.nan
                dist_obs.last_comp_mode = "disabled"

            # PID + 扰动补偿后的基础动作
            if not use_compensation:
                alpha_d = 0.0
            u_base_raw = u_pid - alpha_d * u_d
            u_base = float(np.clip(u_base_raw, 0.0, 1.0))

            # 当前舒适上下界
            margin_low = tz_c - t_low_now
            margin_high = t_high_now - tz_c

            # ---------- 5.2--5.9 低层 GRU 动作残差 ----------
            if use_gru:
                x_action = make_action_nn_features(
                    Tz_C=tz_c,
                    Tsp_C=Tsp_C,
                    T_rule_C=T_rule_C,
                    Tout_C=tout_c,
                    Qsol=qsol,
                    price=price,
                    occ=occ,
                    d_hat=dist_obs.d_hat,
                    theta_g=theta_now[3],
                    u_pid=u_pid,
                    u_d=u_d,
                    u_base=u_base,
                    u_prev=u_prev,
                    hour=hour_now,
                )

                delta_u_teacher = action_teacher_delta_u_by_grid(
                        Tz_C=tz_c,
                    Tsp_C=Tsp_C,
                    T_low_C=t_low_now,
                    T_high_C=t_high_now,
                    Tout_C=tout_c,
                    Qsol=qsol,
                    price=price,
                    d_hat=dist_obs.d_hat,
                    theta=theta_now,
                    u_base=u_base,
                    u_prev=u_prev,
                    delta_u_max=0.08,
                    n_grid=21,
                    predictor=rc_id,
                )

                valid_action_sample = not (u_base <= 0.02 and delta_u_teacher < 0)
                if use_gate and min(margin_low, margin_high) < CONFIG.sample_margin:
                    valid_action_sample = False
                action_feature_history.append(x_action)
                x_action_seq = build_fixed_length_sequence(action_feature_history, action_seq_len, 16)
                valid_action_count += int(valid_action_sample)
                action_train_loss = np.nan
                delta_u_nn_raw = 0.0
                if low_mode == "learned":
                    if valid_action_sample:
                        action_buffer.add(x_action, delta_u_teacher, x_action_seq)
                    action_train_loss = train_action_nn(action_nn, action_optimizer, action_buffer,
                                                       batch_size=CONFIG.batch_size, train_steps=1)
                    if action_train_loss is not None and np.isfinite(action_train_loss):
                        if action_ready is None:
                            action_ready = i
                        delta_u_nn_raw = predict_delta_u_nn(action_nn, x_action, x_action_seq)
                elif low_mode == "direct":
                    delta_u_nn_raw = delta_u_teacher
                    if valid_action_count >= CONFIG.batch_size and action_ready is None:
                        action_ready = i
                alpha_u_raw = participation(i, CONFIG.action_warmup, CONFIG.action_ramp, action_ready)
                delta_u_nn = (gate_action(delta_u_nn_raw, margin_low, margin_high,
                                          u_base, tz_c, future_features)
                              if use_gate else delta_u_nn_raw)
                action_gate = float(not (tz_c > 23.5 and u_base <= .02)) if use_gate else 1.0
                alpha_u = alpha_u_raw * action_gate
            else:
                delta_u_teacher = delta_u_nn_raw = delta_u_nn = 0.0
                alpha_u_raw = alpha_u = action_gate = 0.0
                action_train_loss = np.nan
                valid_action_sample = False

            # ---------- 5.10 最终动作 ----------
            u_final_raw = u_base + alpha_u * delta_u_nn
            u_final = float(np.clip(u_final_raw, 0.0, 1.0))

            # =====================================================
            # 6. 联动 fan / pump
            # =====================================================
            u_hp = float(np.clip(u_final, 0.0, 1.0))
            if u_hp > 0.05:
                # 风机和泵最小开度 0.2
                u_fan = max(0.2, u_hp)
                u_pum = max(0.2, u_hp)
            else:
                u_hp = 0.0
                u_fan = 0.0
                u_pum = 0.0

            inputs = {
                "oveTSet_activate": 1,
                "oveTSet_u": float(T_set_oper_K),

                "oveHeaPumY_activate": 1,
                "oveHeaPumY_u": float(u_hp),

                "oveFan_activate": 1,
                "oveFan_u": float(u_fan),

                "ovePum_activate": 1,
                "ovePum_u": float(u_pum),
            }

            # =====================================================
            # 7. 当前时刻预测下一步，供下一时刻计算误差
            # =====================================================
            phi_now = rc_id.make_phi(
                Tz_C=tz_c,
                Tout_C=tout_c,
                Qsol_Wm2=qsol,
                u_hp=u_hp,
            )

            pred_base_now = rc_id.predict(phi_now)
            pred_with_d_now = pred_base_now + dist_obs.d_hat

            # =====================================================
            # 8. 推进仿真
            # =====================================================
            controller_seconds = time.perf_counter() - step_started
            meas_next = advance(testid, inputs)
            if not np.isfinite(float(meas_next["reaTZon_y"])):
                raise RuntimeError("Non-finite temperature")
            if abs(float(meas_next["time"]) - t_now - STEP) > 1e-3:
                raise RuntimeError("Unexpected simulator time step")

            # =====================================================
            # 9. 日志
            # =====================================================
            t_low, t_high = t_low_now, t_high_now

            violation_low = max(0.0, t_low - tz_c)
            violation_high = max(0.0, tz_c - t_high)
            violation = violation_low + violation_high

            logs.append({
                "variant": variant_name,
                "seed": seed,
                "high_mode": high_mode,
                "low_mode": low_mode,
                "target_ready_step": target_ready,
                "action_ready_step": action_ready,
                "controller_seconds": controller_seconds,
                "scenario": scenario,
                "use_do": int(use_do),
                "use_tcn": int(high_mode=="learned"),
                "use_gru": int(low_mode=="learned"),
                "use_gate": int(use_gate),

                "T_rule_C": float(T_rule_C),
                "delta_T_teacher": float(delta_T_teacher),
                "delta_T_nn": float(delta_T_nn),
                "beta_t": float(beta_t),
                "target_train_loss": float(target_train_loss) if target_train_loss is not None else np.nan,
                "Tsp_minus_rule": float(Tsp_C - T_rule_C),

                "Tout_mean_6h": float(future_features["Tout_mean_6h"]),
                "Tout_max_6h": float(future_features["Tout_max_6h"]),
                "Qsol_mean_6h": float(future_features["Qsol_mean_6h"]),
                "Qsol_max_6h": float(future_features["Qsol_max_6h"]),
                "Qsol_sum_6h": float(future_features["Qsol_sum_6h"]),
                "price_mean_6h": float(future_features["price_mean_6h"]),
                "price_max_6h": float(future_features["price_max_6h"]),
                "price_min_6h": float(future_features["price_min_6h"]),
                "Tlow_mean_6h": float(future_features["Tlow_mean_6h"]),
                "Tlow_max_6h": float(future_features["Tlow_max_6h"]),
                "Thigh_mean_6h": float(future_features["Thigh_mean_6h"]),
                "Thigh_min_6h": float(future_features["Thigh_min_6h"]),

                "time_s": t_now,
                "datetime": dt_now.isoformat(),
                "occupied": int(occ),

                "Tz_C": tz_c,
                "Tz_true_C": true_tz_c,
                "measurement_noise_C": tz_c-true_tz_c,
                "violation_true_C": max(0.,t_low-true_tz_c)+max(0.,true_tz_c-t_high),
                "pred_error_base_true": true_tz_c-pred_base_prev if pred_base_prev is not None else np.nan,
                "pred_error_corrected_true": true_tz_c-pred_with_d_prev if pred_with_d_prev is not None else np.nan,
                "predictor_order": predictor_order,
                "noise_sigma": noise_sigma,
                "environment_seed": environment_seed,
                "theta_lag_dT": float(theta_now[5]) if predictor_order==2 else 0.,
                "theta_lag_u": float(theta_now[6]) if predictor_order==2 else 0.,
                "T_low": t_low,
                "T_high": t_high,
                "violation": violation,
                "violation_low": violation_low,
                "violation_high": violation_high,

                "reaTSetHea_K": float(T_heat_sp_K),
                "Tset_oper_K": float(T_set_oper_K),
                "Tsp_C": float(Tsp_C),

                "T_out_C": tout_c,
                "Q_sol": qsol,
                "price": price,

                "error_T": e,
                "I_term": I_term,
                "D_term": D_term,

                "u_pid_raw": float(u_pid_raw),
                "u_pid": float(u_pid),
                "u_d": float(u_d),
                "u_d_raw": float(dist_obs.last_u_d_raw),
                "L_gain": float(theta_now[3]),
                "L_eff": float(dist_obs.last_L_eff) if np.isfinite(dist_obs.last_L_eff) else np.nan,
                "kappa_d": float(dist_obs.kappa_d),
                "L_min": float(dist_obs.L_min),
                "dist_comp_mode": dist_obs.last_comp_mode,
                "alpha_d": float(alpha_d),
                "u_final_raw": float(u_final_raw),
                "u_hp": float(u_hp),
                "u_fan": float(u_fan),
                "u_pum": float(u_pum),

                "u_base_raw": float(u_base_raw),
                "u_base": float(u_base),

                "delta_u_teacher": float(delta_u_teacher),
                "delta_u_nn_raw": float(delta_u_nn_raw),
                "delta_u_nn": float(delta_u_nn),
                "alpha_u_raw": float(alpha_u_raw),
                "alpha_u": float(alpha_u),
                "action_gate": float(action_gate),
                "action_train_loss": float(action_train_loss) if action_train_loss is not None else np.nan,
                "valid_action_sample": int(valid_action_sample),
                "u_prev": float(u_prev),


                "d_hat": float(dist_obs.d_hat),
                "pred_base_now": float(pred_base_now),
                "pred_with_d_now": float(pred_with_d_now),

                "pred_error_base": float(pred_error_base)
                if np.isfinite(pred_error_base) else np.nan,

                "pred_error_with_d": float(pred_error_with_d)
                if np.isfinite(pred_error_with_d) else np.nan,

                "rls_pred": float(rls_pred)
                if np.isfinite(rls_pred) else np.nan,

                "rls_err": float(rls_err)
                if np.isfinite(rls_err) else np.nan,

                "theta_a": float(theta_now[0]),
                "theta_b": float(theta_now[1]),
                "theta_c": float(theta_now[2]),
                "theta_g": float(theta_now[3]),
                "theta_bias": float(theta_now[4]),

                "P_hp": p_hp,
                "P_fan": p_fan,
                "P_pum": p_pum,
                "COP": cop,
            })

            if i % 96 == 0 or i == steps - 1:
                pd.DataFrame(logs).to_csv(os.path.join(out_dir, "trajectory.partial.csv"), index=False)
                with open(os.path.join(out_dir, "progress.json"), "w") as f:
                    json.dump({"step": i+1, "total_steps": steps, "testid": testid,
                               "elapsed_seconds": time.perf_counter()-controller_started}, f)
                print(
                    f"[{variant_name}/{scenario}] step={i:04d}/{steps} | "
                    f"Tz={tz_c:.2f}°C | Tsp={Tsp_C:.2f}°C | "
                    f"u_pid={u_pid:.3f} | u_d={u_d:.3f} | "
                    f"L={theta_now[3]:.3f} | alpha_d={alpha_d:.2f} | u={u_final:.3f} | "
                    f"d_hat={dist_obs.d_hat:.3f} | "
                    f"theta=[{theta_now[0]:.3f}, {theta_now[1]:.3f}, "
                    f"{theta_now[2]:.3f}, {theta_now[3]:.3f}, {theta_now[4]:.3f}]"
                )

            # =====================================================
            # 10. 更新缓存
            # =====================================================
            rc_id.commit(tz_c,u_hp)
            u_prev = float(u_hp)

            phi_prev = phi_now
            pred_base_prev = pred_base_now
            pred_with_d_prev = pred_with_d_now
            meas = meas_next

        # =========================================================
        # 11. 保存结果
        # =========================================================
        df = pd.DataFrame(logs)
        csv_path = os.path.join(out_dir, f"{variant_name}_{scenario}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        kpi = get_kpi(testid)
        kpi_path = os.path.join(out_dir, f"{variant_name}_{scenario}_kpi.json")
        with open(kpi_path, "w", encoding="utf-8") as f:
            json.dump(kpi, f, indent=2, ensure_ascii=False)

        required = ["cost_tot", "ener_tot", "emis_tot", "tdis_tot"]
        if not all(k in kpi and np.isfinite(kpi[k]) for k in required):
            raise RuntimeError("Missing or non-finite official KPI")
        if len(df) != steps:
            raise RuntimeError("Incomplete trajectory")
        print(f"[OK] {variant_name}/{scenario}: {kpi}", flush=True)

        return {
            "variant": variant_name,
            "scenario": scenario,
            "use_do": int(use_do),
            "use_tcn": int(high_mode=="learned"),
            "use_gru": int(low_mode=="learned"),
            "use_gate": int(use_gate),
            "cost": kpi_get(kpi, ["cost_tot", "cost"]),
            "emissions": kpi_get(kpi, ["emis_tot", "emissions"]),
            "energy": kpi_get(kpi, ["ener_tot", "energy"]),
            "discomfort": kpi_get(kpi, ["tdis_tot", "tdis", "discomfort"]),
            "csv_path": csv_path,
            "kpi_path": kpi_path,
            "steps": steps,
            "target_ready_step": target_ready,
            "action_ready_step": action_ready,
        }

    finally:
        stop(testid)


# ============================================================
