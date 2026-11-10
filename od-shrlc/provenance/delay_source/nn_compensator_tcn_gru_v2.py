# nn_compensator_tcn_gru_v2.py
# -*- coding: utf-8 -*-
"""
Pure temporal neural-network compensators for building heating control.

This file contains ONLY:
    - high-level target residual TCN model
    - low-level action residual GRU model
    - temporal replay buffer
    - feature construction for the two neural networks
    - online supervised training functions
    - participation schedules
    - prediction wrappers

It intentionally does NOT contain rule-teacher logic such as:
    - high_level_teacher_delta_T
    - action_teacher_delta_u_by_grid
    - occupancy rules
    - comfort-bound rules

Those rule/teacher functions should be placed in rule_teacher_tcn_gru_v2.py.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. Temporal utilities
# ============================================================

def build_fixed_length_sequence(history, seq_len: int, feature_dim: int = None):
    """Build a fixed-length temporal sequence from a list/deque history."""
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    items = [np.asarray(x, dtype=np.float32) for x in list(history)]

    if len(items) == 0:
        if feature_dim is None:
            raise ValueError("feature_dim must be provided when history is empty")
        return np.zeros((seq_len, int(feature_dim)), dtype=np.float32)

    if len(items) >= seq_len:
        seq = items[-seq_len:]
    else:
        seq = [items[0]] * (seq_len - len(items)) + items

    return np.stack(seq, axis=0).astype(np.float32)


# ============================================================
# 2. Replay buffer for temporal supervised learning
# ============================================================

class SimpleReplayBuffer:
    """
    Replay buffer for online supervised learning.

    Stored sample:
        x_now : current feature vector, shape=(input_dim,)
        x_seq : temporal feature sequence, shape=(seq_len, input_dim)
        y     : teacher residual label, shape=(1,)
    """

    def __init__(self, maxlen: int = 50000):
        self.buffer = deque(maxlen=maxlen)

    def add(self, x, y, x_seq=None):
        x_now = np.asarray(x, dtype=np.float32)
        if x_seq is None:
            x_seq = x_now[None, :]
        else:
            x_seq = np.asarray(x_seq, dtype=np.float32)
        y = np.asarray([y], dtype=np.float32)
        self.buffer.append((x_now, x_seq, y))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        xs = np.stack([item[0] for item in batch], axis=0)
        xseqs = np.stack([item[1] for item in batch], axis=0)
        ys = np.stack([item[2] for item in batch], axis=0)
        return xs, xseqs, ys


# ============================================================
# 3. High-level target residual network: pure TCN
# ============================================================

class Chomp1d(nn.Module):
    """Remove right-side padding to keep causal convolution length."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """A lightweight residual TCN block."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.05):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.out_relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(y + res)


class TargetTempResidualNN(nn.Module):
    """
    High-level target-temperature residual network: pure TCN.

    Input:
        x_current : kept for API compatibility, shape=(B, input_dim)
        x_seq     : temporal sequence, shape=(B, L, seq_input_dim)

    Output:
        Delta_T_NN in Celsius, bounded by [-delta_t_max, delta_t_max]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,          # kept only for old constructor compatibility
        delta_t_max: float = 0.8,
        seq_input_dim: int = None,
        tcn_channels=(32, 64),
        kernel_size: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_input_dim = int(seq_input_dim or input_dim)
        self.delta_t_max = float(delta_t_max)

        blocks = []
        in_ch = self.seq_input_dim
        for i, out_ch in enumerate(tcn_channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.out_conv = nn.Conv1d(tcn_channels[-1], 1, kernel_size=1)
        self.out_act = nn.Tanh()

    def forward(self, x_current, x_seq=None):
        if x_seq is None:
            x_seq = x_current.unsqueeze(1)

        z = self.tcn(x_seq.transpose(1, 2))
        y_seq = self.out_conv(z)
        y_last = y_seq[:, :, -1]
        return self.delta_t_max * self.out_act(y_last)


# ============================================================
# 4. Low-level action residual network: pure GRU
# ============================================================

class ActionResidualNN(nn.Module):
    """
    Low-level heat-pump action residual network: pure GRU.

    Input:
        x_current : kept for API compatibility, shape=(B, input_dim)
        x_seq     : temporal sequence, shape=(B, L, seq_input_dim)

    Output:
        Delta_u_NN, bounded by [-delta_u_max, delta_u_max]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,          # kept only for old constructor compatibility
        delta_u_max: float = 0.08,
        seq_input_dim: int = None,
        gru_hidden_dim: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.seq_input_dim = int(seq_input_dim or input_dim)
        self.delta_u_max = float(delta_u_max)

        self.gru = nn.GRU(
            input_size=self.seq_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.out_conv = nn.Conv1d(gru_hidden_dim, 1, kernel_size=1)
        self.out_act = nn.Tanh()

    def forward(self, x_current, x_seq=None):
        if x_seq is None:
            x_seq = x_current.unsqueeze(1)

        z_seq, _ = self.gru(x_seq)
        y_seq = self.out_conv(z_seq.transpose(1, 2))
        y_last = y_seq[:, :, -1]
        return self.delta_u_max * self.out_act(y_last)


# ============================================================
# 5. Feature construction for neural networks
# ============================================================

def default_future_features(
    Tout_C: float,
    Qsol: float,
    price: float,
    T_low_C: float = 21.0,
    T_high_C: float = 24.0,
):
    """Fallback future-feature dictionary using current values."""
    return {
        "Tout_mean_6h": float(Tout_C),
        "Tout_max_6h": float(Tout_C),
        "Qsol_mean_6h": float(Qsol),
        "Qsol_max_6h": float(Qsol),
        "Qsol_sum_6h": float(Qsol) * 6.0,
        "price_mean_6h": float(price),
        "price_max_6h": float(price),
        "price_min_6h": float(price),
        "Tlow_mean_6h": float(T_low_C),
        "Tlow_max_6h": float(T_low_C),
        "Thigh_mean_6h": float(T_high_C),
        "Thigh_min_6h": float(T_high_C),
    }


def make_target_nn_features(
    Tz_C: float,
    T_rule_C: float,
    Tout_C: float,
    Qsol: float,
    price: float,
    occ: bool,
    d_hat: float,
    theta_g: float,
    hour: float,
    future_features: dict = None,
):
    """Construct normalized features for the high-level target TCN."""
    hour_rad = 2.0 * np.pi * float(hour) / 24.0

    if future_features is None:
        future_features = default_future_features(
            Tout_C=Tout_C,
            Qsol=Qsol,
            price=price,
            T_low_C=21.0,
            T_high_C=24.0,
        )

    x = np.array([
        (float(Tz_C) - 21.0) / 5.0,
        (float(T_rule_C) - 21.0) / 5.0,
        (float(Tz_C) - float(T_rule_C)) / 3.0,
        (float(Tout_C) - 10.0) / 15.0,
        float(Qsol) / 800.0,
        (float(price) - 0.25) / 0.15,
        1.0 if occ else 0.0,
        float(d_hat) / 2.0,
        float(theta_g) / 1.0,
        np.sin(hour_rad),
        np.cos(hour_rad),
        (float(future_features["Tout_mean_6h"]) - 10.0) / 15.0,
        (float(future_features["Tout_max_6h"]) - 10.0) / 15.0,
        float(future_features["Qsol_mean_6h"]) / 800.0,
        float(future_features["Qsol_max_6h"]) / 800.0,
        float(future_features["Qsol_sum_6h"]) / 4800.0,
        (float(future_features["price_mean_6h"]) - 0.25) / 0.15,
        (float(future_features["price_max_6h"]) - 0.25) / 0.15,
        (float(future_features["price_min_6h"]) - 0.25) / 0.15,
        (float(future_features["Tlow_mean_6h"]) - 21.0) / 5.0,
        (float(future_features["Tlow_max_6h"]) - 21.0) / 5.0,
        (float(future_features["Thigh_mean_6h"]) - 24.0) / 6.0,
        (float(future_features["Thigh_min_6h"]) - 24.0) / 6.0,
    ], dtype=np.float32)

    return x


def make_action_nn_features(
    Tz_C: float,
    Tsp_C: float,
    T_rule_C: float,
    Tout_C: float,
    Qsol: float,
    price: float,
    occ: bool,
    d_hat: float,
    theta_g: float,
    u_pid: float,
    u_d: float,
    u_base: float,
    u_prev: float,
    hour: float,
):
    """Construct normalized features for the low-level action GRU."""
    hour_rad = 2.0 * np.pi * float(hour) / 24.0

    x = np.array([
        (float(Tz_C) - 21.0) / 5.0,
        (float(Tsp_C) - 21.0) / 5.0,
        (float(Tz_C) - float(Tsp_C)) / 3.0,
        (float(T_rule_C) - 21.0) / 5.0,
        (float(Tout_C) - 10.0) / 15.0,
        float(Qsol) / 800.0,
        (float(price) - 0.25) / 0.15,
        1.0 if occ else 0.0,
        float(d_hat) / 2.0,
        float(theta_g) / 1.0,
        float(u_pid),
        float(u_d) / 0.2,
        float(u_base),
        float(u_prev),
        np.sin(hour_rad),
        np.cos(hour_rad),
    ], dtype=np.float32)

    return x


# ============================================================
# 6. Online supervised training
# ============================================================

def train_target_nn(
    model: nn.Module,
    optimizer: optim.Optimizer,
    buffer: SimpleReplayBuffer,
    batch_size: int = 64,
    train_steps: int = 1,
    residual_reg: float = 1.0e-3,
    grad_clip_norm: float = 1.0,
):
    """Online supervised training for the target-temperature TCN."""
    if len(buffer) < batch_size:
        return None

    model.train()
    losses = []

    for _ in range(train_steps):
        xs, xseqs, ys = buffer.sample(batch_size)
        x_tensor = torch.tensor(xs, dtype=torch.float32)
        xseq_tensor = torch.tensor(xseqs, dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)

        pred = model(x_tensor, xseq_tensor)
        mse_loss = torch.mean((pred - y_tensor) ** 2)
        reg_loss = residual_reg * torch.mean(pred ** 2)
        loss = mse_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def train_action_nn(
    model: nn.Module,
    optimizer: optim.Optimizer,
    buffer: SimpleReplayBuffer,
    batch_size: int = 64,
    train_steps: int = 1,
    residual_reg: float = 1.0e-3,
    grad_clip_norm: float = 1.0,
):
    """Online supervised training for the action GRU."""
    if len(buffer) < batch_size:
        return None

    model.train()
    losses = []

    for _ in range(train_steps):
        xs, xseqs, ys = buffer.sample(batch_size)
        x_tensor = torch.tensor(xs, dtype=torch.float32)
        xseq_tensor = torch.tensor(xseqs, dtype=torch.float32)
        y_tensor = torch.tensor(ys, dtype=torch.float32)

        pred = model(x_tensor, xseq_tensor)

        mse_loss = torch.mean((pred - y_tensor) ** 2)
        reg_loss = residual_reg * torch.mean(pred ** 2)
        loss = mse_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


# ============================================================
# 7. Participation schedules and prediction wrappers
# ============================================================

def beta_schedule(
    step_index: int,
    warmup_steps: int = 96,
    ramp_steps: int = 192,
    beta_max: float = 1.0,
):
    """Participation schedule for the high-level TCN residual."""
    if step_index < warmup_steps:
        return 0.0
    beta = (step_index - warmup_steps) / max(1, ramp_steps)
    return beta_max * float(np.clip(beta, 0.0, 1.0))


def alpha_u_schedule(
    step_index: int,
    warmup_steps: int = 288,
    ramp_steps: int = 192,
    alpha_max: float = 0.2,
):
    """Participation schedule for the low-level GRU action residual."""
    if step_index < warmup_steps:
        return 0.0
    alpha = (step_index - warmup_steps) / max(1, ramp_steps)
    return alpha_max * float(np.clip(alpha, 0.0, 1.0))


def predict_delta_T_nn(model: nn.Module, x_feature: np.ndarray, x_seq: np.ndarray = None):
    """Predict Delta_T_NN from one target feature vector and optional sequence."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_feature[None, :], dtype=torch.float32)
        xseq_tensor = None if x_seq is None else torch.tensor(x_seq[None, :, :], dtype=torch.float32)
        return float(model(x_tensor, xseq_tensor).item())


def predict_delta_u_nn(model: nn.Module, x_feature: np.ndarray, x_seq: np.ndarray = None):
    """Predict Delta_u_NN from one action feature vector and optional sequence."""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_feature[None, :], dtype=torch.float32)
        xseq_tensor = None if x_seq is None else torch.tensor(x_seq[None, :, :], dtype=torch.float32)
        return float(model(x_tensor, xseq_tensor).item())


# ============================================================
# 8. Quick self-test
# ============================================================

if __name__ == "__main__":
    target_seq_len = 24
    action_seq_len = 12

    target_model = TargetTempResidualNN(input_dim=23, seq_input_dim=23, delta_t_max=0.8)
    action_model = ActionResidualNN(input_dim=16, seq_input_dim=16, delta_u_max=0.08)

    x_target = np.zeros(23, dtype=np.float32)
    x_action = np.zeros(16, dtype=np.float32)
    x_target_seq = np.zeros((target_seq_len, 23), dtype=np.float32)
    x_action_seq = np.zeros((action_seq_len, 16), dtype=np.float32)

    print("Target TCN output:", predict_delta_T_nn(target_model, x_target, x_target_seq))
    print("Action GRU output:", predict_delta_u_nn(action_model, x_action, x_action_seq))
