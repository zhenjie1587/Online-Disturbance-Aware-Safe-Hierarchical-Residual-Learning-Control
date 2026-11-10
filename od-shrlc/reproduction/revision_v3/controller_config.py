"""Fixed exploratory configuration. No selection on reported test performance."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ControllerConfig:
    version: str = "unified_signals_v2"
    batch_size: int = 64
    target_train_steps: int = 1
    target_warmup: int = 96
    target_ramp: int = 192
    action_warmup: int = 288
    action_ramp: int = 192
    sample_margin: float = 0.25
    directional_margin: float = 0.45
    hard_margin: float = 0.20
    target_bound: float = 0.8
    action_bound: float = 0.08
    occupied_offset: float = 0.5
    relaxed_target: float = 20.8
    rule_min: float = 20.0
    rule_max: float = 22.2
    target_min: float = 19.8
    target_max: float = 22.4
    teacher_low_weight: float = 160.0
    teacher_high_weight: float = 180.0
    teacher_track_weight: float = 1.0
    teacher_energy_weight: float = 0.02
    teacher_smooth_weight: float = 0.30

    def as_dict(self):
        return asdict(self)


CONFIG = ControllerConfig()


def participation(step, warmup, ramp, ready_step):
    """Start ramp after both fixed observation period and readiness are met."""
    if ready_step is None:
        return 0.0
    start = max(warmup, ready_step)
    return min(1.0, max(0.0, (step - start) / max(1, ramp)))
