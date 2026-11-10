# OD-SHRLC: online disturbance-aware hierarchical heating control

Research code accompanying **Disturbance-Aware Online Hierarchical Control for Building Heating without Historical Operating Data under Dynamic Electricity Pricing**.

The controller combines rule/PID feedback, online ARX identification, equivalent-disturbance compensation, a target TCN and an action GRU. Historical-data-free initialization uses no imported building trajectory, building-fitted predictor or pretrained neural weights. Engineering rules, initial coefficients and participation settings are supplied before deployment. Comfort is evaluated empirically; command bounds and margin rules do not establish a formal temperature-safety guarantee. Direct and filtered teachers are competitive alternatives; the code does not assume that both networks improve every objective.

## Install

Use Python 3.12 (the evaluation records report 3.12.8). From this folder:

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The recorded runs used PyTorch 2.9.0+cpu. The PyTorch package index may supply a different accelerator build; select the appropriate CPU build for matching conditions. Pinned dependencies document the evaluated environment and are not a claim of compatibility with every operating system.

Run BOPTEST separately, using the official [v0.8.0 release](https://github.com/ibpsa/project1-boptest/releases/tag/v0.8.0) and its installation instructions. The main test case is `bestest_hydronic_heat_pump`. The apartment adapter requires `twozone_apartment_hydronic`. BOPTEST/Modelica models, Docker images and third-party packages are not redistributed here.

## Quick start

```sh
python run.py --dry-run
python run.py --scenario typical --variant full --seed 2026 --output outputs/typical_2026
python run.py --scenario peak --variant direct_l --output outputs/direct_l_peak
python scripts/summarize_results.py outputs --output outputs/summary.csv
```

The endpoint defaults to `http://127.0.0.1:8080`. Use `--base-url` or the `BOPTEST_BASE_URL` environment variable for the current code. `.env.example` is a reference only; no dotenv loader is used. `--dry-run` prints a plan without importing the controller or contacting a server. An actual run defaults to 14 days, 900-second decisions and seven days of plant conditioning before fresh controller initialization. Use a new output folder for every single run. `python controller.py` alone does not launch a case; use `run.py`.

## Repository contents

| Location | Purpose |
|---|---|
| Root Python modules | Current terminology-aligned controller; `comfort_schedule_offset = 0.5` |
| `run.py` | Single-run CLI with an explicit RLS forgetting factor of 0.998 |
| `reproduction/revision_v2/` | Byte-preserved source used for main, higher-order, date, noise and forecast evaluations |
| `reproduction/revision_v3/` | Byte-preserved extension for sensitivity, low-pass action teachers and auxiliary mapping |
| `reproduction/revision_v3/delay_experiment/` | Original one-step measurement-delay runner; prepares its source locally |
| `provenance/delay_source/` | Prepared source snapshot from the delay evaluation |
| `provenance/source_manifest.json` | SHA-256 values of retained source files |
| `scripts/` | Reproduction launchers and a portable KPI summarizer |
| `tests/` | Offline checks; no BOPTEST calls |

See [reproduction instructions](docs/REPRODUCING.md), [method and configuration notes](docs/METHOD.md) and [upload instructions](UPLOAD_GUIDE_zh.md).

## Scope and provenance

The root version changes the offset name and makes the server URL configurable. It is not byte-identical to the executed snapshots. Archived versions intentionally retain their original names and fixed localhost endpoint. No archived result is relabelled as a run of the newer code. See the source manifest and change notes.

This package covers the proposed controller and its internal experiments. It does **not** include recovered third-party PI/MPC/TD3/DQN projects, their calibrated models/checkpoints, all paper-figure reconstruction scripts, full trajectories, field data, manuscript files or reviewer correspondence. It is not a complete bundle for regenerating every published comparison figure. New results are written locally and excluded by `.gitignore`. No pretrained network weights are required for the proposed controller.

## Citation and license

Use `CITATION.cff` to cite the associated manuscript. No publication DOI or repository URL has been assigned in this prepared package. **A software license has not yet been selected by the authors.** No MIT/Apache or other reuse license is assigned by this preparation step.
